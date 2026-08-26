"""Scientific-integrity regression tests for the frozen revised experiment."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.data.dataset_loader import (
    REVISED_CATEGORY_COUNTS,
    REVISED_USER_BIASES,
    generate_revised_synthetic_data,
    revised_category_ids,
    split_revised_user_observations,
)
from src.evaluation.metrics import (
    evaluate_native_score_method,
    ndcg_from_order,
)
from src.evaluation.statistical_tests import (
    holm_adjust,
    percentile_bootstrap_summary,
    revised_wilcoxon_signed_rank,
)
from src.experiments.run_revised_experiment import (
    assemble_results,
    build_smoke_scientific_payload,
    canonical_json_bytes,
    hash_dataset,
    hash_split,
    load_config,
    run_replicate,
    sha256_json,
)
from src.models.collaborative_filtering import (
    ItemBasedCF,
    UserBasedCF,
    compute_item_cosine_similarity,
    compute_user_cosine_similarity,
)
from src.models.preference_vector import (
    propagate_preference_residuals,
    select_positive_similarity_neighbors,
)
from src.models.ranking import (
    deterministic_score_order,
    interactive_impression_rerank,
)
from src.utils.verify_revised_results import (
    SCIENTIFIC_FLOAT_ATOL,
    SCIENTIFIC_FLOAT_RTOL,
    compare_scientific_payloads,
    verify_results,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/revised_experiment.json"


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


def test_revised_dgp_exact_formula_and_determinism():
    seed = np.random.SeedSequence(1234)
    generated = generate_revised_synthetic_data(
        np.random.Generator(np.random.PCG64(seed))
    )
    repeat = generate_revised_synthetic_data(
        np.random.Generator(np.random.PCG64(np.random.SeedSequence(1234)))
    )
    np.testing.assert_array_equal(generated["ratings"], repeat["ratings"])

    manual_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(1234)))
    q = manual_rng.uniform(2.0, 8.0, 50)
    preferences = manual_rng.normal(0.0, 1.0, (4, 6))
    epsilon = manual_rng.normal(0.0, 0.5, (4, 50))
    category_ids, _ = revised_category_ids(REVISED_CATEGORY_COUNTS)
    biases = np.asarray(list(REVISED_USER_BIASES.values()))
    expected = np.clip(
        q[np.newaxis, :]
        + biases[:, np.newaxis]
        + preferences[:, category_ids]
        + epsilon,
        0.0,
        10.0,
    )
    np.testing.assert_array_equal(generated["ratings"], expected)
    assert generated["ratings"].dtype == np.float64
    assert generated["observed_mask"].all()
    np.testing.assert_array_equal(generated["user_biases"], [0.5, 0.3, 0.0, -0.5])


def test_revised_category_counts_total_exactly_fifty():
    category_ids, names = revised_category_ids(REVISED_CATEGORY_COUNTS)
    assert names == list(REVISED_CATEGORY_COUNTS)
    assert len(category_ids) == 50
    assert np.bincount(category_ids).tolist() == [5, 4, 11, 10, 15, 5]


def test_legitimate_zero_rating_remains_observed():
    ratings = np.asarray([[0.0, 4.0], [2.0, 5.0], [3.0, 6.0], [4.0, 7.0]])
    observed = np.ones_like(ratings, dtype=bool)
    ucf = UserBasedCF(k_neighbors=3).fit(ratings, observed_mask=observed)
    item = ItemBasedCF(k_neighbors=1).fit(
        ratings, observed_mask=observed, reference_users=[1, 2, 3]
    )
    assert ucf.observed_mask_[0, 0]
    assert item.observed_mask_[0, 0]
    assert item.predict(0, 1) == pytest.approx(0.0)


def test_seedsequence_streams_and_hashes_are_independent_and_deterministic(config):
    children = np.random.SeedSequence(config["master_seed"]).spawn(51)
    dataset_ss, split_ss, feedback_ss = children[7].spawn(3)
    assert children[7].spawn_key == (7,)
    assert dataset_ss.spawn_key == (7, 0)
    assert split_ss.spawn_key == (7, 1)
    assert feedback_ss.spawn_key == (7, 2)
    assert (
        len({tuple(x.generate_state(4)) for x in [dataset_ss, split_ss, feedback_ss]})
        == 3
    )

    first = generate_revised_synthetic_data(
        np.random.Generator(np.random.PCG64(dataset_ss))
    )
    again_children = np.random.SeedSequence(config["master_seed"]).spawn(51)
    again_dataset_ss, again_split_ss, _ = again_children[7].spawn(3)
    second = generate_revised_synthetic_data(
        np.random.Generator(np.random.PCG64(again_dataset_ss))
    )
    assert hash_dataset(first["ratings"]) == hash_dataset(second["ratings"])
    mask1, train1, test1 = split_revised_user_observations(
        first["ratings"], np.random.Generator(np.random.PCG64(split_ss))
    )
    mask2, train2, test2 = split_revised_user_observations(
        second["ratings"], np.random.Generator(np.random.PCG64(again_split_ss))
    )
    assert hash_split(train1, test1) == hash_split(train2, test2)
    assert len(train1) == 30 and len(test1) == 20
    assert mask1[0].sum() == 30 and mask2[0].sum() == 30
    assert mask1[1:].all()


def test_common_observation_ucf_similarity_ignores_masked_values():
    ratings = np.asarray([[1.0, 2.0, 999.0], [1.0, 2.0, -999.0]])
    observed = np.asarray([[True, True, False], [True, True, False]])
    similarity = compute_user_cosine_similarity(ratings, observed)
    assert similarity[0, 1] == pytest.approx(1.0)
    ratings[0, 2] = -12345.0
    np.testing.assert_array_equal(
        similarity, compute_user_cosine_similarity(ratings, observed)
    )


def test_bcd_only_item_similarity_ignores_all_user_a_values():
    ratings = np.asarray(
        [[1.0, 9.0, 3.0], [2.0, 5.0, 4.0], [3.0, 6.0, 5.0], [4.0, 7.0, 6.0]]
    )
    observed = np.ones_like(ratings, dtype=bool)
    first = compute_item_cosine_similarity(ratings, observed, [1, 2, 3])
    ratings[0] = [999.0, -999.0, 0.0]
    second = compute_item_cosine_similarity(ratings, observed, [1, 2, 3])
    np.testing.assert_array_equal(first, second)


def test_masked_user_a_test_truth_cannot_change_revised_predictions():
    rng = np.random.default_rng(8)
    ratings = rng.uniform(0.0, 10.0, (4, 8))
    observed = np.ones_like(ratings, dtype=bool)
    observed[0, 4:] = False
    changed = ratings.copy()
    changed[0, 4:] = [1000.0, -1000.0, 9999.0, -9999.0]
    predictions = []
    for matrix in [ratings, changed]:
        ucf = UserBasedCF(k_neighbors=3).fit(matrix, observed_mask=observed)
        item = ItemBasedCF(k_neighbors=3).fit(
            matrix, observed_mask=observed, reference_users=[1, 2, 3]
        )
        predictions.append(
            (
                np.asarray([ucf.predict(0, item_id) for item_id in range(4, 8)]),
                np.asarray([item.predict(0, item_id) for item_id in range(4, 8)]),
                item.item_similarity_,
            )
        )
    for index in range(3):
        np.testing.assert_allclose(predictions[0][index], predictions[1][index])


def test_native_scale_ucf_itemcf_and_hybrid_arithmetic():
    ratings = np.asarray(
        [[2.0, 4.0, 0.0], [3.0, 5.0, 7.0], [4.0, 6.0, 8.0], [5.0, 7.0, 9.0]]
    )
    observed = np.ones_like(ratings, dtype=bool)
    observed[0, 2] = False
    ucf = UserBasedCF(k_neighbors=3).fit(ratings, observed_mask=observed)
    item = ItemBasedCF(k_neighbors=3).fit(
        ratings, observed_mask=observed, reference_users=[1, 2, 3]
    )
    ucf_value = ucf.predict(0, 2)
    item_value = item.predict(0, 2)
    hybrid = (ucf_value + item_value) / 2.0
    assert 0.0 <= ucf_value <= 10.0
    assert 0.0 <= item_value <= 10.0
    assert hybrid == pytest.approx(0.5 * ucf_value + 0.5 * item_value)


def test_f1_unseen_top_ten_ties_and_unchanged_tail():
    unseen = np.arange(20, 40)
    scores = np.linspace(10.0, 1.0, 20)
    initial = deterministic_score_order(unseen, scores)
    feedback = {int(item): 5.0 for item in initial[:10]}
    result = interactive_impression_rerank(initial, 10, feedback)
    np.testing.assert_array_equal(result["displayed_ids"], initial[:10])
    np.testing.assert_array_equal(result["final_order"], initial)
    np.testing.assert_array_equal(result["final_order"][10:], initial[10:])
    assert set(result["final_order"]) == set(unseen)
    with pytest.raises(ValueError):
        interactive_impression_rerank(
            initial, 10, {int(item): 1.0 for item in initial[:9]}
        )


def test_f2_positive_top_k_deterministic_ties_and_fallback():
    row = np.asarray([0.0, 0.8, 0.8, -0.2, 0.4, 0.9])
    ids, weights = select_positive_similarity_neighbors(
        row, np.asarray([1, 2, 3, 4, 5]), 3
    )
    assert ids.tolist() == [5, 1, 2]
    assert weights.tolist() == [0.9, 0.8, 0.8]

    similarity = np.zeros((6, 6))
    fallback = propagate_preference_residuals(
        np.asarray([2.0, -1.0]), np.asarray([0, 1]), np.asarray([4, 5]), similarity, k=5
    )
    np.testing.assert_array_equal(fallback["propagated_residuals"], [0.0, 0.0])
    assert fallback["fallback_count"] == 2


def test_f2_nonzero_residual_changes_unseen_native_score():
    similarity = np.zeros((5, 5))
    similarity[3, [0, 1, 2]] = [0.9, 0.8, 0.7]
    propagated = propagate_preference_residuals(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([0, 1, 2]),
        np.asarray([3]),
        similarity,
        k=5,
    )["propagated_residuals"]
    assert propagated[0] > 0.0
    final = np.clip(np.asarray([5.0]) + propagated, 0.0, 10.0)
    assert 0.0 <= final[0] <= 10.0 and final[0] != 5.0


def test_true_leave_one_out_prediction_is_independent_of_masked_target_value():
    rng = np.random.default_rng(17)
    ratings = rng.uniform(1.0, 9.0, (4, 10))
    observed = np.ones_like(ratings, dtype=bool)
    observed[0, 6:] = False
    observed[0, 2] = False
    item_similarity = compute_item_cosine_similarity(ratings, observed, [1, 2, 3])
    values = []
    for replacement in [0.0, 10.0]:
        candidate = ratings.copy()
        candidate[0, 2] = replacement
        ucf = UserBasedCF(k_neighbors=3).fit(candidate, observed_mask=observed)
        item = ItemBasedCF(k_neighbors=3).fit(
            candidate,
            observed_mask=observed,
            item_similarity=item_similarity,
        )
        values.append((ucf.predict(0, 2) + item.predict(0, 2)) / 2.0)
    assert values[0] == pytest.approx(values[1])


def test_order_aware_linear_gain_ndcg_and_no_binary_metrics(monkeypatch):
    relevance = np.asarray([3.0, 2.0, 1.0])
    assert (
        ndcg_from_order(np.asarray([0, 1, 2]), relevance, np.asarray([0, 1, 2]), 3)
        == 1.0
    )
    assert (
        ndcg_from_order(np.asarray([2, 1, 0]), relevance, np.asarray([0, 1, 2]), 3)
        < 1.0
    )

    import src.evaluation.metrics as metrics_module

    monkeypatch.setattr(
        metrics_module,
        "precision_at_k",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError),
    )
    monkeypatch.setattr(
        metrics_module,
        "recall_at_k",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError),
    )
    metrics = evaluate_native_score_method(
        np.asarray([0, 1, 2]), np.asarray([3.0, 2.0, 1.0]), relevance
    )
    assert set(metrics) == {"rmse", "mae", "ndcg@5", "ndcg@10", "ndcg@20"}


def test_wilcoxon_replicate_shape_all_zero_and_holm():
    differences = np.linspace(-1.0, 1.0, 50)
    result = revised_wilcoxon_signed_rank(differences)
    assert result["n"] == 50 and not result["all_zero"]
    zero = revised_wilcoxon_signed_rank(np.zeros(50))
    assert zero == {"statistic": 0.0, "p_value": 1.0, "all_zero": True, "n": 50}
    holm = holm_adjust({"C1": 0.01, "C2": 0.03, "C3": 0.2}, alpha=0.05)
    assert set(holm) == {"C1", "C2", "C3"}
    assert holm["C1"]["adjusted_p"] == pytest.approx(0.03)
    assert holm["C2"]["adjusted_p"] == pytest.approx(0.06)
    assert holm["C3"]["adjusted_p"] == pytest.approx(0.2)


def test_bootstrap_ci_is_replicate_level_and_deterministic():
    values = np.arange(50, dtype=np.float64)
    first = percentile_bootstrap_summary(values, np.random.default_rng(99), 10000, 0.95)
    second = percentile_bootstrap_summary(
        values, np.random.default_rng(99), 10000, 0.95
    )
    assert first == second
    assert first["n"] == 50 and first["bootstrap_samples"] == 10000


def test_one_replicate_records_required_f1_f2_and_diagnostics(config):
    root = np.random.SeedSequence(config["master_seed"]).spawn(51)[0]
    record = run_replicate(0, root, config)
    assert len(record["train_ids"]) == 30 and len(record["test_ids"]) == 20
    assert set(record["base_methods"]) == {"ucf", "hybrid_cf"}
    for condition in record["noise_conditions"].values():
        assert set(condition["f1"]["metrics"]) == {"ndcg@5", "ndcg@10", "ndcg@20"}
        assert len(condition["f1"]["displayed_ids"]) == 10
        assert len(condition["f2"]["loo_base_predictions"]) == 30
        assert len(condition["f2"]["selected_neighbors"]) == 20
        assert all(0.0 <= value <= 10.0 for value in condition["f2"]["predictions"])
    required = {
        "rating_distribution",
        "user_means",
        "user_user_pearson",
        "user_user_cosine",
        "common_observation_counts",
        "user_a_ucf_neighbor_order",
        "test_rating_distribution",
    }
    assert required <= set(record["diagnostics"])


def test_end_to_end_smoke_payload_is_deterministic_and_has_required_schema(config):
    first = build_smoke_scientific_payload(config, REPO_ROOT, replicate_count=2)
    second = build_smoke_scientific_payload(config, REPO_ROOT, replicate_count=2)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert len(first["replicates"]) == 2
    assert first["holm_family"] == ["C1", "C2", "C3"]
    assert set(first["paired_effects"]) == {"0.0", "0.5", "1.0", "2.0"}
    for contrast in ["C1", "C2", "C3"]:
        assert len(first["paired_effects"]["0.0"][contrast]["differences"]) == 2


def test_scientific_payload_and_provenance_are_separate(config):
    payload = {"schema_version": "test", "config_hash": sha256_json(config)}
    document = assemble_results(payload, REPO_ROOT)
    assert set(document) == {
        "scientific_payload",
        "scientific_payload_sha256",
        "provenance",
    }
    assert "utc_timestamp" not in document["scientific_payload"]
    assert "git_head" not in document["scientific_payload"]
    assert document["scientific_payload_sha256"] == sha256_json(payload)
    assert isinstance(document["provenance"]["worktree_dirty"], bool)


def test_tracked_canonical_artifact_passes_static_verification():
    result_path = REPO_ROOT / "results/revised_experiment/canonical_results.json"
    assert result_path.exists()
    assert verify_results(CONFIG_PATH, result_path, reproduce=False) == []


@pytest.fixture(scope="module")
def canonical_document():
    result_path = REPO_ROOT / "results/revised_experiment/canonical_results.json"
    return json.loads(result_path.read_text(encoding="utf-8"))


def test_scientific_comparison_accepts_exact_payload(canonical_document):
    payload = canonical_document["scientific_payload"]
    comparison = compare_scientific_payloads(payload, deepcopy(payload))
    assert comparison.equivalent
    assert comparison.bitwise_match
    assert SCIENTIFIC_FLOAT_ATOL == 1e-14
    assert SCIENTIFIC_FLOAT_RTOL == 1e-11


def test_scientific_comparison_accepts_measured_float_drift(canonical_document):
    reference = canonical_document["scientific_payload"]
    candidate = deepcopy(reference)
    candidate["replicates"][0]["base_methods"]["ucf"]["predictions"][0] += 2e-15
    comparison = compare_scientific_payloads(reference, candidate)
    assert comparison.equivalent
    assert not comparison.bitwise_match


def test_scientific_comparison_rejects_above_tolerance_drift(canonical_document):
    reference = canonical_document["scientific_payload"]
    candidate = deepcopy(reference)
    candidate["replicates"][0]["base_methods"]["ucf"]["predictions"][0] += 1e-8
    comparison = compare_scientific_payloads(reference, candidate)
    assert not comparison.equivalent
    assert "float tolerance exceeded" in comparison.mismatch_details[0]


@pytest.mark.parametrize("field", ["dataset_hash", "split_hash"])
def test_scientific_comparison_rejects_identity_hash_mutation(
    canonical_document, field
):
    reference = canonical_document["scientific_payload"]
    candidate = deepcopy(reference)
    candidate["replicates"][0][field] = "0" * 64
    assert not compare_scientific_payloads(reference, candidate).equivalent


def test_scientific_comparison_rejects_f1_ranking_mutation(canonical_document):
    reference = canonical_document["scientific_payload"]
    candidate = deepcopy(reference)
    order = candidate["replicates"][0]["noise_conditions"]["0.0"]["f1"]["final_order"]
    order[0], order[1] = order[1], order[0]
    assert not compare_scientific_payloads(reference, candidate).equivalent


def test_scientific_comparison_rejects_f2_neighbor_mutation(canonical_document):
    reference = canonical_document["scientific_payload"]
    candidate = deepcopy(reference)
    neighbors = candidate["replicates"][0]["noise_conditions"]["0.0"]["f2"][
        "selected_neighbors"
    ][0]["neighbor_ids"]
    neighbors[0] = 999
    assert not compare_scientific_payloads(reference, candidate).equivalent


def test_scientific_comparison_rejects_holm_decision_mutation(canonical_document):
    reference = canonical_document["scientific_payload"]
    candidate = deepcopy(reference)
    candidate["primary_tests"]["C1"]["holm_reject"] = not candidate["primary_tests"][
        "C1"
    ]["holm_reject"]
    assert not compare_scientific_payloads(reference, candidate).equivalent


def test_scientific_comparison_requires_exact_embedded_config(canonical_document):
    reference = canonical_document["scientific_payload"]
    candidate = deepcopy(reference)
    candidate["config"]["feedback_sigmas"][0] += 2e-15
    comparison = compare_scientific_payloads(reference, candidate)
    assert not comparison.equivalent
    assert "exact invariant float mismatch" in comparison.mismatch_details[0]


def test_scientific_comparison_ignores_provenance(canonical_document):
    reference = deepcopy(canonical_document)
    candidate = deepcopy(canonical_document)
    candidate["provenance"].update(
        {
            "utc_timestamp": "2099-01-01T00:00:00+00:00",
            "git_head": "different",
            "platform": "different",
        }
    )
    comparison = compare_scientific_payloads(
        reference["scientific_payload"], candidate["scientific_payload"]
    )
    assert comparison.equivalent
    assert comparison.bitwise_match
