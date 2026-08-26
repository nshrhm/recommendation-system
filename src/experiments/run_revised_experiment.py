"""Run the frozen revised review-response experiment.

The revised path is deliberately separate from the historical experiment
runners. It uses explicit observation masks, native 0-10 scores, independent
dataset replicates, and replicate-level inference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.data.dataset_loader import (
    USER_IDS,
    generate_revised_synthetic_data,
    split_revised_user_observations,
)
from src.evaluation.metrics import (
    evaluate_native_score_method,
    evaluate_ranking_order,
)
from src.evaluation.statistical_tests import (
    holm_adjust,
    percentile_bootstrap_summary,
    revised_wilcoxon_signed_rank,
)
from src.models.collaborative_filtering import ItemBasedCF, UserBasedCF
from src.models.preference_vector import propagate_preference_residuals
from src.models.ranking import deterministic_score_order, interactive_impression_rerank


SCIENTIFIC_SOURCE_FILES = [
    "src/data/dataset_loader.py",
    "src/models/collaborative_filtering.py",
    "src/models/ranking.py",
    "src/models/preference_vector.py",
    "src/evaluation/metrics.py",
    "src/evaluation/statistical_tests.py",
    "src/experiments/run_revised_experiment.py",
]
METHOD_ORDER = ["ucf", "hybrid_cf", "f1", "f2"]
METRIC_ORDER = ["ndcg@5", "ndcg@10", "ndcg@20", "rmse", "mae"]


def to_builtin(value: Any) -> Any:
    """Convert NumPy values to strict JSON-compatible Python values."""
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize scientific objects as sorted compact UTF-8 JSON."""
    return json.dumps(
        to_builtin(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def hash_dataset(ratings: np.ndarray) -> str:
    """Hash shape metadata plus C-contiguous little-endian float64 bytes."""
    canonical = np.ascontiguousarray(ratings, dtype="<f8")
    metadata = {"dtype": "<f8", "shape": list(canonical.shape), "order": "C"}
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(metadata))
    digest.update(b"\n")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def hash_split(train_ids: Sequence[int], test_ids: Sequence[int]) -> str:
    payload = {
        "test_ids": sorted(int(item) for item in test_ids),
        "train_ids": sorted(int(item) for item in train_ids),
    }
    return sha256_json(payload)


def seed_sequence_record(seed_sequence: np.random.SeedSequence) -> dict:
    entropy = seed_sequence.entropy
    if isinstance(entropy, tuple):
        entropy = list(entropy)
    return {
        "entropy": to_builtin(entropy),
        "spawn_key": [int(value) for value in seed_sequence.spawn_key],
        "pool_size": int(seed_sequence.pool_size),
    }


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_frozen_config(config)
    return config


def validate_frozen_config(config: dict) -> None:
    """Fail closed if any frozen scientific parameter is changed."""
    expected = {
        "schema_version": "1.0",
        "master_seed": 2604001,
        "n_replicates": 50,
        "feedback_sigmas": [0.0, 0.5, 1.0, 2.0],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Frozen configuration mismatch for {key}")
    rng_config = config["rng"]
    if rng_config != {
        "seed_sequence_children": 51,
        "replicate_child_streams": ["dataset", "split", "feedback"],
        "bit_generator": "PCG64",
        "dataset_draw_order": [
            "item_attractiveness[50]",
            "category_preferences[4,6]",
            "idiosyncratic_noise[4,50]",
        ],
        "bootstrap_child_index": 50,
    }:
        raise ValueError("Frozen RNG convention changed")
    data = config["data_generation"]
    if (
        data["n_users"] != 4
        or data["n_items"] != 50
        or data["rating_min"] != 0.0
        or data["rating_max"] != 10.0
        or data["round_ratings"] is not False
    ):
        raise ValueError("Frozen data dimensions or rating scale changed")
    if [entry["count"] for entry in data["categories"]] != [5, 4, 11, 10, 15, 5]:
        raise ValueError("Frozen category counts changed")
    if sum(entry["count"] for entry in data["categories"]) != 50:
        raise ValueError("Category counts must total 50")
    if data["user_biases"] != {"A": 0.5, "B": 0.3, "C": 0.0, "D": -0.5}:
        raise ValueError("Frozen user biases changed")
    checks = [
        (
            data["item_attractiveness"],
            {"distribution": "uniform", "low": 2.0, "high": 8.0},
        ),
        (
            data["category_preference"],
            {"distribution": "normal", "mean": 0.0, "std": 1.0},
        ),
        (
            data["idiosyncratic_noise"],
            {"distribution": "normal", "mean": 0.0, "std": 0.5},
        ),
    ]
    if any(actual != frozen for actual, frozen in checks):
        raise ValueError("Frozen DGP distribution changed")
    if config["split"]["train_items"] != 30 or config["split"]["test_items"] != 20:
        raise ValueError("Frozen split changed")
    methods = config["methods"]
    if (
        methods["ucf"]["k"] != 3
        or methods["item_cf"]["k"] != 3
        or methods["f1"]["displayed_top_n"] != 10
        or methods["f2"]["propagation_k"] != 5
    ):
        raise ValueError("Frozen method parameter changed")
    if (
        methods["ucf"]["similarity"] != "cosine_common_observed"
        or methods["item_cf"]["similarity"] != "cosine_reference_users"
        or methods["item_cf"]["reference_users"] != ["B", "C", "D"]
        or methods["hybrid_cf"]["user_cf_weight"] != 0.5
        or methods["hybrid_cf"]["item_cf_weight"] != 0.5
        or methods["f1"]["output_type"] != "ranking_only"
        or methods["f2"]["positive_similarities_only"] is not True
    ):
        raise ValueError("Frozen method definition changed")
    metric_config = config["metrics"]
    if (
        metric_config["primary"] != "ndcg@10"
        or metric_config["ranking_cutoffs"] != [5, 10, 20]
        or metric_config["gain"] != "linear_relevance"
        or metric_config["score_methods"] != ["ucf", "hybrid_cf", "f2"]
        or metric_config["binary_metrics_enabled"] is not False
    ):
        raise ValueError("Frozen metric definition changed")
    stats_config = config["statistics"]
    if (
        stats_config["holm_family_alpha"] != 0.05
        or stats_config["bootstrap_samples"] != 10000
        or [item["id"] for item in stats_config["primary_contrasts"]]
        != ["C1", "C2", "C3"]
    ):
        raise ValueError("Frozen inferential parameter changed")
    if stats_config["wilcoxon"] != {
        "alternative": "two-sided",
        "zero_method": "pratt",
        "method": "approx",
        "correction": False,
    }:
        raise ValueError("Frozen Wilcoxon configuration changed")
    if (
        stats_config["bootstrap_confidence_level"] != 0.95
        or stats_config["bootstrap_method"] != "percentile"
        or stats_config["observational_unit"] != "independent_replicate"
    ):
        raise ValueError("Frozen bootstrap or observational unit changed")


def source_fingerprint(repo_root: Path) -> str:
    files = {}
    for relative_path in SCIENTIFIC_SOURCE_FILES:
        contents = (repo_root / relative_path).read_bytes()
        files[relative_path] = hashlib.sha256(contents).hexdigest()
    return sha256_json(files)


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "max": float(quantiles[4]),
    }


def _cosine_matrix(ratings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(ratings, axis=1)
    denominator = norms[:, np.newaxis] * norms[np.newaxis, :]
    return np.divide(
        ratings @ ratings.T,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )


def _predict_items(model: Any, item_ids: Sequence[int]) -> np.ndarray:
    return np.asarray([model.predict(0, int(item_id)) for item_id in item_ids])


def _clip_predictions(values: np.ndarray) -> Tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=np.float64)
    count = int(np.count_nonzero((values < 0.0) | (values > 10.0)))
    return np.clip(values, 0.0, 10.0), count


def _fit_revised_models(
    ratings: np.ndarray,
    observed_mask: np.ndarray,
    item_similarity: np.ndarray | None,
    ucf_k: int,
    item_k: int,
) -> Tuple[UserBasedCF, ItemBasedCF]:
    ucf = UserBasedCF(similarity="cosine", k_neighbors=ucf_k).fit(
        ratings, observed_mask=observed_mask
    )
    item_cf = ItemBasedCF(similarity="cosine", k_neighbors=item_k).fit(
        ratings,
        observed_mask=observed_mask,
        reference_users=[1, 2, 3] if item_similarity is None else None,
        item_similarity=item_similarity,
    )
    return ucf, item_cf


def _score_method_record(
    item_ids: np.ndarray,
    predictions: np.ndarray,
    truth_by_item: np.ndarray,
    clipping_count: int,
) -> dict:
    order = deterministic_score_order(item_ids, predictions)
    return {
        "predictions": predictions,
        "order": order,
        "metrics": evaluate_native_score_method(
            item_ids, predictions, truth_by_item, [5, 10, 20]
        ),
        "prediction_clipping_count": clipping_count,
    }


def run_replicate(
    replicate_id: int,
    replicate_root: np.random.SeedSequence,
    config: dict,
) -> dict:
    """Run one independent dataset/split replicate under the frozen design."""
    dataset_ss, split_ss, feedback_ss = replicate_root.spawn(3)
    dataset_rng = np.random.Generator(np.random.PCG64(dataset_ss))
    split_rng = np.random.Generator(np.random.PCG64(split_ss))
    feedback_rng = np.random.Generator(np.random.PCG64(feedback_ss))

    data_config = config["data_generation"]
    category_counts = {
        entry["name"]: int(entry["count"]) for entry in data_config["categories"]
    }
    generated = generate_revised_synthetic_data(
        dataset_rng,
        category_counts=category_counts,
        user_biases=data_config["user_biases"],
        attractiveness_low=data_config["item_attractiveness"]["low"],
        attractiveness_high=data_config["item_attractiveness"]["high"],
        category_preference_std=data_config["category_preference"]["std"],
        idiosyncratic_noise_std=data_config["idiosyncratic_noise"]["std"],
        rating_min=data_config["rating_min"],
        rating_max=data_config["rating_max"],
    )
    ratings = generated["ratings"]
    observed_mask, train_ids_list, test_ids_list = split_revised_user_observations(
        ratings,
        split_rng,
        user_idx=0,
        n_train=config["split"]["train_items"],
    )
    train_ids = np.asarray(train_ids_list, dtype=np.int64)
    test_ids = np.asarray(test_ids_list, dtype=np.int64)
    feedback_standard_normal = feedback_rng.normal(0.0, 1.0, 50)

    ucf_k = config["methods"]["ucf"]["k"]
    item_k = config["methods"]["item_cf"]["k"]
    ucf, item_cf = _fit_revised_models(ratings, observed_mask, None, ucf_k, item_k)
    item_similarity = item_cf.item_similarity_.copy()

    ucf_predictions, ucf_clipping = _clip_predictions(_predict_items(ucf, test_ids))
    item_predictions, item_clipping = _clip_predictions(
        _predict_items(item_cf, test_ids)
    )
    hybrid_raw = 0.5 * ucf_predictions + 0.5 * item_predictions
    hybrid_predictions, hybrid_clipping = _clip_predictions(hybrid_raw)
    hybrid_order = deterministic_score_order(test_ids, hybrid_predictions)

    base_methods = {
        "ucf": _score_method_record(
            test_ids, ucf_predictions, ratings[0], ucf_clipping
        ),
        "hybrid_cf": _score_method_record(
            test_ids, hybrid_predictions, ratings[0], hybrid_clipping
        ),
    }

    loo_predictions = np.zeros(len(train_ids), dtype=np.float64)
    loo_component_clipping = 0
    for position, item_id in enumerate(train_ids):
        loo_mask = observed_mask.copy()
        loo_mask[0, int(item_id)] = False
        loo_ucf, loo_item_cf = _fit_revised_models(
            ratings, loo_mask, item_similarity, ucf_k, item_k
        )
        loo_ucf_prediction, clip_u = _clip_predictions(
            np.asarray([loo_ucf.predict(0, int(item_id))])
        )
        loo_item_prediction, clip_i = _clip_predictions(
            np.asarray([loo_item_cf.predict(0, int(item_id))])
        )
        loo_component_clipping += clip_u + clip_i
        loo_predictions[position] = 0.5 * (
            loo_ucf_prediction[0] + loo_item_prediction[0]
        )

    noise_conditions = {}
    for sigma in config["feedback_sigmas"]:
        sigma_key = str(float(sigma))

        # Intentional and exclusive pre-evaluation use of User A test truth:
        # simulate feedback only on the ten test candidates displayed by F1.
        displayed_ids = hybrid_order[: config["methods"]["f1"]["displayed_top_n"]]
        f1_feedback = np.clip(
            ratings[0, displayed_ids]
            + float(sigma) * feedback_standard_normal[displayed_ids],
            0.0,
            10.0,
        )
        feedback_map = {
            int(item_id): float(value)
            for item_id, value in zip(displayed_ids, f1_feedback)
        }
        f1_reranking = interactive_impression_rerank(
            hybrid_order,
            config["methods"]["f1"]["displayed_top_n"],
            feedback_map,
        )
        f1_record = {
            "output_type": "post-feedback interactive reranking",
            "initial_order": f1_reranking["initial_order"],
            "displayed_ids": f1_reranking["displayed_ids"],
            "feedback_values": f1_reranking["displayed_feedback"],
            "final_order": f1_reranking["final_order"],
            "changed_positions": int(f1_reranking["changed_positions"]),
            "metrics": evaluate_ranking_order(
                f1_reranking["final_order"], ratings[0], test_ids, [5, 10, 20]
            ),
        }

        f2_feedback = np.clip(
            ratings[0, train_ids] + float(sigma) * feedback_standard_normal[train_ids],
            0.0,
            10.0,
        )
        residuals = f2_feedback - loo_predictions
        propagation = propagate_preference_residuals(
            residuals,
            train_ids,
            test_ids,
            item_similarity,
            k=config["methods"]["f2"]["propagation_k"],
        )
        f2_raw = hybrid_predictions + propagation["propagated_residuals"]
        f2_predictions, f2_clipping = _clip_predictions(f2_raw)
        f2_record = _score_method_record(
            test_ids, f2_predictions, ratings[0], f2_clipping
        )
        f2_record.update(
            {
                "loo_base_predictions": loo_predictions,
                "feedback_values": f2_feedback,
                "residuals": residuals,
                "selected_neighbors": propagation["neighbors"],
                "propagated_residuals": propagation["propagated_residuals"],
                "fallback_count": propagation["fallback_count"],
                "effective_neighbor_counts": propagation["effective_neighbor_counts"],
                "loo_component_clipping_count": loo_component_clipping,
            }
        )
        noise_conditions[sigma_key] = {
            "sigma": float(sigma),
            "f1": f1_record,
            "f2": f2_record,
        }

    user_similarity = ucf.user_similarity_.copy()
    neighbor_ids = np.asarray([1, 2, 3], dtype=np.int64)
    neighbor_order = neighbor_ids[
        np.lexsort((neighbor_ids, -user_similarity[0, neighbor_ids]))
    ]
    common_counts = observed_mask.astype(np.int64) @ observed_mask.astype(np.int64).T

    diagnostics = {
        "rating_distribution": _distribution(ratings.ravel()),
        "user_means": {
            USER_IDS[index]: float(ratings[index].mean()) for index in range(4)
        },
        "user_user_pearson": np.corrcoef(ratings),
        "user_user_cosine": _cosine_matrix(ratings),
        "common_observation_counts": common_counts,
        "user_a_ucf_neighbor_order": [USER_IDS[index] for index in neighbor_order],
        "user_a_ucf_neighbor_similarities": [
            float(user_similarity[0, index]) for index in neighbor_order
        ],
        "test_rating_distribution": _distribution(ratings[0, test_ids]),
        "item_similarity_reference_users": ["B", "C", "D"],
        "item_similarity_vector_dimension": 3,
        "internal_item_cf_predictions": item_predictions,
        "internal_item_cf_clipping_count": item_clipping,
    }

    return to_builtin(
        {
            "replicate_id": replicate_id,
            "rng": {
                "replicate_root": seed_sequence_record(replicate_root),
                "dataset": seed_sequence_record(dataset_ss),
                "split": seed_sequence_record(split_ss),
                "feedback": seed_sequence_record(feedback_ss),
                "bit_generator": "PCG64",
                "numpy_version": np.__version__,
            },
            "dataset_hash": hash_dataset(ratings),
            "split_hash": hash_split(train_ids, test_ids),
            "train_ids": train_ids,
            "test_ids": test_ids,
            "user_a_truth": ratings[0],
            "base_methods": base_methods,
            "noise_conditions": noise_conditions,
            "diagnostics": diagnostics,
        }
    )


def _metric_values(
    replicates: List[dict], sigma: float, method: str, metric: str
) -> np.ndarray:
    sigma_key = str(float(sigma))
    values = []
    for replicate in replicates:
        if method in ("ucf", "hybrid_cf"):
            record = replicate["base_methods"][method]
        else:
            record = replicate["noise_conditions"][sigma_key][method]
        values.append(record["metrics"][metric])
    return np.asarray(values, dtype=np.float64)


def _contrast_values(replicates: List[dict], sigma: float, contrast: str) -> np.ndarray:
    if contrast == "C1":
        return _metric_values(replicates, sigma, "f1", "ndcg@10") - _metric_values(
            replicates, sigma, "hybrid_cf", "ndcg@10"
        )
    if contrast == "C2":
        return _metric_values(replicates, sigma, "f2", "ndcg@10") - _metric_values(
            replicates, sigma, "hybrid_cf", "ndcg@10"
        )
    if contrast == "C3":
        return _metric_values(replicates, sigma, "hybrid_cf", "rmse") - _metric_values(
            replicates, sigma, "f2", "rmse"
        )
    raise ValueError(f"Unknown contrast: {contrast}")


def _aggregate_diagnostics(replicates: List[dict], sigmas: Sequence[float]) -> dict:
    pearson = np.asarray([r["diagnostics"]["user_user_pearson"] for r in replicates])
    cosine = np.asarray([r["diagnostics"]["user_user_cosine"] for r in replicates])
    user_means = {
        user: _distribution(
            np.asarray([r["diagnostics"]["user_means"][user] for r in replicates])
        )
        for user in USER_IDS
    }
    result = {
        "user_user_pearson_mean_matrix": pearson.mean(axis=0),
        "user_user_cosine_mean_matrix": cosine.mean(axis=0),
        "user_means": user_means,
        "f1_changed_positions": {},
        "f2_residual_distribution": {},
        "f2_propagated_residual_distribution": {},
        "f2_effective_neighbor_counts": {},
        "f2_fallback_count": {},
        "prediction_clipping_counts": {},
    }
    for sigma in sigmas:
        key = str(float(sigma))
        f1_changed = np.asarray(
            [r["noise_conditions"][key]["f1"]["changed_positions"] for r in replicates]
        )
        residuals = np.concatenate(
            [
                np.asarray(r["noise_conditions"][key]["f2"]["residuals"])
                for r in replicates
            ]
        )
        propagated = np.concatenate(
            [
                np.asarray(r["noise_conditions"][key]["f2"]["propagated_residuals"])
                for r in replicates
            ]
        )
        effective = np.concatenate(
            [
                np.asarray(
                    r["noise_conditions"][key]["f2"]["effective_neighbor_counts"]
                )
                for r in replicates
            ]
        )
        result["f1_changed_positions"][key] = _distribution(f1_changed)
        result["f2_residual_distribution"][key] = _distribution(residuals)
        result["f2_propagated_residual_distribution"][key] = _distribution(propagated)
        result["f2_effective_neighbor_counts"][key] = _distribution(effective)
        result["f2_fallback_count"][key] = int(
            sum(r["noise_conditions"][key]["f2"]["fallback_count"] for r in replicates)
        )
        result["prediction_clipping_counts"][key] = {
            "ucf": int(
                sum(
                    r["base_methods"]["ucf"]["prediction_clipping_count"]
                    for r in replicates
                )
            ),
            "hybrid_cf": int(
                sum(
                    r["base_methods"]["hybrid_cf"]["prediction_clipping_count"]
                    for r in replicates
                )
            ),
            "f2": int(
                sum(
                    r["noise_conditions"][key]["f2"]["prediction_clipping_count"]
                    for r in replicates
                )
            ),
        }
    return to_builtin(result)


def _build_scientific_payload(
    config: dict, repo_root: Path, replicate_count: int
) -> dict:
    """Build a payload; canonical calls always use all 50 replicates."""
    if not 1 <= replicate_count <= 50:
        raise ValueError("replicate_count must be between 1 and 50")
    master = np.random.SeedSequence(config["master_seed"])
    children = master.spawn(51)  # Frozen: call exactly once.
    replicate_roots = children[:replicate_count]
    inference_ss = children[50]
    replicates = [
        run_replicate(replicate_id, replicate_root, config)
        for replicate_id, replicate_root in enumerate(replicate_roots)
    ]

    inference_rng = np.random.Generator(np.random.PCG64(inference_ss))
    n_bootstrap = config["statistics"]["bootstrap_samples"]
    confidence = config["statistics"]["bootstrap_confidence_level"]
    sigmas = config["feedback_sigmas"]

    aggregate_summaries: Dict[str, dict] = {}
    for sigma in sigmas:
        sigma_key = str(float(sigma))
        aggregate_summaries[sigma_key] = {}
        for method in METHOD_ORDER:
            metrics = ["ndcg@5", "ndcg@10", "ndcg@20"]
            if method != "f1":
                metrics.extend(["rmse", "mae"])
            aggregate_summaries[sigma_key][method] = {}
            for metric in metrics:
                values = _metric_values(replicates, sigma, method, metric)
                aggregate_summaries[sigma_key][method][
                    metric
                ] = percentile_bootstrap_summary(
                    values, inference_rng, n_bootstrap, confidence
                )

    paired_effects: Dict[str, dict] = {}
    for sigma in sigmas:
        sigma_key = str(float(sigma))
        paired_effects[sigma_key] = {}
        for contrast in ["C1", "C2", "C3"]:
            values = _contrast_values(replicates, sigma, contrast)
            summary = percentile_bootstrap_summary(
                values, inference_rng, n_bootstrap, confidence
            )
            summary["differences"] = to_builtin(values)
            paired_effects[sigma_key][contrast] = summary

    primary_tests = {}
    raw_p_values = {}
    for contrast in ["C1", "C2", "C3"]:
        differences = np.asarray(paired_effects["0.0"][contrast]["differences"])
        test = revised_wilcoxon_signed_rank(differences)
        primary_tests[contrast] = test
        raw_p_values[contrast] = test["p_value"]
    holm = holm_adjust(raw_p_values, config["statistics"]["holm_family_alpha"])
    for contrast in primary_tests:
        primary_tests[contrast]["holm_adjusted_p"] = holm[contrast]["adjusted_p"]
        primary_tests[contrast]["holm_reject"] = holm[contrast]["reject"]

    payload = {
        "schema_version": "1.0",
        "config_hash": sha256_json(config),
        "config": config,
        "source_fingerprint": source_fingerprint(repo_root),
        "rng": {
            "master_seed": config["master_seed"],
            "master": seed_sequence_record(master),
            "children_spawned_once": 51,
            "replicate_child_indices": list(range(replicate_count)),
            "inference": seed_sequence_record(inference_ss),
            "bit_generator": "PCG64",
            "bootstrap_sequence_order": "aggregate summaries by sigma/method/metric, then paired effects by sigma/C1-C3",
        },
        "replicates": replicates,
        "aggregate_summaries": aggregate_summaries,
        "paired_effects": paired_effects,
        "primary_tests": primary_tests,
        "holm_family": ["C1", "C2", "C3"],
        "aggregate_diagnostics": _aggregate_diagnostics(replicates, sigmas),
    }
    return to_builtin(payload)


def build_scientific_payload(config: dict, repo_root: Path) -> dict:
    """Build the canonical deterministic payload using all 50 replicates."""
    return _build_scientific_payload(config, repo_root, config["n_replicates"])


def build_smoke_scientific_payload(
    config: dict, repo_root: Path, replicate_count: int = 2
) -> dict:
    """Exercise the complete pipeline without creating a canonical artifact."""
    if replicate_count >= config["n_replicates"]:
        raise ValueError("smoke runs must use fewer than 50 replicates")
    return _build_scientific_payload(config, repo_root, replicate_count)


def _git_output(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def build_provenance(repo_root: Path) -> dict:
    packages = {}
    for package in ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "pytest"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    status = _git_output(repo_root, "status", "--porcelain=v1")
    return {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_output(repo_root, "rev-parse", "HEAD"),
        "git_branch": _git_output(repo_root, "branch", "--show-current"),
        "worktree_dirty": bool(status),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependency_versions": packages,
    }


def build_results(config: dict, repo_root: Path) -> dict:
    scientific_payload = build_scientific_payload(config, repo_root)
    return assemble_results(scientific_payload, repo_root)


def assemble_results(scientific_payload: dict, repo_root: Path) -> dict:
    """Separate deterministic science from execution-specific provenance."""
    return {
        "scientific_payload": scientific_payload,
        "scientific_payload_sha256": sha256_json(scientific_payload),
        "provenance": build_provenance(repo_root),
    }


def write_results(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            to_builtin(results), handle, indent=2, sort_keys=True, allow_nan=False
        )
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/revised_experiment.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/revised_experiment/canonical_results.json"),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(args.config)
    results = build_results(config, repo_root)
    write_results(results, args.output)
    print(f"Wrote {args.output}")
    print(f"config_sha256={results['scientific_payload']['config_hash']}")
    print(f"scientific_payload_sha256={results['scientific_payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
