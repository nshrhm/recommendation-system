"""Independently verify the revised experiment's canonical result artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from src.experiments.run_revised_experiment import (
    build_scientific_payload,
    load_config,
    sha256_json,
)


def verify_results(
    config_path: Path, result_path: Path, reproduce: bool = False
) -> List[str]:
    config = load_config(config_path)
    with result_path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    errors: List[str] = []

    if set(document) != {
        "scientific_payload",
        "scientific_payload_sha256",
        "provenance",
    }:
        errors.append(
            "top-level schema must separate scientific_payload and provenance"
        )
        return errors
    payload = document["scientific_payload"]
    if payload.get("config_hash") != sha256_json(config):
        errors.append("config hash mismatch")
    if document["scientific_payload_sha256"] != sha256_json(payload):
        errors.append("scientific payload hash mismatch")

    replicates = payload.get("replicates", [])
    if len(replicates) != 50:
        errors.append("replicate count is not 50")
    if [record.get("replicate_id") for record in replicates] != list(range(50)):
        errors.append("replicate IDs are not exactly 0..49")

    expected_sigmas = {"0.0", "0.5", "1.0", "2.0"}
    required_diagnostics = {
        "rating_distribution",
        "user_means",
        "user_user_pearson",
        "user_user_cosine",
        "common_observation_counts",
        "user_a_ucf_neighbor_order",
        "test_rating_distribution",
    }
    for replicate_id, record in enumerate(replicates):
        rng = record.get("rng", {})
        expected_spawn_keys = {
            "replicate_root": [replicate_id],
            "dataset": [replicate_id, 0],
            "split": [replicate_id, 1],
            "feedback": [replicate_id, 2],
        }
        for stream, spawn_key in expected_spawn_keys.items():
            if rng.get(stream, {}).get("spawn_key") != spawn_key:
                errors.append(f"replicate {replicate_id}: invalid {stream} spawn key")
        for hash_name in ["dataset_hash", "split_hash"]:
            value = record.get(hash_name, "")
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                errors.append(f"replicate {replicate_id}: invalid {hash_name}")
        train_ids = record.get("train_ids", [])
        test_ids = record.get("test_ids", [])
        if (
            len(train_ids) != 30
            or len(test_ids) != 20
            or set(train_ids) & set(test_ids)
        ):
            errors.append(f"replicate {replicate_id}: invalid 30/20 split")
        if sorted(train_ids + test_ids) != list(range(50)):
            errors.append(f"replicate {replicate_id}: split does not cover 50 items")
        if set(record.get("base_methods", {})) != {"ucf", "hybrid_cf"}:
            errors.append(f"replicate {replicate_id}: invalid base methods")
        if set(record.get("noise_conditions", {})) != expected_sigmas:
            errors.append(f"replicate {replicate_id}: invalid noise conditions")
        if not required_diagnostics.issubset(record.get("diagnostics", {})):
            errors.append(f"replicate {replicate_id}: missing diagnostics")

        for sigma, condition in record.get("noise_conditions", {}).items():
            if set(condition) != {"sigma", "f1", "f2"}:
                errors.append(
                    f"replicate {replicate_id} sigma {sigma}: invalid methods"
                )
                continue
            f1_metrics = condition["f1"].get("metrics", {})
            if "rmse" in f1_metrics or "mae" in f1_metrics:
                errors.append(
                    f"replicate {replicate_id} sigma {sigma}: F1 has score errors"
                )
            if len(condition["f1"].get("displayed_ids", [])) != 10:
                errors.append(
                    f"replicate {replicate_id} sigma {sigma}: F1 display count"
                )
            if len(condition["f2"].get("loo_base_predictions", [])) != 30:
                errors.append(f"replicate {replicate_id} sigma {sigma}: F2 LOO count")
            if len(condition["f2"].get("predictions", [])) != 20:
                errors.append(
                    f"replicate {replicate_id} sigma {sigma}: F2 prediction count"
                )
            for prediction in condition["f2"].get("predictions", []):
                if not 0.0 <= prediction <= 10.0:
                    errors.append(
                        f"replicate {replicate_id} sigma {sigma}: F2 out of range"
                    )

        for method_record in record.get("base_methods", {}).values():
            for prediction in method_record.get("predictions", []):
                if not 0.0 <= prediction <= 10.0:
                    errors.append(
                        f"replicate {replicate_id}: base prediction out of range"
                    )
        for method in ["ucf", "hybrid_cf"]:
            metrics = record.get("base_methods", {}).get(method, {}).get("metrics", {})
            for name, value in metrics.items():
                upper = 1.0 if name.startswith("ndcg") else 10.0
                if not 0.0 <= value <= upper:
                    errors.append(f"replicate {replicate_id}: invalid {method} {name}")

    if payload.get("holm_family") != ["C1", "C2", "C3"]:
        errors.append("Holm family is not exactly C1/C2/C3")
    tests = payload.get("primary_tests", {})
    if set(tests) != {"C1", "C2", "C3"}:
        errors.append("primary test family is not exactly three contrasts")
    for contrast, test in tests.items():
        if test.get("n") != 50:
            errors.append(f"{contrast}: Wilcoxon sample size is not 50")
    paired = payload.get("paired_effects", {})
    if set(paired) != expected_sigmas:
        errors.append("paired effects omit a noise condition")
    for sigma, effects in paired.items():
        if set(effects) != {"C1", "C2", "C3"}:
            errors.append(f"sigma {sigma}: contrast definitions incomplete")
        for contrast, summary in effects.items():
            if summary.get("bootstrap_samples") != 10000 or summary.get("n") != 50:
                errors.append(
                    f"sigma {sigma} {contrast}: bootstrap configuration invalid"
                )

    aggregate_diagnostics = payload.get("aggregate_diagnostics", {})
    for key in [
        "f1_changed_positions",
        "f2_residual_distribution",
        "f2_propagated_residual_distribution",
        "f2_effective_neighbor_counts",
        "f2_fallback_count",
        "prediction_clipping_counts",
    ]:
        if key not in aggregate_diagnostics:
            errors.append(f"missing aggregate diagnostic: {key}")

    provenance = document.get("provenance", {})
    for key in [
        "utc_timestamp",
        "git_head",
        "git_branch",
        "worktree_dirty",
        "python_version",
        "dependency_versions",
    ]:
        if key not in provenance:
            errors.append(f"missing provenance field: {key}")

    if reproduce and not errors:
        repo_root = Path(__file__).resolve().parents[2]
        regenerated = build_scientific_payload(config, repo_root)
        if sha256_json(regenerated) != sha256_json(payload):
            errors.append(
                "regenerated scientific payload differs from canonical payload"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/revised_experiment.json")
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("results/revised_experiment/canonical_results.json"),
    )
    parser.add_argument("--reproduce", action="store_true")
    args = parser.parse_args()
    errors = verify_results(args.config, args.result, reproduce=args.reproduce)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("VERIFICATION=PASS")
    print(f"REPRODUCED={'yes' if args.reproduce else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
