"""Independently verify the revised experiment's canonical result artifact."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from src.experiments.run_revised_experiment import (
    build_scientific_payload,
    load_config,
    sha256_json,
)


SCIENTIFIC_FLOAT_ATOL = 1e-14
SCIENTIFIC_FLOAT_RTOL = 1e-11
MAX_MISMATCH_DETAILS = 20


@dataclass(frozen=True)
class ScientificComparison:
    """Result of comparing two scientific payloads across environments."""

    mismatch_count: int
    mismatch_details: List[str]
    bitwise_match: bool
    max_observed_float_abs_diff: float
    max_observed_float_rel_diff: float

    @property
    def equivalent(self) -> bool:
        return self.mismatch_count == 0


def _requires_exact_value(path: str) -> bool:
    """Return whether a scientific value is a frozen discrete invariant."""
    if path in {
        "$.schema_version",
        "$.config_hash",
        "$.source_fingerprint",
    } or path.startswith("$.config"):
        return True
    if path.startswith("$.rng") or re.match(r"^\$\.replicates\[\d+\]\.rng", path):
        return True
    if re.match(
        r"^\$\.replicates\[\d+\]\.(replicate_id|dataset_hash|split_hash|train_ids|test_ids)",
        path,
    ):
        return True
    if re.match(
        r"^\$\.replicates\[\d+\]\.noise_conditions\.[^.]+\.f1\."
        r"(initial_order|displayed_ids|final_order|changed_positions|output_type)",
        path,
    ):
        return True
    if re.match(
        r"^\$\.replicates\[\d+\]\.noise_conditions\.[^.]+\.f2\."
        r"(fallback_count|effective_neighbor_counts)",
        path,
    ):
        return True
    if re.match(
        r"^\$\.replicates\[\d+\]\.noise_conditions\.[^.]+\.f2\."
        r"selected_neighbors\[\d+\]\.(test_item_id|neighbor_ids)",
        path,
    ):
        return True
    return path.startswith("$.holm_family")


def compare_scientific_payloads(
    reference: Any,
    candidate: Any,
    *,
    max_details: int = MAX_MISMATCH_DETAILS,
) -> ScientificComparison:
    """Compare scientific payloads exactly except for derived finite floats."""
    mismatch_count = 0
    details: List[str] = []
    max_abs_diff = 0.0
    max_rel_diff = 0.0

    def mismatch(
        path: str, reference_value: Any, candidate_value: Any, kind: str
    ) -> None:
        nonlocal mismatch_count
        mismatch_count += 1
        if len(details) < max_details:
            details.append(
                f"{path}: {kind}; reference={reference_value!r}; "
                f"candidate={candidate_value!r}"
            )

    def walk(reference_value: Any, candidate_value: Any, path: str) -> None:
        nonlocal max_abs_diff, max_rel_diff
        if type(reference_value) is not type(candidate_value):
            mismatch(
                path,
                reference_value,
                candidate_value,
                "type mismatch "
                f"({type(reference_value).__name__} != "
                f"{type(candidate_value).__name__})",
            )
            return
        if isinstance(reference_value, dict):
            if set(reference_value) != set(candidate_value):
                mismatch(
                    path,
                    sorted(reference_value),
                    sorted(candidate_value),
                    "dictionary key-set mismatch",
                )
                return
            for key in reference_value:
                walk(reference_value[key], candidate_value[key], f"{path}.{key}")
            return
        if isinstance(reference_value, list):
            if len(reference_value) != len(candidate_value):
                mismatch(
                    path,
                    len(reference_value),
                    len(candidate_value),
                    "list-length mismatch",
                )
                return
            for index, (reference_item, candidate_item) in enumerate(
                zip(reference_value, candidate_value)
            ):
                walk(reference_item, candidate_item, f"{path}[{index}]")
            return
        if isinstance(reference_value, float):
            if not math.isfinite(reference_value) or not math.isfinite(candidate_value):
                mismatch(
                    path,
                    reference_value,
                    candidate_value,
                    "non-finite float",
                )
                return
            absolute = abs(candidate_value - reference_value)
            relative = absolute / max(
                abs(reference_value), abs(candidate_value), 1e-300
            )
            max_abs_diff = max(max_abs_diff, absolute)
            max_rel_diff = max(max_rel_diff, relative)
            exact = _requires_exact_value(path)
            close = math.isclose(
                candidate_value,
                reference_value,
                rel_tol=SCIENTIFIC_FLOAT_RTOL,
                abs_tol=SCIENTIFIC_FLOAT_ATOL,
            )
            if (exact and candidate_value != reference_value) or (
                not exact and not close
            ):
                kind = (
                    "exact invariant float mismatch"
                    if exact
                    else "float tolerance exceeded"
                )
                mismatch(
                    path,
                    reference_value,
                    candidate_value,
                    f"{kind}; abs_diff={absolute}; rel_diff={relative}; "
                    f"atol={SCIENTIFIC_FLOAT_ATOL}; rtol={SCIENTIFIC_FLOAT_RTOL}",
                )
            return
        if candidate_value != reference_value:
            mismatch(path, reference_value, candidate_value, "exact value mismatch")

    walk(reference, candidate, "$")
    return ScientificComparison(
        mismatch_count=mismatch_count,
        mismatch_details=details,
        bitwise_match=sha256_json(reference) == sha256_json(candidate),
        max_observed_float_abs_diff=max_abs_diff,
        max_observed_float_rel_diff=max_rel_diff,
    )


def _load_document(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_results(
    config_path: Path,
    result_path: Path,
    reproduce: bool = False,
    reference_path: Path | None = None,
    _comparisons: List[ScientificComparison] | None = None,
) -> List[str]:
    config = load_config(config_path)
    document = _load_document(result_path)
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
        comparison = compare_scientific_payloads(payload, regenerated)
        if _comparisons is not None:
            _comparisons.append(comparison)
        if not comparison.equivalent:
            errors.append(
                f"scientific payload mismatch count: {comparison.mismatch_count}"
            )
            errors.extend(comparison.mismatch_details)
    if reference_path is not None and not errors:
        reference_errors = verify_results(config_path, reference_path)
        if reference_errors:
            errors.extend(f"reference: {error}" for error in reference_errors)
        else:
            reference_document = _load_document(reference_path)
            comparison = compare_scientific_payloads(
                reference_document["scientific_payload"], payload
            )
            if _comparisons is not None:
                _comparisons.append(comparison)
            if not comparison.equivalent:
                errors.append(
                    f"scientific payload mismatch count: {comparison.mismatch_count}"
                )
                errors.extend(comparison.mismatch_details)
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
    comparison_mode = parser.add_mutually_exclusive_group()
    comparison_mode.add_argument("--reproduce", action="store_true")
    comparison_mode.add_argument("--reference", type=Path)
    args = parser.parse_args()
    comparisons: List[ScientificComparison] = []
    errors = verify_results(
        args.config,
        args.result,
        reproduce=args.reproduce,
        reference_path=args.reference,
        _comparisons=comparisons,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("VERIFICATION=PASS")
    print(f"REPRODUCED={'yes' if args.reproduce else 'no'}")
    if args.reproduce or args.reference is not None:
        comparison = comparisons[0]
        print("SCIENTIFIC_EQUIVALENCE=PASS")
        print(f"BITWISE_PAYLOAD_MATCH={'yes' if comparison.bitwise_match else 'no'}")
        print(f"FLOAT_ATOL={SCIENTIFIC_FLOAT_ATOL}")
        print(f"FLOAT_RTOL={SCIENTIFIC_FLOAT_RTOL}")
        print(f"MAX_OBSERVED_FLOAT_ABS_DIFF={comparison.max_observed_float_abs_diff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
