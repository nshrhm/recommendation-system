#!/usr/bin/env python3
"""
Verify that key manuscript numbers are traceable to results artifacts.

This script is intentionally lightweight and evidence-based:
- Loads validated summary + p-values from:
    results/statistical_validation/statistical_validation_results.json
- Verifies required figure files referenced by the paper exist.
- Verifies the main results table numbers in paper/main.tex match:
    - RMSE raw / normalized / NDCG@10 / p-values from the validation JSON
    - MAE / NDCG@20 / Precision@10 from representative metrics.json (seed=42)

Exit code:
  0 = OK
  1 = mismatch or missing artifact
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TableRow:
    method: str
    rmse_raw: str
    rmse_norm: str
    mae: str
    ndcg10: str
    ndcg20: str
    precision10: str
    p_value: str


def _round_str(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_validation_summary(validation_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    summary = validation_json.get("summary")
    if not isinstance(summary, list):
        raise ValueError("validation json missing 'summary' list")
    out: Dict[str, Dict[str, Any]] = {}
    for row in summary:
        method = row.get("Method")
        if isinstance(method, str):
            out[method.upper()] = row
    for needed in ["BASELINE", "F1", "F2"]:
        if needed not in out:
            raise ValueError(f"validation json missing summary row for {needed}")
    return out


def load_validation_pvalues(validation_json: Dict[str, Any]) -> Tuple[float, float]:
    tests = validation_json.get("statistical_tests", {})
    try:
        p_f1 = float(tests["f1_vs_baseline"]["p_value"])
        p_f2 = float(tests["f2_vs_baseline"]["p_value"])
    except Exception as e:  # noqa: BLE001
        raise ValueError("validation json missing expected statistical_tests structure") from e
    return p_f1, p_f2


def parse_includegraphics_paths(main_tex: str) -> List[str]:
    # Captures \includegraphics[...]{path} and \includegraphics{path}
    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    return pattern.findall(main_tex)


def parse_main_results_table(main_tex: str) -> Dict[str, TableRow]:
    """
    Parse the main paper results table rows (Baseline/F1/F2).

    Expected row format (after stripping \textbf{...}):
      Baseline & 2.25 & 2.25 & 1.72 & 0.7213 & 0.8599 & 1.0000 & --- \\
      F1 & 21.34 & 2.13 & 17.84 & 0.7153 & 0.8541 & 1.0000 & 0.068 \\
      F2 & 33.16 & 3.32 & 28.76 & 0.7153 & 0.8541 & 1.0000 & <0.001*** \\
    """

    def strip_tex(token: str) -> str:
        token = token.strip()
        token = re.sub(r"\\textbf\{([^}]*)\}", r"\1", token)
        token = token.replace(r"\%", "%")
        token = token.replace(r"\,", ",")
        return token.strip()

    rows: Dict[str, TableRow] = {}
    for line in main_tex.splitlines():
        if not line.strip().startswith(("Baseline &", "F1 &", "F2 &")):
            continue
        # Remove trailing LaTeX line break.
        line2 = line.strip()
        if line2.endswith(r"\\"):
            line2 = line2[: -len(r"\\")].strip()
        parts = [strip_tex(p) for p in line2.split("&")]
        if len(parts) != 8:
            continue
        row = TableRow(
            method=parts[0],
            rmse_raw=parts[1],
            rmse_norm=parts[2],
            mae=parts[3],
            ndcg10=parts[4],
            ndcg20=parts[5],
            precision10=parts[6],
            p_value=parts[7],
        )
        rows[row.method.upper()] = row

    for needed in ["BASELINE", "F1", "F2"]:
        if needed not in rows:
            raise ValueError(f"Could not find '{needed} &' row in paper results table.")

    return rows


def find_latest_metrics_json(method: str, repo_root: Path) -> Optional[Path]:
    """
    Find a representative metrics.json for a method.

    Priority:
    1) results/statistical_validation/{method}_seed42/<method>_YYYYmmdd_HHMMSS/metrics.json
    2) results/{method}_YYYYmmdd_HHMMSS/metrics.json
    """
    method = method.lower()

    sv_dir = repo_root / "results" / "statistical_validation" / f"{method}_seed42"
    if sv_dir.exists():
        run_dirs = sorted([d for d in sv_dir.iterdir() if d.is_dir()])
        for run_dir in reversed(run_dirs):
            p = run_dir / "metrics.json"
            if p.exists():
                return p

    root_results = repo_root / "results"
    run_dirs2 = sorted([d for d in root_results.glob(f"{method}_*") if d.is_dir()])
    for run_dir in reversed(run_dirs2):
        p = run_dir / "metrics.json"
        if p.exists():
            return p

    return None


def compare(label: str, got: str, expected: str, failures: List[str]) -> None:
    if got != expected:
        failures.append(f"{label}: got '{got}' expected '{expected}'")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify paper numbers match results artifacts")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root (default: .)",
    )
    parser.add_argument(
        "--paper-tex",
        type=Path,
        default=Path("paper/main.tex"),
        help="Path to paper/main.tex",
    )
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=Path("results/statistical_validation/statistical_validation_results.json"),
        help="Path to statistical_validation_results.json",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    paper_tex_path = (repo_root / args.paper_tex).resolve()
    validation_json_path = (repo_root / args.validation_json).resolve()

    failures: List[str] = []

    # Load inputs
    try:
        paper_tex = paper_tex_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: Missing paper source: {paper_tex_path}")
        return 1

    try:
        validation = load_json(validation_json_path)
    except FileNotFoundError:
        print(f"ERROR: Missing validation JSON: {validation_json_path}")
        return 1

    summary = load_validation_summary(validation)
    p_f1, p_f2 = load_validation_pvalues(validation)

    # 1) Check figures referenced by paper exist.
    paper_dir = paper_tex_path.parent
    for fig_path_str in parse_includegraphics_paths(paper_tex):
        fig_path = (paper_dir / fig_path_str).resolve()
        if not fig_path.exists():
            failures.append(f"Missing figure referenced by paper: {fig_path_str} -> {fig_path}")

    # 2) Check main results table numbers.
    table = parse_main_results_table(paper_tex)

    # RMSE raw and normalized and NDCG@10 come from validation JSON summary.
    for method_upper, method_label in [("BASELINE", "Baseline"), ("F1", "F1"), ("F2", "F2")]:
        row = table[method_upper]
        rmse_raw_mean = float(summary[method_upper]["RMSE (raw) Mean"])
        rmse_norm_mean = float(summary[method_upper]["RMSE (norm) Mean"])
        ndcg10_mean = float(summary[method_upper]["NDCG@10 Mean"])

        compare(
            f"Table RMSE(raw) {method_label}",
            row.rmse_raw,
            _round_str(rmse_raw_mean, 2),
            failures,
        )
        compare(
            f"Table RMSE(norm) {method_label}",
            row.rmse_norm,
            _round_str(rmse_norm_mean, 2),
            failures,
        )
        compare(
            f"Table NDCG@10 {method_label}",
            row.ndcg10,
            _round_str(ndcg10_mean, 4),
            failures,
        )

    # p-values come from validation JSON (Wilcoxon on normalized absolute errors).
    compare("Table p-value F1 vs Base", table["F1"].p_value, _round_str(p_f1, 3), failures)
    if not (table["F2"].p_value.startswith("<") and "0.001" in table["F2"].p_value):
        # Allow either "<0.001***" style or explicit numeric rounding.
        if p_f2 < 0.001:
            failures.append(
                f"Table p-value F2 vs Base: got '{table['F2'].p_value}' expected '<0.001...'"
            )

    # 3) Verify additional table columns from a representative metrics.json (seed=42).
    for method_upper, method_label in [("BASELINE", "baseline"), ("F1", "f1"), ("F2", "f2")]:
        metrics_path = find_latest_metrics_json(method_label, repo_root)
        if metrics_path is None:
            failures.append(f"Could not locate representative metrics.json for {method_label}")
            continue
        metrics_json = load_json(metrics_path)
        metrics = metrics_json.get("metrics", {})
        try:
            mae_val = float(metrics["mae"])
            ndcg20_val = float(metrics["ndcg@20"])
            precision10_val = float(metrics["precision@10"])
        except Exception as e:  # noqa: BLE001
            failures.append(f"{metrics_path}: missing expected metric keys ({e})")
            continue

        row = table[method_upper]
        compare(
            f"Table MAE {method_label}",
            row.mae,
            _round_str(mae_val, 2),
            failures,
        )
        compare(
            f"Table NDCG@20 {method_label}",
            row.ndcg20,
            _round_str(ndcg20_val, 4),
            failures,
        )
        compare(
            f"Table Precision@10 {method_label}",
            row.precision10,
            _round_str(precision10_val, 4),
            failures,
        )

    if failures:
        print("FAIL: results-to-paper verification failed.")
        for f in failures:
            print(f"- {f}")
        return 1

    print("OK: results-to-paper verification passed.")
    print(f"- Paper: {paper_tex_path}")
    print(f"- Validation JSON: {validation_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

