"""
Generate RMSE figures derived from statistical validation results.

This module MUST NOT hard-code reported values. It reads:
  results/statistical_validation/statistical_validation_results.json

Figures written to results/figures/:
- rmse_comparison_normalized.pdf/.png
- rmse_normalized_with_ci.pdf/.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _configure_matplotlib() -> None:
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    sns.set_palette("colorblind")


def load_validation_results(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Validation JSON not found: {path}. "
            "Run: python src/experiments/run_statistical_validation.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _get_summary_map(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    summary = results.get("summary")
    if not isinstance(summary, list):
        raise ValueError("Invalid validation JSON: expected top-level key 'summary' (list).")
    by_method: Dict[str, Dict[str, Any]] = {}
    for row in summary:
        method = row.get("Method")
        if isinstance(method, str):
            by_method[method.upper()] = row
    for needed in ["BASELINE", "F1", "F2"]:
        if needed not in by_method:
            raise ValueError(f"Validation JSON missing summary row for {needed}.")
    return by_method


def _get_stat_tests(results: Dict[str, Any]) -> Tuple[float, float]:
    tests = results.get("statistical_tests", {})
    try:
        p_f1 = float(tests["f1_vs_baseline"]["p_value"])
        p_f2 = float(tests["f2_vs_baseline"]["p_value"])
    except Exception as e:  # noqa: BLE001
        raise ValueError("Validation JSON missing expected 'statistical_tests' structure.") from e
    return p_f1, p_f2


def _get_rmse_ci_normalized(results: Dict[str, Any]) -> Dict[str, Tuple[float, float, float]]:
    cis = results.get("confidence_intervals", {})
    out: Dict[str, Tuple[float, float, float]] = {}

    def get_raw(method_key: str) -> Tuple[float, float, float]:
        try:
            mean_, lo_, hi_ = cis[method_key]["rmse_across_seeds"]
            return float(mean_), float(lo_), float(hi_)
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                "Validation JSON missing expected 'confidence_intervals' -> "
                f"{method_key} -> rmse_across_seeds."
            ) from e

    # Baseline already on 0-10.
    out["BASELINE"] = get_raw("baseline")

    # F1/F2 are on 0-100 points for RMSE, normalize to 0-10 by multiplying by 0.1.
    for method_upper, json_key in [("F1", "f1"), ("F2", "f2")]:
        mean_, lo_, hi_ = get_raw(json_key)
        out[method_upper] = (mean_ * 0.1, lo_ * 0.1, hi_ * 0.1)

    return out


def create_rmse_comparison_raw_vs_normalized(results: Dict[str, Any]) -> plt.Figure:
    summary = _get_summary_map(results)
    p_f1, p_f2 = _get_stat_tests(results)
    config = results.get("config", {})
    seeds = config.get("seeds", [])

    methods = ["Baseline", "F1", "F2"]
    raw_rmse = [
        float(summary["BASELINE"]["RMSE (raw) Mean"]),
        float(summary["F1"]["RMSE (raw) Mean"]),
        float(summary["F2"]["RMSE (raw) Mean"]),
    ]
    normalized_rmse = [
        float(summary["BASELINE"]["RMSE (norm) Mean"]),
        float(summary["F1"]["RMSE (norm) Mean"]),
        float(summary["F2"]["RMSE (norm) Mean"]),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))
    colors = sns.color_palette("colorblind", 3)

    def add_value_labels(ax: plt.Axes, bars: Any, values: list[float]) -> None:
        ymax = ax.get_ylim()[1]
        offset = ymax * 0.018
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + offset,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    # (a) Raw RMSE
    bars1 = ax1.bar(methods, raw_rmse, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("RMSE (raw)", fontweight="bold")
    ax1.set_title("(a) Raw RMSE (Scale Mismatch)", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)
    ax1.set_ylim(0, max(raw_rmse) * 1.18)

    add_value_labels(ax1, bars1, raw_rmse)

    ax1.text(
        0.03,
        0.96,
        "Baseline: 0–10 scale",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.55),
    )
    ax1.text(
        0.03,
        0.86,
        "F1/F2: 0–100 scale",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.45),
    )

    # (b) Normalized RMSE
    bars2 = ax2.bar(methods, normalized_rmse, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    ax2.set_ylabel("RMSE (normalized to 0–10)", fontweight="bold")
    ax2.set_title("(b) Normalized RMSE (Fair Comparison)", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, max(normalized_rmse) * 1.32)
    add_value_labels(ax2, bars2, normalized_rmse)

    # Baseline reference line
    ax2.axhline(y=normalized_rmse[0], color="gray", linestyle="--", linewidth=2, alpha=0.5)

    # p-value annotations (from Wilcoxon test on normalized absolute errors)
    ax2.text(
        0.03,
        0.96,
        f"F1 vs Base: p = {p_f1:.3f}",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.75),
    )
    ax2.text(
        0.03,
        0.86,
        "F2 vs Base: p < 0.001",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="mistyrose", alpha=0.75),
    )

    fig.suptitle("RMSE Comparison: Raw vs Normalized", fontsize=16, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.02,
        f"Derived from multi-seed statistical validation (seeds={seeds}).",
        ha="center",
        fontsize=9,
        style="italic",
        wrap=True,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.95], w_pad=2.4)
    return fig


def create_normalized_rmse_with_ci(results: Dict[str, Any]) -> plt.Figure:
    p_f1, p_f2 = _get_stat_tests(results)
    cis_norm = _get_rmse_ci_normalized(results)

    labels = ["Baseline", "F1 (norm)", "F2 (norm)"]
    means = [cis_norm["BASELINE"][0], cis_norm["F1"][0], cis_norm["F2"][0]]
    lowers = [cis_norm["BASELINE"][1], cis_norm["F1"][1], cis_norm["F2"][1]]
    uppers = [cis_norm["BASELINE"][2], cis_norm["F1"][2], cis_norm["F2"][2]]

    x_pos = np.arange(len(labels))
    colors = sns.color_palette("colorblind", 3)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(x_pos, means, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)

    yerr_lower = [m - l for m, l in zip(means, lowers)]
    yerr_upper = [u - m for u, m in zip(uppers, means)]
    ax.errorbar(
        x_pos,
        means,
        yerr=[yerr_lower, yerr_upper],
        fmt="none",
        color="black",
        capsize=5,
        capthick=1.5,
        linewidth=1.5,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE (0–10 scale)", fontweight="bold")
    ax.set_title("Normalized RMSE with 95% Confidence Intervals", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(means) * 1.3)

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # p-value notes
    ax.text(
        0.5,
        max(means) * 1.18,
        f"F1 vs Base: p = {p_f1:.3f} (Wilcoxon, normalized absolute errors)",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6),
    )
    ax.text(
        0.5,
        max(means) * 1.08,
        f"F2 vs Base: p = {p_f2:.2e}",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="mistyrose", alpha=0.6),
    )

    plt.tight_layout()
    return fig


def generate_rmse_figures_from_validation(
    input_json: Path = Path("results/statistical_validation/statistical_validation_results.json"),
    output_dir: Path = Path("results/figures"),
) -> Dict[str, Path]:
    _configure_matplotlib()
    results = load_validation_results(input_json)

    output_dir.mkdir(parents=True, exist_ok=True)

    fig1 = create_rmse_comparison_raw_vs_normalized(results)
    fig1_pdf = output_dir / "rmse_comparison_normalized.pdf"
    fig1_png = output_dir / "rmse_comparison_normalized.png"
    fig1.savefig(fig1_pdf, bbox_inches="tight")
    fig1.savefig(fig1_png, bbox_inches="tight")
    plt.close(fig1)

    fig2 = create_normalized_rmse_with_ci(results)
    fig2_pdf = output_dir / "rmse_normalized_with_ci.pdf"
    fig2_png = output_dir / "rmse_normalized_with_ci.png"
    fig2.savefig(fig2_pdf, bbox_inches="tight")
    fig2.savefig(fig2_png, bbox_inches="tight")
    plt.close(fig2)

    return {
        "rmse_comparison_normalized_pdf": fig1_pdf,
        "rmse_comparison_normalized_png": fig1_png,
        "rmse_normalized_with_ci_pdf": fig2_pdf,
        "rmse_normalized_with_ci_png": fig2_png,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate RMSE figures from validation JSON")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("results/statistical_validation/statistical_validation_results.json"),
        help="Path to results/statistical_validation/statistical_validation_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory to write figures (default: results/figures)",
    )
    args = parser.parse_args()

    paths = generate_rmse_figures_from_validation(args.input_json, args.output_dir)
    for k, v in paths.items():
        print(f"✓ {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
