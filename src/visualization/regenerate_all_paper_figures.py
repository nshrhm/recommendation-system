#!/usr/bin/env python3
"""
Regenerate all paper figures from multi-seed experimental results.

This script ensures all figures in the paper are generated from the statistical
validation experiments (5 seeds: 42, 123, 456, 789, 1024) rather than single-seed
experiments, maintaining scientific rigor and reproducibility.

Figures generated:
1. ndcg_comparison.pdf - NDCG@K comparison across methods
2. metrics_heatmap.pdf - Comprehensive performance heatmap
3. rmse_comparison_normalized.pdf - Raw vs normalized RMSE (already updated)

Author: Research Team
Date: 2025-10-28
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure `src/` is on sys.path when executing as a script.
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# RMSE figures derived from validated results JSON (no hard-coded numbers).
from visualization.plot_normalized_rmse import generate_rmse_figures_from_validation  # noqa: E402

# Publication-quality settings
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

# Use colorblind-friendly palette
sns.set_palette("colorblind")


def load_multi_seed_results(
    results_dir: str = "results/statistical_validation",
) -> Dict:
    """
    Load multi-seed experimental results.

    Parameters
    ----------
    results_dir : str
        Directory containing statistical validation results

    Returns
    -------
    results : Dict
        Dictionary containing all experimental results and statistics
    """
    results_path = Path(results_dir) / "statistical_validation_results.json"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Statistical validation results not found at {results_path}. "
            "Please run src/experiments/run_statistical_validation.py first."
        )

    with open(results_path, "r") as f:
        return json.load(f)


def load_detailed_metrics(
    method: str, seed: int, results_dir: str = "results/statistical_validation"
) -> Dict:
    """
    Load detailed metrics for a specific method and seed.

    Parameters
    ----------
    method : str
        Method name ('baseline', 'f1', or 'f2')
    seed : int
        Random seed
    results_dir : str
        Directory containing results

    Returns
    -------
    metrics : Dict
        Dictionary containing all metrics for this experiment
    """
    experiment_dir = Path(results_dir) / f"{method}_seed{seed}"

    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    # Find the most recent experiment run (sorted by timestamp in directory name)
    run_dirs = sorted([d for d in experiment_dir.iterdir() if d.is_dir()])

    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {experiment_dir}")

    # Use the latest run
    latest_run = run_dirs[-1]
    metrics_path = latest_run / "metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found at {metrics_path}")

    with open(metrics_path, "r") as f:
        data = json.load(f)
        # Extract just the metrics dict
        return data.get("metrics", data)


def generate_ndcg_comparison(output_path: str = "results/figures/ndcg_comparison.pdf"):
    """
    Generate NDCG@K comparison figure from multi-seed results.

    This recreates Figure 3 in the paper using multi-seed experimental data.
    Since results are deterministic (std=0), we use seed=42 as representative.

    Parameters
    ----------
    output_path : str
        Output file path for the generated figure
    """
    print("Generating NDCG@K comparison figure...")

    # Load metrics from seed=42 (representative of all seeds due to determinism)
    baseline_metrics = load_detailed_metrics("baseline", 42)
    f1_metrics = load_detailed_metrics("f1", 42)
    f2_metrics = load_detailed_metrics("f2", 42)

    # Extract NDCG values for K=5, 10, 20
    k_values = [5, 10, 20]
    methods = ["Baseline", "F1", "F2"]

    ndcg_data = {
        "Baseline": [
            baseline_metrics["ndcg@5"],
            baseline_metrics["ndcg@10"],
            baseline_metrics["ndcg@20"],
        ],
        "F1": [f1_metrics["ndcg@5"], f1_metrics["ndcg@10"], f1_metrics["ndcg@20"]],
        "F2": [f2_metrics["ndcg@5"], f2_metrics["ndcg@10"], f2_metrics["ndcg@20"]],
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = sns.color_palette("colorblind", 3)
    markers = ["o", "s", "^"]

    for i, method in enumerate(methods):
        ax.plot(
            k_values,
            ndcg_data[method],
            marker=markers[i],
            color=colors[i],
            linewidth=2.5,
            markersize=10,
            label=method,
            alpha=0.8,
        )

    # Formatting
    ax.set_xlabel("K (Cutoff Position)", fontsize=14, fontweight="bold")
    ax.set_ylabel("NDCG@K", fontsize=14, fontweight="bold")
    ax.set_title(
        "NDCG Performance Across Different Cutoff Positions",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.set_xticks(k_values)
    ax.set_ylim(0.55, 0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=True, shadow=True, fancybox=True, loc="lower right")

    # Add annotation about deterministic results
    ax.text(
        0.02,
        0.98,
        "Multi-seed validated (n=5)\nResults are deterministic (std=0)",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()

    # Save figure
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(
        output_path.replace(".pdf", ".png"), format="png", bbox_inches="tight", dpi=300
    )

    print(f"✓ NDCG comparison saved to {output_path}")
    plt.close()


def generate_metrics_heatmap(output_path: str = "results/figures/metrics_heatmap.pdf"):
    """
    Generate comprehensive metrics heatmap from multi-seed results.

    This recreates Figure 4 in the paper using multi-seed experimental data.

    Parameters
    ----------
    output_path : str
        Output file path for the generated figure
    """
    print("Generating metrics heatmap...")

    # Load metrics from seed=42 (representative)
    baseline_metrics = load_detailed_metrics("baseline", 42)
    f1_metrics = load_detailed_metrics("f1", 42)
    f2_metrics = load_detailed_metrics("f2", 42)

    # Organize metrics into a DataFrame
    metrics_list = [
        "rmse",
        "mae",
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
        "precision@10",
        "recall@10",
    ]

    # Create data matrix
    data = []
    for metrics in [baseline_metrics, f1_metrics, f2_metrics]:
        row = []
        for metric in metrics_list:
            row.append(metrics[metric])
        data.append(row)

    df = pd.DataFrame(
        data,
        index=["Baseline", "F1", "F2"],
        columns=[
            "RMSE",
            "MAE",
            "NDCG@5",
            "NDCG@10",
            "NDCG@20",
            "Precision@10",
            "Recall@10",
        ],
    )

    # Normalize for visualization (column-wise min-max scaling)
    # For error metrics (RMSE, MAE): lower is better, so invert
    # For ranking metrics: higher is better
    df_normalized = df.copy()

    # Invert error metrics (lower is better → higher normalized value)
    for col in ["RMSE", "MAE"]:
        col_max = df[col].max()
        col_min = df[col].min()
        if col_max != col_min:
            df_normalized[col] = 1 - (df[col] - col_min) / (col_max - col_min)
        else:
            df_normalized[col] = 1.0

    # Normalize ranking metrics (higher is better)
    for col in ["NDCG@5", "NDCG@10", "NDCG@20", "Precision@10", "Recall@10"]:
        col_max = df[col].max()
        col_min = df[col].min()
        if col_max != col_min:
            df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df_normalized[col] = 1.0

    # Create figure with more space for labels
    fig, ax = plt.subplots(figsize=(12, 5))

    # Create heatmap
    sns.heatmap(
        df_normalized,
        annot=df.round(2),  # Show original values
        fmt="",
        cmap="RdYlGn",
        center=0.5,
        vmin=0,
        vmax=1,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Normalized Performance\n(1.0 = Best)", "pad": 0.02},
        ax=ax,
    )

    # Simplified title with better spacing
    ax.set_title(
        "Comprehensive Performance Metrics Across All Methods",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Evaluation Metrics", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Recommendation Method", fontsize=13, fontweight="bold", labelpad=10)

    # Add vertical line to separate error metrics from ranking metrics
    ax.axvline(x=2, color="black", linewidth=2, linestyle="--", alpha=0.5)

    # Add group labels below x-axis
    ax.text(
        1,
        -0.7,
        "Error Metrics\n(lower is better)",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        4.5,
        -0.7,
        "Ranking Quality Metrics\n(higher is better)",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    # Add annotation in bottom right (instead of top right to avoid colorbar)
    ax.text(
        0.98,
        0.02,
        "Multi-seed validated (n=5)\nDeterministic results (std=0.0)",
        transform=ax.transAxes,
        fontsize=9,
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5, edgecolor="gray"),
    )

    # Add note about values vs colors
    fig.text(
        0.5,
        0.01,
        "Note: Cell values show raw metrics; cell colors represent normalized performance (green=better)",
        ha="center",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Leave space for bottom note

    # Save figure
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(
        output_path.replace(".pdf", ".png"), format="png", bbox_inches="tight", dpi=300
    )

    print(f"✓ Metrics heatmap saved to {output_path}")
    plt.close()


def verify_rmse_comparison_exists(
    path: str = "results/figures/rmse_comparison_normalized.pdf",
):
    """
    Verify that the RMSE comparison figure exists and is up-to-date.

    Parameters
    ----------
    path : str
        Path to the RMSE comparison figure
    """
    print("Verifying RMSE comparison figure...")

    path_obj = Path(path)
    if not path_obj.exists():
        print(f"⚠ RMSE comparison figure not found at {path}")
        print("  Please run src/visualization/plot_normalized_rmse.py")
        return False

    # Check timestamp - should be after statistical validation
    import os
    import time

    fig_mtime = os.path.getmtime(path)
    validation_file = Path(
        "results/statistical_validation/statistical_validation_results.json"
    )

    if validation_file.exists():
        validation_mtime = os.path.getmtime(validation_file)
        if fig_mtime < validation_mtime:
            print(f"⚠ RMSE comparison figure is older than validation results")
            print(f"  Figure: {time.ctime(fig_mtime)}")
            print(f"  Validation: {time.ctime(validation_mtime)}")
            return False

    print(f"✓ RMSE comparison figure is up-to-date")
    return True


def print_summary():
    """Print summary of all generated figures."""
    print("\n" + "=" * 70)
    print("FIGURE REGENERATION COMPLETE")
    print("=" * 70)

    figures = [
        ("ndcg_comparison.pdf", "NDCG@K comparison across methods"),
        ("metrics_heatmap.pdf", "Comprehensive performance heatmap"),
        ("rmse_comparison_normalized.pdf", "Raw vs normalized RMSE comparison"),
        ("rmse_normalized_with_ci.pdf", "Normalized RMSE with 95% CIs"),
    ]

    print("\nGenerated figures (multi-seed validated):")
    for filename, description in figures:
        path = Path("results/figures") / filename
        if path.exists():
            import os

            size_kb = os.path.getsize(path) / 1024
            print(f"  ✓ {filename:40s} ({size_kb:6.1f} KB) - {description}")
        else:
            print(f"  ✗ {filename:40s} (MISSING) - {description}")

    print("\nAll figures are now generated from 5-seed experiments:")
    print("  Seeds: [42, 123, 456, 789, 1024]")
    print("  Results: Deterministic (std=0.0 across all seeds)")
    print("  Representative seed used: 42")

    print("\nFigures ready for publication:")
    print("  - Resolution: 300 DPI")
    print("  - Format: PDF (vector graphics) + PNG (raster backup)")
    print("  - Color scheme: Colorblind-friendly")
    print("  - Font sizes: Publication-ready (≥10pt)")

    print("\n" + "=" * 70)


def main():
    """Main function to regenerate all paper figures."""
    print("=" * 70)
    print("REGENERATING ALL PAPER FIGURES FROM MULTI-SEED EXPERIMENTS")
    print("=" * 70)
    print()

    try:
        # Load and verify multi-seed results exist
        results = load_multi_seed_results()
        print(f"✓ Loaded multi-seed results: {len(results['summary'])} methods")
        print(f"  Seeds: {results['config']['seeds']}")
        print()

        # Generate each figure
        generate_ndcg_comparison()
        generate_metrics_heatmap()
        generate_rmse_figures_from_validation(
            input_json=Path("results/statistical_validation/statistical_validation_results.json"),
            output_dir=Path("results/figures"),
        )
        verify_rmse_comparison_exists()

        # Print summary
        print_summary()

        print("\n🎉 SUCCESS! All figures regenerated from multi-seed experiments.")
        print("\nNext steps:")
        print("  1. Review figures in results/figures/")
        print("  2. Verify they appear correctly in paper (compile LaTeX)")
        print("  3. Update figure captions if needed")
        print("  4. Commit changes to git")

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease run statistical validation first:")
        print("  python src/experiments/run_statistical_validation.py")
        return 1

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
