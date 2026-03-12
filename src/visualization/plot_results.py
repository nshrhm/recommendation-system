"""
Visualization Module for Recommendation System

Generates publication-quality figures for academic papers.

Requirements:
- 300 DPI or higher
- Colorblind-friendly palettes
- Consistent fonts (minimum 10pt)
- PDF/EPS format for LaTeX
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Set publication-quality defaults
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    }
)

# Use colorblind-friendly palette
sns.set_palette("colorblind")


def load_experiment_results(results_dir: str) -> Dict[str, Dict]:
    """
    Load results from multiple experiment runs.

    Parameters
    ----------
    results_dir : str
        Directory containing experiment results

    Returns
    -------
    results : Dict[str, Dict]
        Dictionary mapping method names to results
    """
    results_path = Path(results_dir)
    results = {}

    for method in ["baseline", "f1", "f2"]:
        # Find most recent result for this method
        method_dirs = sorted(results_path.glob(f"{method}_*"))
        if method_dirs:
            latest_dir = method_dirs[-1]
            metrics_file = latest_dir / "metrics.json"

            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    results[method] = json.load(f)

    return results


def plot_rmse_comparison(
    results: Dict[str, Dict],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Figure:
    """
    Plot RMSE comparison across methods.

    Parameters
    ----------
    results : Dict[str, Dict]
        Experiment results
    output_path : Optional[str]
        Path to save figure (if None, display only)
    figsize : Tuple[float, float]
        Figure size in inches

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    methods = []
    rmse_values = []
    colors = sns.color_palette("colorblind", n_colors=3)

    for method, data in results.items():
        methods.append(method.upper())
        rmse_values.append(data["metrics"]["rmse"])

    # Create bar plot
    bars = ax.bar(
        methods, rmse_values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
    )

    # Add value labels on bars
    for bar, value in zip(bars, rmse_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("RMSE (Root Mean Square Error)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Method", fontsize=14, fontweight="bold")
    ax.set_title(
        "RMSE Comparison Across Methods", fontsize=16, fontweight="bold", pad=20
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved RMSE comparison plot to {output_path}")

    return fig


def plot_ndcg_comparison(
    results: Dict[str, Dict],
    k_values: List[int] = [5, 10, 20],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    Plot NDCG@K comparison across methods.

    Parameters
    ----------
    results : Dict[str, Dict]
        Experiment results
    k_values : List[int]
        K values for NDCG
    output_path : Optional[str]
        Path to save figure
    figsize : Tuple[float, float]
        Figure size in inches

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    methods = list(results.keys())
    x = np.arange(len(k_values))
    width = 0.25

    colors = sns.color_palette("colorblind", n_colors=len(methods))

    for i, (method, data) in enumerate(results.items()):
        ndcg_values = [data["metrics"][f"ndcg@{k}"] for k in k_values]
        offset = width * (i - len(methods) / 2 + 0.5)
        bars = ax.bar(
            x + offset,
            ndcg_values,
            width,
            label=method.upper(),
            color=colors[i],
            alpha=0.8,
            edgecolor="black",
            linewidth=1.2,
        )

        # Add value labels
        for bar, value in zip(bars, ndcg_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel("NDCG (Normalized DCG)", fontsize=14, fontweight="bold")
    ax.set_xlabel("K (Top-K Items)", fontsize=14, fontweight="bold")
    ax.set_title(
        "NDCG@K Comparison Across Methods", fontsize=16, fontweight="bold", pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"@{k}" for k in k_values])
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="black")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved NDCG comparison plot to {output_path}")

    return fig


def plot_all_metrics_heatmap(
    results: Dict[str, Dict],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
) -> plt.Figure:
    """
    Plot heatmap of all metrics across methods.

    Parameters
    ----------
    results : Dict[str, Dict]
        Experiment results
    output_path : Optional[str]
        Path to save figure
    figsize : Tuple[float, float]
        Figure size in inches

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    # Extract metrics
    methods = []
    metric_names = [
        "rmse",
        "mae",
        "ndcg@5",
        "ndcg@10",
        "precision@5",
        "precision@10",
        "recall@5",
        "recall@10",
    ]
    data = []

    for method, result in results.items():
        methods.append(method.upper())
        row = []
        for metric in metric_names:
            row.append(result["metrics"][metric])
        data.append(row)

    df = pd.DataFrame(data, index=methods, columns=metric_names)

    # Normalize each column to 0-1 for better visualization
    df_normalized = df.copy()
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)

    # For error metrics (RMSE, MAE), invert so lower is better (appears darker)
    for col in ["rmse", "mae"]:
        df_normalized[col] = 1 - df_normalized[col]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df_normalized,
        annot=df,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Normalized Score (higher is better)"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    ax.set_title("Performance Metrics Heatmap", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Methods", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved metrics heatmap to {output_path}")

    return fig


def plot_precision_recall_curve(
    results: Dict[str, Dict],
    k_values: List[int] = [5, 10, 15, 20],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    Plot Precision-Recall curve for different K values.

    Parameters
    ----------
    results : Dict[str, Dict]
        Experiment results
    k_values : List[int]
        K values to plot
    output_path : Optional[str]
        Path to save figure
    figsize : Tuple[float, float]
        Figure size in inches

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("colorblind", n_colors=len(results))
    markers = ["o", "s", "^", "D", "v", "<", ">"]

    for i, (method, data) in enumerate(results.items()):
        precisions = []
        recalls = []

        for k in k_values:
            precision_key = f"precision@{k}"
            recall_key = f"recall@{k}"

            if precision_key in data["metrics"] and recall_key in data["metrics"]:
                precisions.append(data["metrics"][precision_key])
                recalls.append(data["metrics"][recall_key])

        ax.plot(
            recalls,
            precisions,
            marker=markers[i % len(markers)],
            markersize=10,
            linewidth=2.5,
            label=method.upper(),
            color=colors[i],
            alpha=0.8,
        )

    ax.set_xlabel("Recall", fontsize=14, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=14, fontweight="bold")
    ax.set_title("Precision-Recall Curve", fontsize=16, fontweight="bold", pad=20)
    ax.legend(loc="best", framealpha=0.9, edgecolor="black")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved precision-recall curve to {output_path}")

    return fig


def generate_all_figures(
    results_dir: str = "results",
    output_dir: str = "results/figures",
    formats: List[str] = ["pdf", "png"],
) -> None:
    """
    Generate all publication-quality figures.

    Parameters
    ----------
    results_dir : str
        Directory containing experiment results
    output_dir : str
        Directory to save figures
    formats : List[str]
        Output formats (e.g., ['pdf', 'png'])
    """
    # Load results
    print("Loading experiment results...")
    results = load_experiment_results(results_dir)

    if not results:
        print("No experiment results found!")
        return

    print(f"Found results for methods: {list(results.keys())}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")

    # Generate each figure
    figures = {
        "rmse_comparison": lambda: plot_rmse_comparison(results),
        "ndcg_comparison": lambda: plot_ndcg_comparison(results),
        "metrics_heatmap": lambda: plot_all_metrics_heatmap(results),
        "precision_recall": lambda: plot_precision_recall_curve(results),
    }

    for fig_name, fig_func in figures.items():
        print(f"\n  Generating {fig_name}...")
        fig = fig_func()

        for fmt in formats:
            output_file = output_path / f"{fig_name}.{fmt}"
            fig.savefig(output_file, dpi=300, bbox_inches="tight", format=fmt)
            print(f"    Saved: {output_file}")

        plt.close(fig)

    print(f"\nAll figures saved to: {output_dir}")


def main():
    """Main entry point for visualization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate visualizations for recommendation system experiments"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["pdf", "png"],
        help="Output formats (pdf, png, eps, svg)",
    )

    args = parser.parse_args()

    generate_all_figures(
        results_dir=args.results_dir, output_dir=args.output_dir, formats=args.formats
    )


if __name__ == "__main__":
    main()
