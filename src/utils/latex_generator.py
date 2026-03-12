"""
LaTeX Table and Content Generator

Automatically generates publication-quality LaTeX tables and content
from experimental results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


def format_float(value: float, precision: int = 4, bold_if_best: bool = False) -> str:
    """
    Format a float for LaTeX table.

    Parameters
    ----------
    value : float
        Value to format
    precision : int
        Number of decimal places
    bold_if_best : bool
        If True, wrap in \\textbf{}

    Returns
    -------
    formatted : str
        LaTeX-formatted string
    """
    formatted = f"{value:.{precision}f}"
    if bold_if_best:
        formatted = f"\\textbf{{{formatted}}}"
    return formatted


def generate_results_table(
    results_dict: Dict[str, Dict],
    metrics: List[str] = ["rmse", "mae", "ndcg@10", "precision@10"],
    caption: str = "Performance comparison of recommendation methods",
    label: str = "tab:results",
    highlight_best: bool = True,
    precision: int = 4,
) -> str:
    """
    Generate LaTeX table from experimental results.

    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping method names to results
    metrics : List[str]
        List of metrics to include
    caption : str
        Table caption
    label : str
        LaTeX label for referencing
    highlight_best : bool
        If True, bold the best value for each metric
    precision : int
        Number of decimal places

    Returns
    -------
    latex_code : str
        LaTeX table code

    Examples
    --------
    >>> results = {
    ...     'baseline': {'metrics': {'rmse': 2.5, 'mae': 2.0}},
    ...     'f1': {'metrics': {'rmse': 2.2, 'mae': 1.8}}
    ... }
    >>> latex = generate_results_table(results, metrics=['rmse', 'mae'])
    >>> print(latex)
    """
    lines = []

    # Table preamble
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")

    # Column specification
    n_metrics = len(metrics)
    col_spec = "l" + "c" * n_metrics
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")

    # Header row
    metric_names = [metric.upper().replace("@", "@") for metric in metrics]
    header = "    Method & " + " & ".join(metric_names) + " \\\\"
    lines.append(header)
    lines.append("    \\midrule")

    # Determine best values for each metric
    best_values = {}
    if highlight_best:
        for metric in metrics:
            values = []
            for method_results in results_dict.values():
                if metric in method_results["metrics"]:
                    values.append(method_results["metrics"][metric])

            if metric in ["rmse", "mae"]:  # Lower is better
                best_values[metric] = min(values) if values else None
            else:  # Higher is better
                best_values[metric] = max(values) if values else None

    # Data rows
    for method, method_results in results_dict.items():
        row_values = [method.upper()]

        for metric in metrics:
            if metric in method_results["metrics"]:
                value = method_results["metrics"][metric]
                is_best = (
                    highlight_best
                    and best_values.get(metric) is not None
                    and abs(value - best_values[metric]) < 1e-6
                )
                row_values.append(format_float(value, precision, is_best))
            else:
                row_values.append("---")

        row = "    " + " & ".join(row_values) + " \\\\"
        lines.append(row)

    # Table footer
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_comparison_table_with_pvalues(
    results_dict: Dict[str, Dict],
    pvalues_df: pd.DataFrame,
    metric: str = "rmse",
    caption: str = "Pairwise comparison with statistical significance",
    label: str = "tab:comparison",
    significance_level: float = 0.05,
) -> str:
    """
    Generate LaTeX table with pairwise comparisons and p-values.

    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping method names to results
    pvalues_df : pd.DataFrame
        DataFrame with pairwise p-values (from statistical_tests.py)
    metric : str
        Metric name
    caption : str
        Table caption
    label : str
        LaTeX label
    significance_level : float
        Significance level for marking

    Returns
    -------
    latex_code : str
        LaTeX table code
    """
    lines = []

    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")

    methods = list(results_dict.keys())
    n_methods = len(methods)

    # Column specification
    col_spec = "l" + "c" * n_methods
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("    \\toprule")

    # Header row
    header = "    Method & " + " & ".join([m.upper() for m in methods]) + " \\\\"
    lines.append(header)
    lines.append("    \\midrule")

    # Create p-value matrix
    pvalue_matrix = np.ones((n_methods, n_methods))
    for _, row in pvalues_df.iterrows():
        method_a = row["Method A"]
        method_b = row["Method B"]
        p_value = row["P-value"]

        if method_a in methods and method_b in methods:
            i = methods.index(method_a)
            j = methods.index(method_b)
            pvalue_matrix[i, j] = p_value
            pvalue_matrix[j, i] = p_value

    # Data rows
    for i, method in enumerate(methods):
        row_values = [method.upper()]

        for j in range(n_methods):
            if i == j:
                # Diagonal: show metric value
                value = results_dict[method]["metrics"][metric]
                row_values.append(format_float(value, precision=4))
            else:
                # Off-diagonal: show p-value
                p_value = pvalue_matrix[i, j]
                if p_value < significance_level:
                    # Significant difference
                    row_values.append(f"\\textbf{{{p_value:.4f}}}$^*$")
                else:
                    row_values.append(f"{p_value:.4f}")

        row = "    " + " & ".join(row_values) + " \\\\"
        lines.append(row)

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  \\vspace{0.5em}")
    lines.append("  \\\\")
    lines.append(
        "  \\footnotesize{Diagonal: metric values; Off-diagonal: p-values ($^*$p < 0.05)}"
    )
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_experimental_setup_table(
    n_users: int = 4,
    n_items: int = 50,
    n_rated: int = 30,
    n_test: int = 20,
    rating_scale: Tuple[int, int] = (0, 10),
    caption: str = "Experimental setup",
    label: str = "tab:setup",
) -> str:
    """
    Generate LaTeX table describing experimental setup.

    Parameters
    ----------
    n_users : int
        Number of users
    n_items : int
        Total number of items
    n_rated : int
        Number of items rated by target user
    n_test : int
        Number of test items
    rating_scale : Tuple[int, int]
        Rating scale (min, max)
    caption : str
        Table caption
    label : str
        LaTeX label

    Returns
    -------
    latex_code : str
        LaTeX table code
    """
    lines = []

    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\begin{tabular}{ll}")
    lines.append("    \\toprule")
    lines.append("    Parameter & Value \\\\")
    lines.append("    \\midrule")
    lines.append(f"    Number of users & {n_users} \\\\")
    lines.append(f"    Total items & {n_items} \\\\")
    lines.append(f"    Items rated (training) & {n_rated} \\\\")
    lines.append(f"    Items for testing & {n_test} \\\\")
    lines.append(f"    Rating scale & {rating_scale[0]}--{rating_scale[1]} \\\\")
    lines.append("    Similarity metric & Cosine \\\\")
    lines.append("    k-neighbors & 3 \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_results_section(
    results_dict: Dict[str, Dict], output_file: Optional[str] = None
) -> str:
    """
    Generate LaTeX content for Results section.

    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping method names to results
    output_file : Optional[str]
        If provided, save to file

    Returns
    -------
    latex_code : str
        LaTeX content for results section
    """
    lines = []

    lines.append("\\section{Results}")
    lines.append("\\label{sec:results}")
    lines.append("")

    lines.append(
        "We evaluated the proposed recommendation system using a dataset of "
        "cosmetics ratings from four users across 50 items. User A rated 30 items "
        "for training, and the remaining 20 items were held out for testing."
    )
    lines.append("")

    # Performance comparison
    lines.append("\\subsection{Performance Comparison}")
    lines.append("")

    # Find best method
    best_rmse = float("inf")
    best_method = None
    for method, results in results_dict.items():
        rmse = results["metrics"]["rmse"]
        if rmse < best_rmse:
            best_rmse = rmse
            best_method = method

    lines.append(
        f"Table~\\ref{{tab:results}} presents the performance comparison across all methods. "
        f"The {best_method.upper()} method achieved the lowest RMSE of {best_rmse:.4f}, "
        f"indicating superior prediction accuracy."
    )
    lines.append("")

    # Generate results table
    results_table = generate_results_table(
        results_dict,
        metrics=["rmse", "mae", "ndcg@5", "ndcg@10", "ndcg@20"],
        caption="Performance comparison of recommendation methods",
        label="tab:results",
    )
    lines.append(results_table)
    lines.append("")

    # Method-specific analysis
    lines.append("\\subsection{Method-Specific Analysis}")
    lines.append("")

    for method, results in results_dict.items():
        rmse = results["metrics"]["rmse"]
        ndcg10 = results["metrics"]["ndcg@10"]

        lines.append(
            f"\\textbf{{{method.upper()}}}: "
            f"This method achieved an RMSE of {rmse:.4f} and NDCG@10 of {ndcg10:.4f}. "
        )

        if method == "f1":
            lines.append(
                "The F1 method incorporates user impression ranking (Equation 5), "
                "which combines User-based CF, Item-based CF, and user feedback."
            )
        elif method == "f2":
            lines.append(
                "The F2 method utilizes preference vectors (Equations 6 and 11) to "
                "adjust recommendations based on user preferences."
            )

        lines.append("")

    # Figures reference
    lines.append(
        "Figure~\\ref{fig:rmse_comparison} shows the RMSE comparison across methods, "
        "while Figure~\\ref{fig:ndcg_comparison} illustrates the ranking quality "
        "measured by NDCG@K for different values of K."
    )
    lines.append("")

    latex_content = "\n".join(lines)

    if output_file:
        with open(output_file, "w") as f:
            f.write(latex_content)
        print(f"LaTeX results section saved to: {output_file}")

    return latex_content


def generate_figure_references() -> str:
    """
    Generate LaTeX figure references.

    Returns
    -------
    latex_code : str
        LaTeX figure inclusion code
    """
    lines = []

    figures = [
        ("rmse_comparison", "RMSE comparison across recommendation methods"),
        ("ndcg_comparison", "NDCG@K comparison across recommendation methods"),
        ("metrics_heatmap", "Performance metrics heatmap"),
        ("precision_recall", "Precision-recall curves"),
    ]

    for i, (fig_name, caption) in enumerate(figures, 1):
        lines.append("\\begin{figure}[htbp]")
        lines.append("  \\centering")
        lines.append(
            f"  \\includegraphics[width=0.8\\textwidth]{{figures/{fig_name}.pdf}}"
        )
        lines.append(f"  \\caption{{{caption}}}")
        lines.append(f"  \\label{{fig:{fig_name}}}")
        lines.append("\\end{figure}")
        lines.append("")

    return "\n".join(lines)


def save_all_latex_tables(
    results_dir: str = "results", output_dir: str = "paper/tables"
) -> None:
    """
    Generate and save all LaTeX tables.

    Parameters
    ----------
    results_dir : str
        Directory containing experiment results
    output_dir : str
        Directory to save LaTeX files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating LaTeX tables...")

    # Load results
    results_dict = {}
    results_path = Path(results_dir)

    for method in ["baseline", "f1", "f2"]:
        method_dirs = sorted(results_path.glob(f"{method}_*"))
        if method_dirs:
            latest_dir = method_dirs[-1]
            metrics_file = latest_dir / "metrics.json"

            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    results_dict[method] = json.load(f)

    if not results_dict:
        print("No results found!")
        return

    # Generate tables
    print("\n1. Results table...")
    results_table = generate_results_table(results_dict)
    with open(output_path / "results_table.tex", "w") as f:
        f.write(results_table)
    print(f"   Saved: {output_path / 'results_table.tex'}")

    print("\n2. Experimental setup table...")
    setup_table = generate_experimental_setup_table()
    with open(output_path / "setup_table.tex", "w") as f:
        f.write(setup_table)
    print(f"   Saved: {output_path / 'setup_table.tex'}")

    print("\n3. Results section...")
    results_section = generate_results_section(results_dict)
    with open(output_path / "results_section.tex", "w") as f:
        f.write(results_section)
    print(f"   Saved: {output_path / 'results_section.tex'}")

    print("\n4. Figure references...")
    figures_code = generate_figure_references()
    with open(output_path / "figures_code.tex", "w") as f:
        f.write(figures_code)
    print(f"   Saved: {output_path / 'figures_code.tex'}")

    print(f"\nAll LaTeX tables saved to: {output_dir}")


def main():
    """Main entry point for LaTeX generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from experiment results"
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
        default="paper/tables",
        help="Output directory for LaTeX files",
    )

    args = parser.parse_args()

    save_all_latex_tables(results_dir=args.results_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
