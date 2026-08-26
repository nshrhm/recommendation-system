"""
Statistical Testing for Recommendation System Experiments

Provides statistical tests to determine significance of differences
between methods.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import pandas as pd


def paired_t_test(
    method_a_scores: np.ndarray,
    method_b_scores: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Perform paired t-test between two methods.

    Used when comparing the same items evaluated by different methods.

    Parameters
    ----------
    method_a_scores : np.ndarray
        Scores from method A
    method_b_scores : np.ndarray
        Scores from method B
    alternative : {'two-sided', 'less', 'greater'}
        Alternative hypothesis

    Returns
    -------
    statistic : float
        t-statistic
    p_value : float
        p-value

    Examples
    --------
    >>> method_a = np.array([2.5, 3.0, 2.8, 3.2])
    >>> method_b = np.array([2.2, 2.8, 2.5, 3.0])
    >>> t_stat, p_val = paired_t_test(method_a, method_b)
    >>> print(f"p-value: {p_val:.4f}")
    p-value: 0.0123
    """
    if len(method_a_scores) != len(method_b_scores):
        raise ValueError("Arrays must have the same length for paired t-test")

    result = stats.ttest_rel(method_a_scores, method_b_scores, alternative=alternative)
    return result.statistic, result.pvalue


def wilcoxon_signed_rank_test(
    method_a_scores: np.ndarray,
    method_b_scores: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Used when data may not be normally distributed.

    Parameters
    ----------
    method_a_scores : np.ndarray
        Scores from method A
    method_b_scores : np.ndarray
        Scores from method B
    alternative : {'two-sided', 'less', 'greater'}
        Alternative hypothesis

    Returns
    -------
    statistic : float
        Test statistic
    p_value : float
        p-value

    Examples
    --------
    >>> method_a = np.array([2.5, 3.0, 2.8, 3.2, 2.9])
    >>> method_b = np.array([2.2, 2.8, 2.5, 3.0, 2.7])
    >>> w_stat, p_val = wilcoxon_signed_rank_test(method_a, method_b)
    >>> print(f"p-value: {p_val:.4f}")
    """
    if len(method_a_scores) != len(method_b_scores):
        raise ValueError("Arrays must have the same length for Wilcoxon test")

    result = stats.wilcoxon(method_a_scores, method_b_scores, alternative=alternative)
    return result.statistic, result.pvalue


def mcnemar_test(
    method_a_correct: np.ndarray, method_b_correct: np.ndarray
) -> Tuple[float, float]:
    """
    Perform McNemar's test for comparing binary classifications.

    Used for comparing whether items are correctly recommended (binary outcome).

    Parameters
    ----------
    method_a_correct : np.ndarray
        Binary array (1=correct, 0=incorrect) for method A
    method_b_correct : np.ndarray
        Binary array (1=correct, 0=incorrect) for method B

    Returns
    -------
    statistic : float
        Test statistic
    p_value : float
        p-value

    Examples
    --------
    >>> method_a = np.array([1, 1, 0, 1, 0])
    >>> method_b = np.array([1, 0, 0, 1, 1])
    >>> stat, p_val = mcnemar_test(method_a, method_b)
    """
    if len(method_a_correct) != len(method_b_correct):
        raise ValueError("Arrays must have the same length")

    # Create contingency table
    # [[both_correct, a_correct_b_incorrect],
    #  [a_incorrect_b_correct, both_incorrect]]
    both_correct = np.sum((method_a_correct == 1) & (method_b_correct == 1))
    a_correct_b_incorrect = np.sum((method_a_correct == 1) & (method_b_correct == 0))
    a_incorrect_b_correct = np.sum((method_a_correct == 0) & (method_b_correct == 1))
    both_incorrect = np.sum((method_a_correct == 0) & (method_b_correct == 0))

    table = np.array(
        [[both_correct, a_correct_b_incorrect], [a_incorrect_b_correct, both_incorrect]]
    )

    # McNemar's test
    result = stats.mcnemar(table, exact=False)
    return result.statistic, result.pvalue


def compare_all_methods(
    results_dict: Dict[str, np.ndarray],
    test_type: str = "paired_t",
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """
    Perform pairwise statistical tests between all methods.

    Parameters
    ----------
    results_dict : Dict[str, np.ndarray]
        Dictionary mapping method names to score arrays
    test_type : {'paired_t', 'wilcoxon'}
        Type of statistical test
    significance_level : float
        Significance level (default: 0.05)

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with pairwise comparison results

    Examples
    --------
    >>> results = {
    ...     'baseline': np.array([2.5, 3.0, 2.8]),
    ...     'f1': np.array([2.2, 2.8, 2.5]),
    ...     'f2': np.array([2.3, 2.9, 2.6])
    ... }
    >>> comparison_df = compare_all_methods(results, test_type='paired_t')
    >>> print(comparison_df)
    """
    methods = list(results_dict.keys())
    n_methods = len(methods)

    comparison_results = []

    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            method_a = methods[i]
            method_b = methods[j]

            scores_a = results_dict[method_a]
            scores_b = results_dict[method_b]

            # Perform test
            if test_type == "paired_t":
                statistic, p_value = paired_t_test(scores_a, scores_b)
            elif test_type == "wilcoxon":
                statistic, p_value = wilcoxon_signed_rank_test(scores_a, scores_b)
            else:
                raise ValueError(f"Unknown test type: {test_type}")

            # Determine significance
            is_significant = p_value < significance_level

            # Calculate effect size (Cohen's d)
            mean_diff = np.mean(scores_a) - np.mean(scores_b)
            pooled_std = np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            comparison_results.append(
                {
                    "Method A": method_a,
                    "Method B": method_b,
                    "Mean A": np.mean(scores_a),
                    "Mean B": np.mean(scores_b),
                    "Mean Difference": mean_diff,
                    "Statistic": statistic,
                    "P-value": p_value,
                    "Significant": is_significant,
                    "Cohen's d": cohens_d,
                }
            )

    results_df = pd.DataFrame(comparison_results)
    return results_df


def effect_size_interpretation(cohens_d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Parameters
    ----------
    cohens_d : float
        Cohen's d value

    Returns
    -------
    interpretation : str
        Interpretation of effect size

    References
    ----------
    Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
    """
    abs_d = abs(cohens_d)

    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def generate_statistical_report(
    results_dict: Dict[str, np.ndarray],
    metric_name: str = "RMSE",
    test_type: str = "paired_t",
    significance_level: float = 0.05,
    output_file: Optional[str] = None,
) -> str:
    """
    Generate a comprehensive statistical comparison report.

    Parameters
    ----------
    results_dict : Dict[str, np.ndarray]
        Dictionary mapping method names to score arrays
    metric_name : str
        Name of the metric being compared
    test_type : {'paired_t', 'wilcoxon'}
        Type of statistical test
    significance_level : float
        Significance level
    output_file : Optional[str]
        If provided, save report to file

    Returns
    -------
    report : str
        Formatted statistical report

    Examples
    --------
    >>> results = {
    ...     'baseline': np.array([2.5, 3.0, 2.8]),
    ...     'f1': np.array([2.2, 2.8, 2.5])
    ... }
    >>> report = generate_statistical_report(results, metric_name='RMSE')
    >>> print(report)
    """
    report_lines = []

    report_lines.append("=" * 70)
    report_lines.append(f"STATISTICAL SIGNIFICANCE TESTING: {metric_name}")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Descriptive statistics
    report_lines.append("Descriptive Statistics:")
    report_lines.append("-" * 70)
    for method, scores in results_dict.items():
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        median = np.median(scores)
        report_lines.append(
            f"{method.upper():15} Mean: {mean:8.4f}  Std: {std:8.4f}  Median: {median:8.4f}"
        )
    report_lines.append("")

    # Pairwise comparisons
    comparison_df = compare_all_methods(results_dict, test_type, significance_level)

    report_lines.append(f"Pairwise Comparisons ({test_type}):")
    report_lines.append("-" * 70)
    report_lines.append(f"Significance level: α = {significance_level}")
    report_lines.append("")

    for _, row in comparison_df.iterrows():
        method_a = row["Method A"]
        method_b = row["Method B"]
        mean_diff = row["Mean Difference"]
        p_value = row["P-value"]
        is_sig = row["Significant"]
        cohens_d = row["Cohen's d"]
        effect_interp = effect_size_interpretation(cohens_d)

        sig_marker = "***" if is_sig else "n.s."

        report_lines.append(f"{method_a.upper()} vs {method_b.upper()}:")
        report_lines.append(f"  Mean difference: {mean_diff:+.4f}")
        report_lines.append(f"  p-value: {p_value:.4f} {sig_marker}")
        report_lines.append(
            f"  Effect size (Cohen's d): {cohens_d:.4f} ({effect_interp})"
        )
        report_lines.append("")

    # Summary
    report_lines.append("=" * 70)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 70)

    significant_comparisons = comparison_df[comparison_df["Significant"]]
    if len(significant_comparisons) > 0:
        report_lines.append("Statistically significant differences found:")
        for _, row in significant_comparisons.iterrows():
            better_method = (
                row["Method A"] if row["Mean Difference"] < 0 else row["Method B"]
            )
            report_lines.append(
                f"  - {better_method.upper()} shows significant improvement (p={row['P-value']:.4f})"
            )
    else:
        report_lines.append(
            "No statistically significant differences found between methods."
        )

    report_lines.append("")
    report_lines.append("*** p < 0.05 (statistically significant)")
    report_lines.append("n.s. = not significant")

    report = "\n".join(report_lines)

    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Statistical report saved to: {output_file}")

    return report


def bootstrap_confidence_interval(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for the mean.

    Parameters
    ----------
    data : np.ndarray
        Data to bootstrap
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    random_seed : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    mean : float
        Sample mean
    lower_bound : float
        Lower confidence bound
    upper_bound : float
        Upper confidence bound

    Examples
    --------
    >>> data = np.array([2.5, 3.0, 2.8, 3.2, 2.9])
    >>> mean, lower, upper = bootstrap_confidence_interval(data, n_bootstrap=1000)
    >>> print(f"Mean: {mean:.2f}, 95% CI: [{lower:.2f}, {upper:.2f}]")
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(data)
    bootstrap_means = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)

    mean = np.mean(data)
    alpha = 1 - confidence_level
    lower_bound = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper_bound = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return mean, lower_bound, upper_bound


def revised_wilcoxon_signed_rank(differences: np.ndarray) -> Dict[str, object]:
    """Frozen paired Wilcoxon configuration for the revised experiment."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("differences must be a non-empty one-dimensional array")
    all_zero = bool(np.all(differences == 0.0))
    if all_zero:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "all_zero": True,
            "n": int(differences.size),
        }
    result = stats.wilcoxon(
        differences,
        zero_method="pratt",
        alternative="two-sided",
        method="approx",
        correction=False,
    )
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "all_zero": False,
        "n": int(differences.size),
    }


def holm_adjust(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, dict]:
    """Apply Holm's step-down familywise correction to named p-values."""
    if not p_values:
        raise ValueError("p_values must not be empty")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    running_max = 0.0
    adjusted = {}
    still_rejecting = True
    for rank, (name, raw_p) in enumerate(ordered):
        if not 0.0 <= raw_p <= 1.0:
            raise ValueError("p-values must be between zero and one")
        multiplier = family_size - rank
        running_max = max(running_max, multiplier * raw_p)
        threshold = alpha / multiplier
        reject = still_rejecting and raw_p <= threshold
        if not reject:
            still_rejecting = False
        adjusted[name] = {
            "raw_p": float(raw_p),
            "adjusted_p": float(min(1.0, running_max)),
            "reject": bool(reject),
            "holm_rank": rank + 1,
            "threshold": float(threshold),
        }
    return {name: adjusted[name] for name in p_values}


def percentile_bootstrap_summary(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
) -> Dict[str, object]:
    """Bootstrap the mean of independent replicate-level values."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence_level": float(confidence_level),
        "bootstrap_samples": int(n_bootstrap),
        "n": int(values.size),
    }


if __name__ == "__main__":
    # Example usage
    print("Statistical Testing Module for Recommendation Systems")
    print("\nExample: Comparing three methods")

    # Simulated RMSE scores for 20 test items
    np.random.seed(42)
    baseline = np.random.normal(2.5, 0.5, 20)
    f1 = np.random.normal(2.2, 0.5, 20)
    f2 = np.random.normal(2.3, 0.5, 20)

    results = {"baseline": baseline, "f1": f1, "f2": f2}

    report = generate_statistical_report(results, metric_name="RMSE")
    print(report)
