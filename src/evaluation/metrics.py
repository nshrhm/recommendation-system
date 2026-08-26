"""
Evaluation Metrics for Recommendation Systems

Implements standard metrics for evaluating recommendation quality:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- NDCG@K (Normalized Discounted Cumulative Gain)
- Precision@K
- Recall@K
- F1@K
"""

import numpy as np
from typing import List, Optional, Tuple
import warnings


def rmse(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.

    RMSE measures the average magnitude of prediction errors.
    Lower values indicate better performance.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted ratings
    ground_truth : np.ndarray
        True ratings

    Returns
    -------
    rmse_value : float
        Root Mean Square Error

    Examples
    --------
    >>> predictions = np.array([8.5, 7.0, 6.5])
    >>> ground_truth = np.array([8.0, 7.5, 6.0])
    >>> rmse_value = rmse(predictions, ground_truth)
    >>> print(f"RMSE: {rmse_value:.2f}")
    RMSE: 0.48
    """
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} "
            f"vs ground_truth {ground_truth.shape}"
        )

    mse = np.mean((predictions - ground_truth) ** 2)
    return np.sqrt(mse)


def mae(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    MAE measures the average absolute difference between predictions
    and ground truth.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted ratings
    ground_truth : np.ndarray
        True ratings

    Returns
    -------
    mae_value : float
        Mean Absolute Error

    Examples
    --------
    >>> predictions = np.array([8.5, 7.0, 6.5])
    >>> ground_truth = np.array([8.0, 7.5, 6.0])
    >>> mae_value = mae(predictions, ground_truth)
    >>> print(f"MAE: {mae_value:.2f}")
    MAE: 0.42
    """
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} "
            f"vs ground_truth {ground_truth.shape}"
        )

    return np.mean(np.abs(predictions - ground_truth))


def dcg_at_k(scores: np.ndarray, k: Optional[int] = None) -> float:
    """
    Calculate Discounted Cumulative Gain at K.

    DCG@K measures the quality of a ranking by giving higher weights
    to highly relevant items appearing earlier in the list.

    Formula: DCG@K = sum(rel_i / log2(i + 1)) for i in [0, k-1]

    Parameters
    ----------
    scores : np.ndarray
        Relevance scores (higher is better), sorted by ranking
    k : Optional[int], default=None
        Position cutoff (if None, use all scores)

    Returns
    -------
    dcg : float
        Discounted Cumulative Gain

    Examples
    --------
    >>> # Perfect ranking: [10, 8, 5, 3, 1]
    >>> scores = np.array([10., 8., 5., 3., 1.])
    >>> dcg = dcg_at_k(scores, k=5)
    >>> print(f"DCG@5: {dcg:.2f}")
    DCG@5: 21.13
    """
    if k is None:
        k = len(scores)
    else:
        k = min(k, len(scores))

    if k == 0:
        return 0.0

    # Calculate discounts: 1/log2(i+2) for i in [0, k-1]
    # Note: i+2 because ranking positions start at 1, and log2(1) = 0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    # DCG = sum(scores * discounts)
    dcg = np.sum(scores[:k] * discounts)

    return dcg


def ndcg_at_k(
    predictions: np.ndarray, ground_truth: np.ndarray, k: Optional[int] = None
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    NDCG@K normalizes DCG by the ideal DCG (IDCG), allowing comparison
    across different query sets. Value ranges from 0 to 1.

    Formula: NDCG@K = DCG@K / IDCG@K

    Parameters
    ----------
    predictions : np.ndarray
        Predicted relevance scores (will be sorted)
    ground_truth : np.ndarray
        True relevance scores
    k : Optional[int], default=None
        Position cutoff (if None, use all items)

    Returns
    -------
    ndcg : float
        Normalized Discounted Cumulative Gain (0-1, higher is better)

    Examples
    --------
    >>> predictions = np.array([0.9, 0.8, 0.5, 0.3, 0.1])
    >>> ground_truth = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    >>> ndcg = ndcg_at_k(predictions, ground_truth, k=5)
    >>> print(f"NDCG@5: {ndcg:.3f}")
    NDCG@5: 0.987
    """
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} "
            f"vs ground_truth {ground_truth.shape}"
        )

    # Sort ground truth by predictions (ranking produced by system)
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_ground_truth = ground_truth[sorted_indices]

    # Calculate DCG
    dcg = dcg_at_k(sorted_ground_truth, k=k)

    # Calculate ideal DCG (sort ground truth optimally)
    ideal_sorted_ground_truth = np.sort(ground_truth)[::-1]
    idcg = dcg_at_k(ideal_sorted_ground_truth, k=k)

    # Avoid division by zero
    if idcg == 0:
        return 0.0

    ndcg = dcg / idcg

    return ndcg


def ndcg_from_order(
    item_order: np.ndarray,
    relevance_by_item: np.ndarray,
    candidate_item_ids: np.ndarray,
    k: Optional[int] = None,
) -> float:
    """Calculate linear-gain NDCG from an explicit item ordering."""
    item_order = np.asarray(item_order, dtype=np.int64)
    relevance_by_item = np.asarray(relevance_by_item, dtype=np.float64)
    candidates = np.asarray(candidate_item_ids, dtype=np.int64)
    if item_order.ndim != 1 or candidates.ndim != 1:
        raise ValueError("item_order and candidate_item_ids must be one-dimensional")
    if len(item_order) != len(candidates) or set(item_order) != set(candidates):
        raise ValueError("item_order must be a permutation of candidate_item_ids")
    if np.any(item_order < 0) or np.any(item_order >= len(relevance_by_item)):
        raise ValueError("item_order contains an invalid item ID")

    ordered_relevance = relevance_by_item[item_order]
    candidate_relevance = relevance_by_item[candidates]
    dcg = dcg_at_k(ordered_relevance, k)
    idcg = dcg_at_k(np.sort(candidate_relevance)[::-1], k)
    return 0.0 if idcg == 0 else float(dcg / idcg)


def evaluate_ranking_order(
    item_order: np.ndarray,
    relevance_by_item: np.ndarray,
    candidate_item_ids: np.ndarray,
    k_values: List[int] = [5, 10, 20],
) -> dict:
    """Evaluate only graded ranking metrics for an explicit ranking."""
    return {
        f"ndcg@{k}": ndcg_from_order(
            item_order, relevance_by_item, candidate_item_ids, k=k
        )
        for k in k_values
    }


def evaluate_native_score_method(
    item_ids: np.ndarray,
    predictions: np.ndarray,
    relevance_by_item: np.ndarray,
    k_values: List[int] = [5, 10, 20],
) -> dict:
    """Evaluate native 0-10 predictions without binary ranking metrics."""
    item_ids = np.asarray(item_ids, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if item_ids.shape != predictions.shape:
        raise ValueError("item_ids and predictions must have matching shapes")
    truth = np.asarray(relevance_by_item, dtype=np.float64)[item_ids]
    order_positions = np.lexsort((item_ids, -predictions))
    item_order = item_ids[order_positions]
    metrics = {
        "rmse": float(rmse(predictions, truth)),
        "mae": float(mae(predictions, truth)),
    }
    metrics.update(
        evaluate_ranking_order(item_order, relevance_by_item, item_ids, k_values)
    )
    return metrics


def precision_at_k(
    predictions: np.ndarray, ground_truth: np.ndarray, k: int, threshold: float = 0.5
) -> float:
    """
    Calculate Precision at K.

    Precision@K measures the proportion of relevant items
    in the top-K recommendations.

    Formula: Precision@K = (# relevant items in top-K) / K

    Parameters
    ----------
    predictions : np.ndarray
        Predicted scores
    ground_truth : np.ndarray
        True relevance (1 = relevant, 0 = not relevant)
        or continuous scores (will be binarized using threshold)
    k : int
        Number of top items to consider
    threshold : float, default=0.5
        Threshold for binarizing ground truth (if continuous)

    Returns
    -------
    precision : float
        Precision at K (0-1, higher is better)

    Examples
    --------
    >>> predictions = np.array([0.9, 0.8, 0.5, 0.3, 0.1])
    >>> ground_truth = np.array([1, 1, 0, 1, 0])  # Binary relevance
    >>> prec = precision_at_k(predictions, ground_truth, k=3)
    >>> print(f"Precision@3: {prec:.2f}")
    Precision@3: 0.67  # 2 out of 3 are relevant
    """
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} "
            f"vs ground_truth {ground_truth.shape}"
        )

    k = min(k, len(predictions))

    # Get top-K items
    top_k_indices = np.argsort(predictions)[::-1][:k]

    # Binarize ground truth if needed
    if ground_truth.dtype != bool and not np.all(np.isin(ground_truth, [0, 1])):
        ground_truth_binary = ground_truth >= threshold
    else:
        ground_truth_binary = ground_truth.astype(bool)

    # Count relevant items in top-K
    n_relevant = np.sum(ground_truth_binary[top_k_indices])

    precision = n_relevant / k

    return precision


def recall_at_k(
    predictions: np.ndarray, ground_truth: np.ndarray, k: int, threshold: float = 0.5
) -> float:
    """
    Calculate Recall at K.

    Recall@K measures the proportion of all relevant items
    that appear in the top-K recommendations.

    Formula: Recall@K = (# relevant items in top-K) / (# total relevant items)

    Parameters
    ----------
    predictions : np.ndarray
        Predicted scores
    ground_truth : np.ndarray
        True relevance (1 = relevant, 0 = not relevant)
        or continuous scores (will be binarized using threshold)
    k : int
        Number of top items to consider
    threshold : float, default=0.5
        Threshold for binarizing ground truth (if continuous)

    Returns
    -------
    recall : float
        Recall at K (0-1, higher is better)

    Examples
    --------
    >>> predictions = np.array([0.9, 0.8, 0.5, 0.3, 0.1])
    >>> ground_truth = np.array([1, 1, 0, 1, 0])  # 3 relevant items total
    >>> rec = recall_at_k(predictions, ground_truth, k=3)
    >>> print(f"Recall@3: {rec:.2f}")
    Recall@3: 0.67  # 2 out of 3 relevant items found
    """
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} "
            f"vs ground_truth {ground_truth.shape}"
        )

    k = min(k, len(predictions))

    # Get top-K items
    top_k_indices = np.argsort(predictions)[::-1][:k]

    # Binarize ground truth if needed
    if ground_truth.dtype != bool and not np.all(np.isin(ground_truth, [0, 1])):
        ground_truth_binary = ground_truth >= threshold
    else:
        ground_truth_binary = ground_truth.astype(bool)

    # Count relevant items in top-K
    n_relevant_in_top_k = np.sum(ground_truth_binary[top_k_indices])

    # Count total relevant items
    n_total_relevant = np.sum(ground_truth_binary)

    if n_total_relevant == 0:
        warnings.warn("No relevant items in ground truth", UserWarning)
        return 0.0

    recall = n_relevant_in_top_k / n_total_relevant

    return recall


def f1_at_k(
    predictions: np.ndarray, ground_truth: np.ndarray, k: int, threshold: float = 0.5
) -> float:
    """
    Calculate F1 Score at K.

    F1@K is the harmonic mean of Precision@K and Recall@K.

    Formula: F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)

    Parameters
    ----------
    predictions : np.ndarray
        Predicted scores
    ground_truth : np.ndarray
        True relevance
    k : int
        Number of top items to consider
    threshold : float, default=0.5
        Threshold for binarizing ground truth

    Returns
    -------
    f1 : float
        F1 Score at K (0-1, higher is better)

    Examples
    --------
    >>> predictions = np.array([0.9, 0.8, 0.5, 0.3, 0.1])
    >>> ground_truth = np.array([1, 1, 0, 1, 0])
    >>> f1 = f1_at_k(predictions, ground_truth, k=3)
    >>> print(f"F1@3: {f1:.2f}")
    F1@3: 0.67
    """
    prec = precision_at_k(predictions, ground_truth, k, threshold)
    rec = recall_at_k(predictions, ground_truth, k, threshold)

    if prec + rec == 0:
        return 0.0

    f1 = 2 * (prec * rec) / (prec + rec)

    return f1


def evaluate_all_metrics(
    predictions: np.ndarray, ground_truth: np.ndarray, k_values: List[int] = [5, 10, 20]
) -> dict:
    """
    Evaluate all metrics at once.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted scores
    ground_truth : np.ndarray
        True ratings/relevance
    k_values : List[int], default=[5, 10, 20]
        List of K values for ranking metrics

    Returns
    -------
    metrics : dict
        Dictionary containing all metric values

    Examples
    --------
    >>> predictions = np.random.rand(50)
    >>> ground_truth = np.random.rand(50)
    >>> metrics = evaluate_all_metrics(predictions, ground_truth, k_values=[5, 10])
    >>> print(metrics.keys())
    dict_keys(['rmse', 'mae', 'ndcg@5', 'ndcg@10', 'precision@5', ...])
    """
    metrics = {}

    # Error metrics
    metrics["rmse"] = rmse(predictions, ground_truth)
    metrics["mae"] = mae(predictions, ground_truth)

    # Ranking metrics
    for k in k_values:
        metrics[f"ndcg@{k}"] = ndcg_at_k(predictions, ground_truth, k)
        metrics[f"precision@{k}"] = precision_at_k(predictions, ground_truth, k)
        metrics[f"recall@{k}"] = recall_at_k(predictions, ground_truth, k)
        metrics[f"f1@{k}"] = f1_at_k(predictions, ground_truth, k)

    return metrics


def compare_methods(
    method_predictions: dict,
    ground_truth: np.ndarray,
    k_values: List[int] = [5, 10, 20],
) -> dict:
    """
    Compare multiple recommendation methods.

    Parameters
    ----------
    method_predictions : dict
        Dictionary mapping method names to prediction arrays
    ground_truth : np.ndarray
        True ratings/relevance
    k_values : List[int], default=[5, 10, 20]
        List of K values for ranking metrics

    Returns
    -------
    comparison : dict
        Nested dictionary: {method_name: {metric_name: value}}

    Examples
    --------
    >>> baseline = np.array([0.5, 0.6, 0.4, 0.8])
    >>> proposed = np.array([0.6, 0.7, 0.5, 0.9])
    >>> ground_truth = np.array([0.55, 0.65, 0.45, 0.85])
    >>> methods = {'Baseline': baseline, 'Proposed': proposed}
    >>> comparison = compare_methods(methods, ground_truth, k_values=[2])
    >>> print(comparison['Proposed']['rmse'])
    0.05
    """
    comparison = {}

    for method_name, predictions in method_predictions.items():
        comparison[method_name] = evaluate_all_metrics(
            predictions, ground_truth, k_values
        )

    return comparison
