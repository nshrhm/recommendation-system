"""
Ranking and Scoring Functions

Implements ranking-based scoring mechanisms for recommendation systems.

References:
    - Paper Equation (10): Point calculation from ranking
"""

import numpy as np
from typing import Dict, Optional, List, Tuple


def scores_to_ranking(scores: np.ndarray, descending: bool = True) -> np.ndarray:
    """
    Convert recommendation scores to ranking positions.

    Parameters
    ----------
    scores : np.ndarray, shape (n_items,)
        Recommendation scores for items
    descending : bool, default=True
        If True, higher scores get better (lower) ranks

    Returns
    -------
    rankings : np.ndarray, shape (n_items,)
        Ranking positions (0-indexed: 0 is best, n-1 is worst)

    Examples
    --------
    >>> scores = np.array([0.5, 0.9, 0.3, 0.7])
    >>> rankings = scores_to_ranking(scores)
    >>> print(rankings)
    [2 0 3 1]  # Item 1 (0.9) is rank 0, Item 3 (0.7) is rank 1, etc.
    """
    if descending:
        # Higher scores -> better (lower) ranks
        rankings = np.argsort(np.argsort(-scores))
    else:
        # Lower scores -> better (lower) ranks
        rankings = np.argsort(np.argsort(scores))

    return rankings


def ranking_to_points(rankings: np.ndarray, max_points: float = 100.0) -> np.ndarray:
    """
    Convert ranking positions to points.

    Implements Equation (10) from the paper:
    P_i = ((n - r_i - 1) / n) × 100

    where:
    - n is the number of items
    - r_i is the ranking position of item i (0-indexed)
    - P_i is the point value for item i

    Best item (rank 0) gets highest points.
    Worst item (rank n-1) gets 0 points.

    Parameters
    ----------
    rankings : np.ndarray, shape (n_items,)
        Ranking positions (0-indexed: 0 is best)
    max_points : float, default=100.0
        Maximum points to assign (for best-ranked item)

    Returns
    -------
    points : np.ndarray, shape (n_items,)
        Point values for each item

    Examples
    --------
    >>> rankings = np.array([2, 0, 3, 1])  # Item 1 is best, Item 3 is worst
    >>> points = ranking_to_points(rankings, max_points=100.0)
    >>> print(points)
    [25. 75. 0. 50.]  # Item 1 gets 75 points, Item 3 gets 0 points

    Notes
    -----
    The formula ensures:
    - Best item (rank 0): P = ((n-0-1)/n) × 100 = ((n-1)/n) × 100
    - Worst item (rank n-1): P = ((n-(n-1)-1)/n) × 100 = 0
    """
    n = len(rankings)

    # Equation (10): P_i = ((n - r_i - 1) / n) × max_points
    points = ((n - rankings - 1) / n) * max_points

    return points


def scores_to_points(
    scores: np.ndarray, max_points: float = 100.0, descending: bool = True
) -> np.ndarray:
    """
    Convert scores directly to points (convenience function).

    Combines scores_to_ranking() and ranking_to_points().

    Parameters
    ----------
    scores : np.ndarray, shape (n_items,)
        Recommendation scores
    max_points : float, default=100.0
        Maximum points to assign
    descending : bool, default=True
        If True, higher scores get more points

    Returns
    -------
    points : np.ndarray, shape (n_items,)
        Point values for each item

    Examples
    --------
    >>> scores = np.array([0.5, 0.9, 0.3, 0.7])
    >>> points = scores_to_points(scores)
    >>> print(points)
    [50. 75. 0. 25.]
    """
    rankings = scores_to_ranking(scores, descending=descending)
    points = ranking_to_points(rankings, max_points=max_points)

    return points


def create_ranking_vector(
    item_order: List[int], n_items: int, max_points: float = 100.0
) -> np.ndarray:
    """
    Create a ranking vector from user-provided item ordering.

    Used for impression ranking (F1 method) where users manually
    reorder recommended items.

    Parameters
    ----------
    item_order : List[int]
        List of item indices in user's preferred order
        (first item is most preferred)
    n_items : int
        Total number of items in the system
    max_points : float, default=100.0
        Maximum points to assign

    Returns
    -------
    ranking_vector : np.ndarray, shape (n_items,)
        Point values for all items (0 for unranked items)

    Examples
    --------
    >>> # User ranks items in order: 5, 2, 8
    >>> ranking_vec = create_ranking_vector([5, 2, 8], n_items=10)
    >>> print(ranking_vec[5])  # Item 5 gets highest points
    100.0
    >>> print(ranking_vec[8])  # Item 8 gets lowest points among ranked
    0.0
    """
    ranking_vector = np.zeros(n_items)

    # Create rankings for ordered items only
    n_ranked = len(item_order)
    rankings = np.arange(n_ranked)
    points = ranking_to_points(rankings, max_points=max_points)

    # Assign points to corresponding items
    for item_idx, point_value in zip(item_order, points):
        ranking_vector[item_idx] = point_value

    return ranking_vector


def normalize_points(
    points: np.ndarray, min_val: float = 0.0, max_val: float = 100.0
) -> np.ndarray:
    """
    Normalize points to a specified range.

    Parameters
    ----------
    points : np.ndarray
        Point values to normalize
    min_val : float, default=0.0
        Minimum value after normalization
    max_val : float, default=100.0
        Maximum value after normalization

    Returns
    -------
    normalized : np.ndarray
        Normalized point values

    Examples
    --------
    >>> points = np.array([25., 50., 75., 100.])
    >>> normalized = normalize_points(points, min_val=0, max_val=10)
    >>> print(normalized)
    [2.5 5.  7.5 10.]
    """
    current_min = points.min()
    current_max = points.max()

    # Avoid division by zero
    if current_max == current_min:
        return np.full_like(points, (min_val + max_val) / 2)

    # Linear normalization
    normalized = (points - current_min) / (current_max - current_min)
    normalized = normalized * (max_val - min_val) + min_val

    return normalized


def get_top_k_items(
    points: np.ndarray, k: int, exclude_indices: Optional[List[int]] = None
) -> List[Tuple[int, float]]:
    """
    Get top-K items based on point values.

    Parameters
    ----------
    points : np.ndarray, shape (n_items,)
        Point values for items
    k : int
        Number of top items to return
    exclude_indices : List[int], optional
        Item indices to exclude from results

    Returns
    -------
    top_k : List[Tuple[int, float]]
        List of (item_index, point_value) tuples, sorted by points descending

    Examples
    --------
    >>> points = np.array([25., 75., 0., 50.])
    >>> top_k = get_top_k_items(points, k=2)
    >>> print(top_k)
    [(1, 75.0), (3, 50.0)]
    """
    if exclude_indices is None:
        exclude_indices = []

    # Create mask for items to consider
    mask = np.ones(len(points), dtype=bool)
    mask[exclude_indices] = False

    # Get valid items
    valid_points = points[mask]
    valid_indices = np.where(mask)[0]

    # Sort by points descending
    top_k_idx = np.argsort(valid_points)[::-1][:k]

    top_k = [(int(valid_indices[i]), float(valid_points[i])) for i in top_k_idx]

    return top_k


def deterministic_score_order(item_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Order item IDs by descending score and then ascending item ID."""
    item_ids = np.asarray(item_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if item_ids.ndim != 1 or scores.ndim != 1 or item_ids.shape != scores.shape:
        raise ValueError("item_ids and scores must be one-dimensional and aligned")
    if len(np.unique(item_ids)) != len(item_ids):
        raise ValueError("item_ids must be unique")
    order = np.lexsort((item_ids, -scores))
    return item_ids[order]


def interactive_impression_rerank(
    initial_order: np.ndarray,
    displayed_top_n: int,
    feedback_by_item: Dict[int, float],
) -> Dict[str, np.ndarray]:
    """Post-feedback reranking of only the displayed prefix.

    Equal feedback values preserve the previous Hybrid-CF order. The
    undisplayed suffix is returned byte-for-byte in its original order.
    """
    initial_order = np.asarray(initial_order, dtype=np.int64)
    if initial_order.ndim != 1 or len(np.unique(initial_order)) != len(initial_order):
        raise ValueError("initial_order must contain unique one-dimensional item IDs")
    if not 0 < displayed_top_n <= len(initial_order):
        raise ValueError("displayed_top_n must be within the ranking length")

    displayed = initial_order[:displayed_top_n]
    if set(feedback_by_item) != set(int(item_id) for item_id in displayed):
        raise ValueError("feedback must be supplied for exactly the displayed items")
    feedback = np.asarray(
        [feedback_by_item[int(item_id)] for item_id in displayed], dtype=np.float64
    )
    previous_positions = np.arange(displayed_top_n)
    reorder = np.lexsort((previous_positions, -feedback))
    reranked_displayed = displayed[reorder]
    final_order = np.concatenate([reranked_displayed, initial_order[displayed_top_n:]])

    return {
        "initial_order": initial_order.copy(),
        "displayed_ids": displayed.copy(),
        "displayed_feedback": feedback.copy(),
        "final_order": final_order,
        "changed_positions": np.asarray(
            np.count_nonzero(final_order != initial_order), dtype=np.int64
        ),
    }
