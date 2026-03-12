"""
Preference Vector for User Feedback

Implements user preference feedback mechanism that captures the difference
between user's adjusted ratings and system recommendations.

References:
    - Paper Equation (11): Preference vector calculation
"""

import numpy as np
from typing import Optional, Dict, List, Tuple


def calculate_preference_vector(
    user_adjusted_scores: np.ndarray, system_recommendations: np.ndarray
) -> np.ndarray:
    """
    Calculate user preference vector.

    Implements Equation (11) from the paper:
    L_A30i = S_A30i - R_A30i

    where:
    - S_A30i: User A's adjusted scores for 30 items
    - R_A30i: System's recommendation scores for those 30 items
    - L_A30i: Preference vector (difference)

    Positive values indicate the user likes the item MORE than the system predicted.
    Negative values indicate the user likes the item LESS than the system predicted.

    Parameters
    ----------
    user_adjusted_scores : np.ndarray, shape (n_items,)
        User's adjusted ratings for items (S_A30)
        Typically on a 0-100 point scale
    system_recommendations : np.ndarray, shape (n_items,)
        System's recommendation scores for items (R_A30)
        Typically on a 0-100 point scale

    Returns
    -------
    preference_vector : np.ndarray, shape (n_items,)
        Preference vector for the user (L_A30)

    Examples
    --------
    >>> # User increases score for item 0, decreases for items 1 and 2
    >>> user_scores = np.array([90., 50., 30., 60.])
    >>> system_scores = np.array([85., 60., 40., 60.])
    >>> pref_vec = calculate_preference_vector(user_scores, system_scores)
    >>> print(pref_vec)
    [ 5. -10. -10.   0.]

    Notes
    -----
    The preference vector captures user feedback in a numerical form that can
    be stored and reused for future recommendations.
    """
    # Equation (11): L_A30i = S_A30i - R_A30i
    if user_adjusted_scores.shape != system_recommendations.shape:
        raise ValueError(
            f"Shape mismatch: user_adjusted_scores {user_adjusted_scores.shape} "
            f"vs system_recommendations {system_recommendations.shape}"
        )

    preference_vector = user_adjusted_scores - system_recommendations

    return preference_vector


def apply_preference_vector(
    recommendation_scores: np.ndarray,
    preference_vector: np.ndarray,
    scale_factor: float = 1.0,
    clip_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Apply preference vector to recommendation scores.

    This function adjusts the recommendation scores based on the user's
    previously learned preferences.

    Parameters
    ----------
    recommendation_scores : np.ndarray, shape (n_items,)
        Original recommendation scores from CF model
    preference_vector : np.ndarray, shape (n_items,)
        User's preference vector (from calculate_preference_vector)
    scale_factor : float, default=1.0
        Scaling factor for preference vector influence (0-1 recommended)
        Use < 1.0 to make adjustments more conservative
    clip_range : Optional[Tuple[float, float]], default=None
        If provided, clip adjusted scores to [min, max] range
        e.g., (0.0, 100.0) for 100-point scale

    Returns
    -------
    adjusted_scores : np.ndarray, shape (n_items,)
        Recommendation scores adjusted by preference vector

    Examples
    --------
    >>> # Original CF recommendations
    >>> cf_scores = np.array([80., 60., 40., 20.])
    >>> # User's learned preferences
    >>> pref_vec = np.array([5., -10., 10., 0.])
    >>> # Apply preferences with full strength
    >>> adjusted = apply_preference_vector(cf_scores, pref_vec, scale_factor=1.0)
    >>> print(adjusted)
    [85. 50. 50. 20.]
    >>> # Apply preferences more conservatively
    >>> adjusted = apply_preference_vector(cf_scores, pref_vec, scale_factor=0.5)
    >>> print(adjusted)
    [82.5 55.  45.  20. ]
    """
    if recommendation_scores.shape != preference_vector.shape:
        raise ValueError(
            f"Shape mismatch: recommendation_scores {recommendation_scores.shape} "
            f"vs preference_vector {preference_vector.shape}"
        )

    # Apply preference vector with scaling
    adjusted_scores = recommendation_scores + (scale_factor * preference_vector)

    # Clip to valid range if specified
    if clip_range is not None:
        min_val, max_val = clip_range
        adjusted_scores = np.clip(adjusted_scores, min_val, max_val)

    return adjusted_scores


def extend_preference_vector(
    preference_vector: np.ndarray,
    n_total_items: int,
    rated_item_indices: List[int],
    fill_method: str = "zero",
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Extend a partial preference vector to cover all items.

    In the paper, User A provides feedback on 30 items but the system
    has 50 items total. This function extends the 30-item preference
    vector to 50 items.

    Parameters
    ----------
    preference_vector : np.ndarray, shape (n_rated,)
        Preference vector for rated items only
    n_total_items : int
        Total number of items in the system
    rated_item_indices : List[int]
        Indices of items that were rated
    fill_method : {'zero', 'mean', 'value'}, default='zero'
        Method to fill unrated items:
        - 'zero': Fill with 0 (neutral preference)
        - 'mean': Fill with mean of rated preferences
        - 'value': Fill with specific value (fill_value parameter)
    fill_value : float, default=0.0
        Value to use when fill_method='value'

    Returns
    -------
    extended_vector : np.ndarray, shape (n_total_items,)
        Preference vector extended to all items

    Examples
    --------
    >>> # Preference vector for 3 rated items
    >>> pref_vec = np.array([5., -10., 10.])
    >>> rated_indices = [0, 5, 10]
    >>> # Extend to 20 total items with zero fill
    >>> extended = extend_preference_vector(pref_vec, 20, rated_indices, 'zero')
    >>> print(extended.shape)
    (20,)
    >>> print(extended[[0, 5, 10]])
    [ 5. -10.  10.]
    """
    if len(preference_vector) != len(rated_item_indices):
        raise ValueError(
            f"Length mismatch: preference_vector has {len(preference_vector)} "
            f"elements but rated_item_indices has {len(rated_item_indices)}"
        )

    # Initialize extended vector
    extended_vector = np.zeros(n_total_items, dtype=np.float64)

    # Determine fill value
    if fill_method == "zero":
        fill = 0.0
    elif fill_method == "mean":
        fill = np.mean(preference_vector)
    elif fill_method == "value":
        fill = fill_value
    else:
        raise ValueError(f"Unknown fill_method: {fill_method}")

    # Fill with default value
    extended_vector.fill(fill)

    # Place preference values at rated positions
    for idx, pref in zip(rated_item_indices, preference_vector):
        if idx >= n_total_items:
            raise ValueError(
                f"Item index {idx} is out of bounds for {n_total_items} items"
            )
        extended_vector[idx] = pref

    return extended_vector


def merge_preference_vectors(
    preference_vectors: List[np.ndarray],
    method: str = "mean",
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Merge multiple preference vectors into one.

    Useful when combining feedback from multiple interactions or sessions.

    Parameters
    ----------
    preference_vectors : List[np.ndarray]
        List of preference vectors to merge
    method : {'mean', 'weighted_mean', 'max', 'min'}, default='mean'
        Method to merge vectors:
        - 'mean': Simple average
        - 'weighted_mean': Weighted average (requires weights parameter)
        - 'max': Take maximum preference for each item
        - 'min': Take minimum preference for each item
    weights : Optional[List[float]], default=None
        Weights for weighted_mean method

    Returns
    -------
    merged_vector : np.ndarray
        Merged preference vector

    Examples
    --------
    >>> # Two feedback sessions
    >>> pref1 = np.array([5., -10., 10., 0.])
    >>> pref2 = np.array([10., -5., 5., 5.])
    >>> merged = merge_preference_vectors([pref1, pref2], method='mean')
    >>> print(merged)
    [ 7.5 -7.5  7.5  2.5]
    """
    if len(preference_vectors) == 0:
        raise ValueError("preference_vectors list cannot be empty")

    # Stack vectors
    pref_matrix = np.vstack(preference_vectors)

    if method == "mean":
        return np.mean(pref_matrix, axis=0)
    elif method == "weighted_mean":
        if weights is None:
            raise ValueError("weights must be provided for weighted_mean method")
        if len(weights) != len(preference_vectors):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of vectors ({len(preference_vectors)})"
            )
        weights_array = np.array(weights)
        weights_array = weights_array / weights_array.sum()  # Normalize
        return np.dot(weights_array, pref_matrix)
    elif method == "max":
        return np.max(pref_matrix, axis=0)
    elif method == "min":
        return np.min(pref_matrix, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")


class PreferenceVectorManager:
    """
    Manager for storing and retrieving user preference vectors.

    Maintains a database of preference vectors for multiple users,
    allowing for persistent user feedback across sessions.

    Attributes
    ----------
    preference_db : Dict[str, np.ndarray]
        Dictionary mapping user IDs to preference vectors

    Examples
    --------
    >>> manager = PreferenceVectorManager()
    >>> # Store preference vector for User A
    >>> pref_vec = np.array([5., -10., 10., 0.])
    >>> manager.store('user_a', pref_vec)
    >>> # Retrieve later
    >>> retrieved = manager.retrieve('user_a')
    >>> print(retrieved)
    [ 5. -10.  10.   0.]
    """

    def __init__(self):
        self.preference_db: Dict[str, np.ndarray] = {}

    def store(self, user_id: str, preference_vector: np.ndarray) -> None:
        """
        Store a preference vector for a user.

        Parameters
        ----------
        user_id : str
            User identifier
        preference_vector : np.ndarray
            Preference vector to store
        """
        self.preference_db[user_id] = preference_vector.copy()

    def retrieve(self, user_id: str) -> Optional[np.ndarray]:
        """
        Retrieve a preference vector for a user.

        Parameters
        ----------
        user_id : str
            User identifier

        Returns
        -------
        preference_vector : Optional[np.ndarray]
            Preference vector if found, None otherwise
        """
        return self.preference_db.get(user_id)

    def update(
        self,
        user_id: str,
        new_preference_vector: np.ndarray,
        merge_method: str = "mean",
    ) -> None:
        """
        Update an existing preference vector by merging with new feedback.

        Parameters
        ----------
        user_id : str
            User identifier
        new_preference_vector : np.ndarray
            New preference vector to merge
        merge_method : str, default='mean'
            Method to merge vectors (see merge_preference_vectors)
        """
        if user_id in self.preference_db:
            old_vector = self.preference_db[user_id]
            merged = merge_preference_vectors(
                [old_vector, new_preference_vector], method=merge_method
            )
            self.preference_db[user_id] = merged
        else:
            self.store(user_id, new_preference_vector)

    def clear(self, user_id: Optional[str] = None) -> None:
        """
        Clear preference vectors.

        Parameters
        ----------
        user_id : Optional[str], default=None
            If provided, clear only this user's vector.
            If None, clear all vectors.
        """
        if user_id is None:
            self.preference_db.clear()
        else:
            self.preference_db.pop(user_id, None)

    def list_users(self) -> List[str]:
        """
        Get list of all user IDs with stored preferences.

        Returns
        -------
        user_ids : List[str]
            List of user IDs
        """
        return list(self.preference_db.keys())
