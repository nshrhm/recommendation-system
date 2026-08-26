"""
Collaborative Filtering Recommendation Models

Implements:
- User-based Collaborative Filtering
- Item-based Collaborative Filtering

References:
    - Linden, G., Smith, B., & York, J. (2003). Amazon.com recommendations.
    - Sarwar, B., et al. (2001). Item-based collaborative filtering.
"""

import numpy as np
from typing import Literal, Optional, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances


def _cosine_on_common_observations(
    left: np.ndarray,
    right: np.ndarray,
    common_mask: np.ndarray,
) -> float:
    """Cosine similarity using only explicitly common observations."""
    if not common_mask.any():
        return 0.0
    left_common = left[common_mask]
    right_common = right[common_mask]
    denominator = np.linalg.norm(left_common) * np.linalg.norm(right_common)
    if denominator == 0:
        return 0.0
    return float(np.dot(left_common, right_common) / denominator)


def compute_user_cosine_similarity(
    ratings: np.ndarray, observed_mask: np.ndarray
) -> np.ndarray:
    """Compute user similarities on each pair's common observed items."""
    if ratings.shape != observed_mask.shape:
        raise ValueError("ratings and observed_mask must have the same shape")
    n_users = ratings.shape[0]
    similarities = np.zeros((n_users, n_users), dtype=np.float64)
    for left in range(n_users):
        for right in range(left + 1, n_users):
            common = observed_mask[left] & observed_mask[right]
            value = _cosine_on_common_observations(
                ratings[left], ratings[right], common
            )
            similarities[left, right] = value
            similarities[right, left] = value
    return similarities


def compute_item_cosine_similarity(
    ratings: np.ndarray,
    observed_mask: np.ndarray,
    reference_users: List[int],
) -> np.ndarray:
    """Compute item similarities using only specified training-safe users."""
    if ratings.shape != observed_mask.shape:
        raise ValueError("ratings and observed_mask must have the same shape")
    if not reference_users:
        raise ValueError("reference_users must not be empty")
    refs = np.asarray(reference_users, dtype=np.int64)
    if np.any(refs < 0) or np.any(refs >= ratings.shape[0]):
        raise ValueError("reference_users contains an invalid user index")

    n_items = ratings.shape[1]
    similarities = np.zeros((n_items, n_items), dtype=np.float64)
    ref_ratings = ratings[refs]
    ref_mask = observed_mask[refs]
    for left in range(n_items):
        for right in range(left + 1, n_items):
            common = ref_mask[:, left] & ref_mask[:, right]
            value = _cosine_on_common_observations(
                ref_ratings[:, left], ref_ratings[:, right], common
            )
            similarities[left, right] = value
            similarities[right, left] = value
    return similarities


def _top_positive_indices(
    similarities: np.ndarray, candidate_mask: np.ndarray, k: int
) -> np.ndarray:
    """Select highest positive similarities, breaking ties by item/user ID."""
    candidates = np.where(candidate_mask & (similarities > 0))[0]
    if candidates.size == 0:
        return candidates
    order = np.lexsort((candidates, -similarities[candidates]))
    return candidates[order[:k]]


class UserBasedCF:
    """
    User-based Collaborative Filtering

    Recommends items based on similar users' preferences.

    Parameters
    ----------
    similarity : {'cosine', 'pearson', 'euclidean'}, default='cosine'
        Similarity metric to use
    k_neighbors : int, default=5
        Number of similar users to consider

    Attributes
    ----------
    user_item_matrix_ : np.ndarray, shape (n_users, n_items)
        User-item rating matrix
    user_similarity_ : np.ndarray, shape (n_users, n_users)
        User similarity matrix

    Examples
    --------
    >>> from src.models.collaborative_filtering import UserBasedCF
    >>> import numpy as np
    >>>
    >>> # Create sample user-item matrix
    >>> ratings = np.array([
    ...     [5, 3, 0, 1],
    ...     [4, 0, 0, 1],
    ...     [1, 1, 0, 5],
    ...     [1, 0, 0, 4],
    ... ])
    >>>
    >>> # Train model
    >>> model = UserBasedCF(similarity='cosine', k_neighbors=2)
    >>> model.fit(ratings)
    >>>
    >>> # Predict rating for user 0, item 2
    >>> score = model.predict(user_id=0, item_id=2)
    >>> print(f"Predicted score: {score:.2f}")
    """

    def __init__(
        self,
        similarity: Literal["cosine", "pearson", "euclidean"] = "cosine",
        k_neighbors: int = 5,
    ):
        self.similarity = similarity
        self.k_neighbors = k_neighbors
        self.user_item_matrix_: Optional[np.ndarray] = None
        self.user_similarity_: Optional[np.ndarray] = None
        self.observed_mask_: Optional[np.ndarray] = None

    def fit(
        self,
        user_item_matrix: np.ndarray,
        observed_mask: Optional[np.ndarray] = None,
    ) -> "UserBasedCF":
        """
        Fit the model by computing user similarity matrix.

        Parameters
        ----------
        user_item_matrix : np.ndarray, shape (n_users, n_items)
            User-item rating matrix

        Returns
        -------
        self : UserBasedCF
            Fitted model
        """
        self.user_item_matrix_ = user_item_matrix.copy()

        if observed_mask is None:
            self.observed_mask_ = user_item_matrix > 0
        else:
            if observed_mask.shape != user_item_matrix.shape:
                raise ValueError("observed_mask must match user_item_matrix shape")
            self.observed_mask_ = observed_mask.astype(bool, copy=True)

        # Compute user similarity matrix
        if self.similarity == "cosine":
            if observed_mask is None:
                self.user_similarity_ = cosine_similarity(user_item_matrix)
            else:
                self.user_similarity_ = compute_user_cosine_similarity(
                    user_item_matrix, self.observed_mask_
                )
        elif self.similarity == "pearson":
            # TODO: Implement Pearson correlation
            raise NotImplementedError("Pearson correlation not yet implemented")
        elif self.similarity == "euclidean":
            self.user_similarity_ = 1 / (
                1 + pairwise_distances(user_item_matrix, metric="euclidean")
            )
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity}")

        # Set diagonal to 0 (user is not similar to themselves)
        np.fill_diagonal(self.user_similarity_, 0)

        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a specific user-item pair.

        Parameters
        ----------
        user_id : int
            User index
        item_id : int
            Item index

        Returns
        -------
        score : float
            Predicted rating score
        """
        if (
            self.user_item_matrix_ is None
            or self.user_similarity_ is None
            or self.observed_mask_ is None
        ):
            raise ValueError("Model not fitted. Call fit() first.")

        # Find k most similar users who rated this item
        user_sims = self.user_similarity_[user_id]
        rated_mask = self.observed_mask_[:, item_id]

        # Get similar users who rated the item
        similar_users_mask = rated_mask & (user_sims > 0)

        if not similar_users_mask.any():
            # No similar users rated this item, return mean rating
            user_ratings = self.user_item_matrix_[user_id]
            observed = self.observed_mask_[user_id]
            return float(user_ratings[observed].mean()) if observed.any() else 0.0

        # Get top-k similar users
        similar_users_idx = _top_positive_indices(
            user_sims, similar_users_mask, self.k_neighbors
        )

        # Weighted average of ratings
        sims = user_sims[similar_users_idx]
        ratings = self.user_item_matrix_[similar_users_idx, item_id]

        # Avoid division by zero
        if sims.sum() == 0:
            return ratings.mean()

        return np.dot(sims, ratings) / sims.sum()

    def predict_for_user(self, user_id: int) -> np.ndarray:
        """
        Predict ratings for all items for a given user.

        Parameters
        ----------
        user_id : int
            User index

        Returns
        -------
        scores : np.ndarray, shape (n_items,)
            Predicted scores for all items
        """
        if self.user_item_matrix_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        n_items = self.user_item_matrix_.shape[1]
        scores = np.zeros(n_items)

        for item_id in range(n_items):
            scores[item_id] = self.predict(user_id, item_id)

        return scores


class ItemBasedCF:
    """
    Item-based Collaborative Filtering

    Recommends items based on item similarity.

    Parameters
    ----------
    similarity : {'cosine', 'pearson', 'euclidean'}, default='cosine'
        Similarity metric to use
    k_neighbors : int, default=5
        Number of similar items to consider

    Attributes
    ----------
    user_item_matrix_ : np.ndarray, shape (n_users, n_items)
        User-item rating matrix
    item_similarity_ : np.ndarray, shape (n_items, n_items)
        Item similarity matrix

    Examples
    --------
    >>> model = ItemBasedCF(similarity='cosine', k_neighbors=3)
    >>> model.fit(ratings)
    >>> scores = model.predict_for_user(user_id=0)
    """

    def __init__(
        self,
        similarity: Literal["cosine", "pearson", "euclidean"] = "cosine",
        k_neighbors: int = 5,
    ):
        self.similarity = similarity
        self.k_neighbors = k_neighbors
        self.user_item_matrix_: Optional[np.ndarray] = None
        self.item_similarity_: Optional[np.ndarray] = None
        self.observed_mask_: Optional[np.ndarray] = None
        self.reference_users_: Optional[List[int]] = None

    def fit(
        self,
        user_item_matrix: np.ndarray,
        observed_mask: Optional[np.ndarray] = None,
        reference_users: Optional[List[int]] = None,
        item_similarity: Optional[np.ndarray] = None,
    ) -> "ItemBasedCF":
        """
        Fit the model by computing item similarity matrix.

        Parameters
        ----------
        user_item_matrix : np.ndarray, shape (n_users, n_items)
            User-item rating matrix

        Returns
        -------
        self : ItemBasedCF
            Fitted model
        """
        self.user_item_matrix_ = user_item_matrix.copy()
        if observed_mask is None:
            self.observed_mask_ = user_item_matrix > 0
        else:
            if observed_mask.shape != user_item_matrix.shape:
                raise ValueError("observed_mask must match user_item_matrix shape")
            self.observed_mask_ = observed_mask.astype(bool, copy=True)
        self.reference_users_ = (
            list(reference_users) if reference_users is not None else None
        )

        if item_similarity is not None:
            expected_shape = (user_item_matrix.shape[1], user_item_matrix.shape[1])
            if item_similarity.shape != expected_shape:
                raise ValueError("item_similarity has the wrong shape")
            self.item_similarity_ = item_similarity.astype(np.float64, copy=True)
            np.fill_diagonal(self.item_similarity_, 0)
            return self

        # Transpose to get item-user matrix
        item_user_matrix = user_item_matrix.T

        # Compute item similarity matrix
        if self.similarity == "cosine":
            if reference_users is not None:
                self.item_similarity_ = compute_item_cosine_similarity(
                    user_item_matrix, self.observed_mask_, reference_users
                )
            elif observed_mask is None:
                self.item_similarity_ = cosine_similarity(item_user_matrix)
            else:
                self.item_similarity_ = compute_item_cosine_similarity(
                    user_item_matrix,
                    self.observed_mask_,
                    list(range(user_item_matrix.shape[0])),
                )
        elif self.similarity == "pearson":
            # TODO: Implement Pearson correlation
            raise NotImplementedError("Pearson correlation not yet implemented")
        elif self.similarity == "euclidean":
            self.item_similarity_ = 1 / (
                1 + pairwise_distances(item_user_matrix, metric="euclidean")
            )
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity}")

        # Set diagonal to 0
        np.fill_diagonal(self.item_similarity_, 0)

        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a specific user-item pair.

        Parameters
        ----------
        user_id : int
            User index
        item_id : int
            Item index

        Returns
        -------
        score : float
            Predicted rating score
        """
        if (
            self.user_item_matrix_ is None
            or self.item_similarity_ is None
            or self.observed_mask_ is None
        ):
            raise ValueError("Model not fitted. Call fit() first.")

        # Find k most similar items that the user has rated
        item_sims = self.item_similarity_[item_id]
        rated_mask = self.observed_mask_[user_id]

        # Get similar items that the user rated
        similar_items_mask = rated_mask & (item_sims > 0)

        if not similar_items_mask.any():
            # No similar items rated by user, return mean rating
            item_ratings = self.user_item_matrix_[:, item_id]
            observed = self.observed_mask_[:, item_id]
            return float(item_ratings[observed].mean()) if observed.any() else 0.0

        # Get top-k similar items
        similar_items_idx = _top_positive_indices(
            item_sims, similar_items_mask, self.k_neighbors
        )

        # Weighted average of ratings
        sims = item_sims[similar_items_idx]
        ratings = self.user_item_matrix_[user_id, similar_items_idx]

        # Avoid division by zero
        if sims.sum() == 0:
            return ratings.mean()

        return np.dot(sims, ratings) / sims.sum()

    def predict_for_user(self, user_id: int) -> np.ndarray:
        """
        Predict ratings for all items for a given user.

        Parameters
        ----------
        user_id : int
            User index

        Returns
        -------
        scores : np.ndarray, shape (n_items,)
            Predicted scores for all items
        """
        if self.user_item_matrix_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        n_items = self.user_item_matrix_.shape[1]
        scores = np.zeros(n_items)

        for item_id in range(n_items):
            scores[item_id] = self.predict(user_id, item_id)

        return scores


def generate_recommendation_list(
    scores: np.ndarray,
    n_recommendations: int = 10,
    exclude_indices: Optional[List[int]] = None,
) -> List[Tuple[int, float]]:
    """
    Generate top-N recommendation list from scores.

    Parameters
    ----------
    scores : np.ndarray, shape (n_items,)
        Recommendation scores for items
    n_recommendations : int, default=10
        Number of recommendations to return
    exclude_indices : List[int], optional
        Item indices to exclude from recommendations

    Returns
    -------
    recommendations : List[Tuple[int, float]]
        List of (item_index, score) tuples, sorted by score descending

    Examples
    --------
    >>> scores = np.array([0.5, 0.9, 0.3, 0.7])
    >>> recommendations = generate_recommendation_list(scores, n_recommendations=2)
    >>> print(recommendations)
    [(1, 0.9), (3, 0.7)]
    """
    if exclude_indices is None:
        exclude_indices = []

    # Create mask for items to consider
    mask = np.ones(len(scores), dtype=bool)
    mask[exclude_indices] = False

    # Get top-N items
    valid_scores = scores[mask]
    valid_indices = np.where(mask)[0]

    # Sort by score descending
    top_n_idx = np.argsort(valid_scores)[::-1][:n_recommendations]

    recommendations = [
        (int(valid_indices[i]), float(valid_scores[i])) for i in top_n_idx
    ]

    return recommendations
