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

    def fit(self, user_item_matrix: np.ndarray) -> "UserBasedCF":
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

        # Compute user similarity matrix
        if self.similarity == "cosine":
            self.user_similarity_ = cosine_similarity(user_item_matrix)
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
        if self.user_item_matrix_ is None or self.user_similarity_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Find k most similar users who rated this item
        user_sims = self.user_similarity_[user_id]
        rated_mask = self.user_item_matrix_[:, item_id] > 0

        # Get similar users who rated the item
        similar_users_mask = rated_mask & (user_sims > 0)

        if not similar_users_mask.any():
            # No similar users rated this item, return mean rating
            user_ratings = self.user_item_matrix_[user_id]
            return (
                user_ratings[user_ratings > 0].mean()
                if (user_ratings > 0).any()
                else 0.0
            )

        # Get top-k similar users
        similar_users = np.argsort(user_sims[similar_users_mask])[-self.k_neighbors :]
        similar_users_idx = np.where(similar_users_mask)[0][similar_users]

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

    def fit(self, user_item_matrix: np.ndarray) -> "ItemBasedCF":
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

        # Transpose to get item-user matrix
        item_user_matrix = user_item_matrix.T

        # Compute item similarity matrix
        if self.similarity == "cosine":
            self.item_similarity_ = cosine_similarity(item_user_matrix)
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
        if self.user_item_matrix_ is None or self.item_similarity_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Find k most similar items that the user has rated
        item_sims = self.item_similarity_[item_id]
        rated_mask = self.user_item_matrix_[user_id] > 0

        # Get similar items that the user rated
        similar_items_mask = rated_mask & (item_sims > 0)

        if not similar_items_mask.any():
            # No similar items rated by user, return mean rating
            item_ratings = self.user_item_matrix_[:, item_id]
            return (
                item_ratings[item_ratings > 0].mean()
                if (item_ratings > 0).any()
                else 0.0
            )

        # Get top-k similar items
        similar_items = np.argsort(item_sims[similar_items_mask])[-self.k_neighbors :]
        similar_items_idx = np.where(similar_items_mask)[0][similar_items]

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
