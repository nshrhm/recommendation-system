"""
Ensemble Learning for Recommendation Systems

Combines multiple recommendation results into a single unified result.

Implements:
- F1 Method: User-based CF + Item-based CF + Impression Ranking
- F2 Method: User-based CF + Item-based CF (with preference vectors)

References:
    - Paper Equation (5): F1 ensemble learning
    - Paper Equation (6): F2 ensemble learning
"""

import numpy as np
from typing import List, Optional, Tuple, Literal
import warnings


class EnsembleLearner:
    """
    Ensemble Learning for Recommendation Systems.

    Combines multiple recommendation results using averaging methods.

    Parameters
    ----------
    method : {'mean', 'weighted_mean'}, default='mean'
        Ensemble method to use
    weights : Optional[List[float]], default=None
        Weights for each recommendation model (used only with 'weighted_mean')
        If None, equal weights are used

    Attributes
    ----------
    method : str
        The ensemble method
    weights : Optional[np.ndarray]
        Normalized weights for each model

    Examples
    --------
    >>> import numpy as np
    >>> from src.models.ensemble_learner import EnsembleLearner
    >>>
    >>> # Create sample recommendation scores
    >>> user_cf = np.array([0.8, 0.6, 0.4, 0.2])
    >>> item_cf = np.array([0.7, 0.5, 0.6, 0.3])
    >>> impression = np.array([0.9, 0.4, 0.5, 0.1])
    >>>
    >>> # F1 method: ensemble 3 models
    >>> learner = EnsembleLearner(method='mean')
    >>> result = learner.ensemble([user_cf, item_cf, impression])
    >>> print(result)
    [0.8 0.5 0.5 0.2]
    """

    def __init__(
        self,
        method: Literal["mean", "weighted_mean"] = "mean",
        weights: Optional[List[float]] = None,
    ):
        self.method = method
        self.weights = None

        if weights is not None:
            self.weights = np.array(weights, dtype=np.float64)
            # Normalize weights to sum to 1
            self.weights = self.weights / self.weights.sum()

    def ensemble(
        self, recommendations: List[np.ndarray], normalize: bool = False
    ) -> np.ndarray:
        """
        Ensemble multiple recommendation results.

        Parameters
        ----------
        recommendations : List[np.ndarray]
            List of recommendation score arrays, each with shape (n_items,)
            All arrays must have the same length
        normalize : bool, default=False
            If True, normalize each recommendation before ensembling

        Returns
        -------
        ensemble_result : np.ndarray, shape (n_items,)
            Ensembled recommendation scores

        Raises
        ------
        ValueError
            If recommendations list is empty or arrays have different lengths
        """
        if len(recommendations) == 0:
            raise ValueError("recommendations list cannot be empty")

        # Check all arrays have same shape
        n_items = recommendations[0].shape[0]
        for i, rec in enumerate(recommendations):
            if rec.shape[0] != n_items:
                raise ValueError(
                    f"All recommendation arrays must have same length. "
                    f"Expected {n_items}, got {rec.shape[0]} at index {i}"
                )

        # Stack recommendations into matrix (m_models, n_items)
        rec_matrix = np.vstack(recommendations)

        # Normalize if requested
        if normalize:
            rec_matrix = self._normalize_rows(rec_matrix)

        # Perform ensemble
        if self.method == "mean":
            ensemble_result = self._ensemble_mean(rec_matrix)
        elif self.method == "weighted_mean":
            ensemble_result = self._ensemble_weighted_mean(rec_matrix, self.weights)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

        return ensemble_result

    def ensemble_f1(
        self,
        user_cf_scores: np.ndarray,
        item_cf_scores: np.ndarray,
        impression_ranking: np.ndarray,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        F1 Method: Ensemble User-based CF, Item-based CF, and Impression Ranking.

        Implements Equation (5) from the paper:
        R_i = (R_Ui + R_Ii + R_Ni) / m

        where:
        - R_Ui: User-based collaborative filtering scores
        - R_Ii: Item-based collaborative filtering scores
        - R_Ni: User's impression ranking (manually adjusted)
        - m: number of models (3 in this case)

        Parameters
        ----------
        user_cf_scores : np.ndarray, shape (n_items,)
            Recommendation scores from User-based CF
        item_cf_scores : np.ndarray, shape (n_items,)
            Recommendation scores from Item-based CF
        impression_ranking : np.ndarray, shape (n_items,)
            User's impression ranking scores (from Equation 10)
        normalize : bool, default=False
            If True, normalize scores before ensembling

        Returns
        -------
        ensemble_scores : np.ndarray, shape (n_items,)
            Ensembled recommendation scores

        Examples
        --------
        >>> learner = EnsembleLearner()
        >>> user_scores = np.array([80., 60., 40., 20.])
        >>> item_scores = np.array([70., 50., 60., 30.])
        >>> impression = np.array([90., 40., 50., 10.])
        >>> result = learner.ensemble_f1(user_scores, item_scores, impression)
        >>> # result = (80+70+90)/3, (60+50+40)/3, (40+60+50)/3, (20+30+10)/3
        >>> print(result)
        [80. 50. 50. 20.]
        """
        # Equation (5): R_i = (R_Ui + R_Ii + R_Ni) / m
        recommendations = [user_cf_scores, item_cf_scores, impression_ranking]
        return self.ensemble(recommendations, normalize=normalize)

    def ensemble_f2(
        self,
        user_cf_scores: np.ndarray,
        item_cf_scores: np.ndarray,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        F2 Method: Ensemble User-based CF and Item-based CF (with preference vectors).

        Implements Equation (6) from the paper:
        R_i = (R_Ui + R_Ii) / m

        where:
        - R_Ui: User-based CF scores (with preference vector applied)
        - R_Ii: Item-based CF scores (with preference vector applied)
        - m: number of models (2 in this case)

        Note: Preference vectors should be applied to CF scores BEFORE
        calling this method (see preference_vector.py).

        Parameters
        ----------
        user_cf_scores : np.ndarray, shape (n_items,)
            Recommendation scores from User-based CF (with preference applied)
        item_cf_scores : np.ndarray, shape (n_items,)
            Recommendation scores from Item-based CF (with preference applied)
        normalize : bool, default=False
            If True, normalize scores before ensembling

        Returns
        -------
        ensemble_scores : np.ndarray, shape (n_items,)
            Ensembled recommendation scores

        Examples
        --------
        >>> learner = EnsembleLearner()
        >>> user_scores = np.array([85., 50., 30., 20.])
        >>> item_scores = np.array([75., 60., 40., 30.])
        >>> result = learner.ensemble_f2(user_scores, item_scores)
        >>> # result = (85+75)/2, (50+60)/2, (30+40)/2, (20+30)/2
        >>> print(result)
        [80. 55. 35. 25.]
        """
        # Equation (6): R_i = (R_Ui + R_Ii) / m
        recommendations = [user_cf_scores, item_cf_scores]
        return self.ensemble(recommendations, normalize=normalize)

    def _ensemble_mean(self, rec_matrix: np.ndarray) -> np.ndarray:
        """
        Simple mean ensemble.

        Parameters
        ----------
        rec_matrix : np.ndarray, shape (m_models, n_items)
            Recommendation scores from multiple models

        Returns
        -------
        ensemble : np.ndarray, shape (n_items,)
            Mean of all models
        """
        return np.mean(rec_matrix, axis=0)

    def _ensemble_weighted_mean(
        self, rec_matrix: np.ndarray, weights: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Weighted mean ensemble.

        Parameters
        ----------
        rec_matrix : np.ndarray, shape (m_models, n_items)
            Recommendation scores from multiple models
        weights : Optional[np.ndarray], shape (m_models,)
            Weights for each model

        Returns
        -------
        ensemble : np.ndarray, shape (n_items,)
            Weighted mean of all models
        """
        m_models = rec_matrix.shape[0]

        if weights is None:
            # Fall back to simple mean
            return self._ensemble_mean(rec_matrix)

        if len(weights) != m_models:
            warnings.warn(
                f"Number of weights ({len(weights)}) does not match "
                f"number of models ({m_models}). Using simple mean instead.",
                UserWarning,
            )
            return self._ensemble_mean(rec_matrix)

        # Weighted average: sum(w_i * r_i) / sum(w_i)
        # Note: weights are already normalized in __init__
        return np.dot(weights, rec_matrix)

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize each row of a matrix to [0, 1] range.

        Parameters
        ----------
        matrix : np.ndarray, shape (m, n)
            Matrix to normalize

        Returns
        -------
        normalized : np.ndarray, shape (m, n)
            Normalized matrix
        """
        normalized = np.zeros_like(matrix, dtype=np.float64)

        for i in range(matrix.shape[0]):
            row = matrix[i]
            min_val = row.min()
            max_val = row.max()

            if max_val == min_val:
                # All values are the same
                normalized[i] = 0.5
            else:
                normalized[i] = (row - min_val) / (max_val - min_val)

        return normalized


def optimize_ensemble_weights(
    recommendations: List[np.ndarray],
    ground_truth: np.ndarray,
    metric: Literal["rmse", "mae"] = "rmse",
) -> np.ndarray:
    """
    Optimize ensemble weights using grid search.

    This function is useful for finding optimal weights when you have
    ground truth data available.

    Parameters
    ----------
    recommendations : List[np.ndarray]
        List of recommendation score arrays
    ground_truth : np.ndarray
        True ratings or scores
    metric : {'rmse', 'mae'}, default='rmse'
        Evaluation metric to minimize

    Returns
    -------
    optimal_weights : np.ndarray
        Optimal weights for each model

    Examples
    --------
    >>> user_cf = np.array([0.8, 0.6, 0.4])
    >>> item_cf = np.array([0.7, 0.5, 0.6])
    >>> ground_truth = np.array([0.75, 0.55, 0.55])
    >>> weights = optimize_ensemble_weights([user_cf, item_cf], ground_truth)
    """
    from itertools import product

    n_models = len(recommendations)

    # Create weight grid (increments of 0.1)
    weight_range = np.arange(0.1, 1.1, 0.1)
    weight_combinations = list(product(weight_range, repeat=n_models))

    # Filter to only normalized weights (sum to ~1.0)
    valid_combinations = [w for w in weight_combinations if abs(sum(w) - 1.0) < 0.01]

    best_weights = None
    best_score = float("inf")

    learner = EnsembleLearner(method="weighted_mean")

    for weights in valid_combinations:
        learner.weights = np.array(weights)
        ensemble_result = learner.ensemble(recommendations)

        # Calculate error
        if metric == "rmse":
            error = np.sqrt(np.mean((ensemble_result - ground_truth) ** 2))
        elif metric == "mae":
            error = np.mean(np.abs(ensemble_result - ground_truth))
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if error < best_score:
            best_score = error
            best_weights = weights

    return np.array(best_weights)
