"""
Integration Tests for Recommendation System

Tests the complete workflow from data loading to evaluation.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.collaborative_filtering import UserBasedCF, ItemBasedCF
from models.ranking import scores_to_points, ranking_to_points, scores_to_ranking
from models.ensemble_learner import EnsembleLearner
from models.preference_vector import (
    calculate_preference_vector,
    apply_preference_vector,
    extend_preference_vector,
)
from evaluation.metrics import rmse, ndcg_at_k, evaluate_all_metrics


class TestBasicWorkflow:
    """Test basic recommendation workflow."""

    @pytest.fixture
    def sample_user_item_matrix(self):
        """Create a sample user-item matrix."""
        np.random.seed(42)
        matrix = np.random.rand(4, 50) * 10
        # Add some zeros (missing ratings)
        mask = np.random.rand(4, 50) < 0.3
        matrix[mask] = 0
        return matrix

    def test_collaborative_filtering_workflow(self, sample_user_item_matrix):
        """Test User-based and Item-based CF workflow."""
        # Train models
        user_cf = UserBasedCF(similarity="cosine", k_neighbors=3)
        user_cf.fit(sample_user_item_matrix)

        item_cf = ItemBasedCF(similarity="cosine", k_neighbors=3)
        item_cf.fit(sample_user_item_matrix)

        # Generate predictions for user 0
        user_scores = user_cf.predict_for_user(user_id=0)
        item_scores = item_cf.predict_for_user(user_id=0)

        # Check shape
        assert user_scores.shape == (50,)
        assert item_scores.shape == (50,)

        # Check values are reasonable (0-10 scale)
        assert np.all(user_scores >= 0)
        assert np.all(item_scores >= 0)
        assert np.all(user_scores <= 15)  # Allow some buffer for CF
        assert np.all(item_scores <= 15)

    def test_ranking_conversion_workflow(self):
        """Test score to ranking to points conversion."""
        scores = np.array([8.5, 7.0, 9.0, 6.5, 8.0])

        # Convert to rankings
        rankings = scores_to_ranking(scores)

        # Rankings should be 0-indexed
        assert rankings.min() == 0
        assert rankings.max() == len(scores) - 1

        # Best score (9.0) should have rank 0
        assert rankings[2] == 0

        # Convert to points
        points = ranking_to_points(rankings, max_points=100.0)

        # Best rank should have highest points
        assert points[2] == max(points)

        # Worst rank should have 0 points
        assert points[rankings == len(scores) - 1] == 0

    def test_ensemble_learning_workflow(self, sample_user_item_matrix):
        """Test ensemble learning with F1 and F2 methods."""
        # Train CF models
        user_cf = UserBasedCF(similarity="cosine", k_neighbors=3)
        user_cf.fit(sample_user_item_matrix)

        item_cf = ItemBasedCF(similarity="cosine", k_neighbors=3)
        item_cf.fit(sample_user_item_matrix)

        # Get predictions
        user_scores = user_cf.predict_for_user(user_id=0)
        item_scores = item_cf.predict_for_user(user_id=0)

        # Convert to points
        user_points = scores_to_points(user_scores, max_points=100.0)
        item_points = scores_to_points(item_scores, max_points=100.0)

        # Create impression ranking (simulate user feedback)
        impression_ranking = np.random.rand(50) * 100

        # Test F1 method (3 models)
        learner = EnsembleLearner(method="mean")
        f1_result = learner.ensemble_f1(user_points, item_points, impression_ranking)

        assert f1_result.shape == (50,)
        assert np.all(f1_result >= 0)
        assert np.all(f1_result <= 100)

        # Test F2 method (2 models)
        f2_result = learner.ensemble_f2(user_points, item_points)

        assert f2_result.shape == (50,)
        assert np.all(f2_result >= 0)
        assert np.all(f2_result <= 100)


class TestF1Method:
    """Test F1 method (with Impression Ranking)."""

    def test_f1_complete_workflow(self):
        """Test complete F1 workflow from CF to ensemble."""
        np.random.seed(42)

        # Create sample data
        user_item_matrix = np.random.rand(4, 50) * 10

        # Step 1: Train CF models
        user_cf = UserBasedCF(similarity="cosine", k_neighbors=3)
        user_cf.fit(user_item_matrix)

        item_cf = ItemBasedCF(similarity="cosine", k_neighbors=3)
        item_cf.fit(user_item_matrix)

        # Step 2: Get recommendations for User A (index 0)
        user_scores = user_cf.predict_for_user(user_id=0)
        item_scores = item_cf.predict_for_user(user_id=0)

        # Step 3: Convert to points (Equation 10)
        user_points = scores_to_points(user_scores, max_points=100.0)
        item_points = scores_to_points(item_scores, max_points=100.0)

        # Step 4: Simulate user creating impression ranking
        # User reorders top 10 items based on their preference
        top_10_items = np.argsort(user_scores)[::-1][:10]
        np.random.shuffle(top_10_items)  # User reorders

        # Create impression ranking vector
        impression_ranking = np.zeros(50)
        for rank, item_id in enumerate(top_10_items):
            impression_ranking[item_id] = ((10 - rank - 1) / 10) * 100

        # Step 5: Ensemble (Equation 5)
        learner = EnsembleLearner(method="mean")
        final_scores = learner.ensemble_f1(user_points, item_points, impression_ranking)

        # Verify results
        assert final_scores.shape == (50,)
        assert np.all(final_scores >= 0)

        # Items with impression ranking should be influenced
        assert np.any(final_scores[top_10_items] > 0)


class TestF2Method:
    """Test F2 method (with Preference Vector)."""

    def test_f2_complete_workflow(self):
        """Test complete F2 workflow with preference vector."""
        np.random.seed(42)

        # Create sample data (User A rated 30 items)
        user_item_matrix = np.random.rand(4, 50) * 10

        # User A initially rates only 30 items
        user_a_30_indices = np.random.choice(50, 30, replace=False)
        training_matrix = user_item_matrix.copy()
        all_indices = set(range(50))
        test_indices = list(all_indices - set(user_a_30_indices))
        training_matrix[0, test_indices] = 0

        # Step 1: Train CF models on training data
        user_cf = UserBasedCF(similarity="cosine", k_neighbors=3)
        user_cf.fit(training_matrix)

        item_cf = ItemBasedCF(similarity="cosine", k_neighbors=3)
        item_cf.fit(training_matrix)

        # Step 2: Get initial recommendations (30 items)
        user_scores_30 = user_cf.predict_for_user(user_id=0)
        item_scores_30 = item_cf.predict_for_user(user_id=0)

        # Convert to points
        user_points_30 = scores_to_points(user_scores_30, max_points=100.0)
        item_points_30 = item_scores_30_points = scores_to_points(
            item_scores_30, max_points=100.0
        )

        # Initial ensemble for 30 items
        learner = EnsembleLearner(method="mean")
        initial_rec = learner.ensemble_f2(user_points_30, item_scores_30_points)

        # Step 3: User adjusts scores (simulate feedback)
        user_adjusted = initial_rec.copy()
        # Simulate: User likes items 0-9 more, dislikes 10-19
        user_adjusted[:10] = np.minimum(user_adjusted[:10] + 10, 100)
        user_adjusted[10:20] = np.maximum(user_adjusted[10:20] - 10, 0)

        # Step 4: Calculate preference vector (Equation 11)
        pref_vector = calculate_preference_vector(user_adjusted, initial_rec)

        # Verify preference vector
        assert pref_vector.shape == (50,)
        # Items 0-9 should have positive preference
        assert np.mean(pref_vector[:10]) > 0
        # Items 10-19 should have negative preference
        assert np.mean(pref_vector[10:20]) < 0

        # Step 5: Apply preference vector to CF scores
        user_scores_adjusted = apply_preference_vector(
            user_points_30, pref_vector, scale_factor=1.0
        )
        item_scores_adjusted = apply_preference_vector(
            item_scores_30_points, pref_vector, scale_factor=1.0
        )

        # Step 6: Final ensemble
        final_scores = learner.ensemble_f2(user_scores_adjusted, item_scores_adjusted)

        # Verify results
        assert final_scores.shape == (50,)
        # Items with positive preference should have higher scores
        assert np.mean(final_scores[:10]) > np.mean(initial_rec[:10])


class TestEvaluation:
    """Test evaluation metrics."""

    def test_evaluation_workflow(self):
        """Test evaluation with all metrics."""
        np.random.seed(42)

        # Generate predictions and ground truth
        predictions = np.random.rand(50) * 10
        ground_truth = np.random.rand(50) * 10

        # Test individual metrics
        rmse_val = rmse(predictions, ground_truth)
        assert 0 <= rmse_val <= 10  # Reasonable range

        ndcg_val = ndcg_at_k(predictions, ground_truth, k=10)
        assert 0 <= ndcg_val <= 1  # NDCG is normalized

        # Test all metrics at once
        metrics = evaluate_all_metrics(predictions, ground_truth, k_values=[5, 10, 20])

        # Check all expected metrics are present
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "ndcg@5" in metrics
        assert "ndcg@10" in metrics
        assert "precision@5" in metrics
        assert "recall@10" in metrics

        # All metric values should be valid
        for key, value in metrics.items():
            assert isinstance(value, float)
            assert not np.isnan(value)
            assert not np.isinf(value)


class TestPaperScenario:
    """Test the exact scenario described in the paper."""

    def test_paper_30_50_scenario(self):
        """
        Test the paper's scenario:
        - 4 users (A, B, C, D)
        - 50 cosmetics items
        - User A rates 30 items, system recommends remaining 20
        """
        np.random.seed(42)

        # Create full dataset (4 users × 50 items)
        full_matrix = np.random.rand(4, 50) * 10

        # User A (index 0) rates only 30 items
        user_a_30_indices = sorted(np.random.choice(50, 30, replace=False).tolist())
        user_a_20_indices = sorted(list(set(range(50)) - set(user_a_30_indices)))

        # Create training matrix
        training_matrix = full_matrix.copy()
        training_matrix[0, user_a_20_indices] = 0

        # Train CF models
        user_cf = UserBasedCF(similarity="cosine", k_neighbors=3)
        user_cf.fit(training_matrix)

        item_cf = ItemBasedCF(similarity="cosine", k_neighbors=3)
        item_cf.fit(training_matrix)

        # Get predictions for all 50 items
        user_scores = user_cf.predict_for_user(user_id=0)
        item_scores = item_cf.predict_for_user(user_id=0)

        # Ensemble
        learner = EnsembleLearner(method="mean")
        user_points = scores_to_points(user_scores)
        item_points = scores_to_points(item_scores)
        ensemble_scores = learner.ensemble_f2(user_points, item_points)

        # Evaluate on the 20 test items
        test_predictions = ensemble_scores[user_a_20_indices]
        test_ground_truth = full_matrix[0, user_a_20_indices]

        # Calculate RMSE
        rmse_val = rmse(test_predictions, test_ground_truth)

        # RMSE should be reasonable (not too high)
        assert rmse_val < 50  # Very lenient check for this test

        # Verify we made predictions for all 20 test items
        assert len(test_predictions) == 20
        assert np.all(test_predictions >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
