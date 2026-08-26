"""
Dataset Loader for Cosmetics Recommendation System

Handles loading and preprocessing of cosmetics evaluation data.

Data structure:
- 4 users (A, B, C, D)
- 50 cosmetics items
- 6 categories: Makeup Bases, Powders, Foundations, Lipsticks, Eyeshadows, Eyeliners
- Ratings: 0-10 scale (0 means not rated)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
import os


# Category definitions from the paper
COSMETICS_CATEGORIES = {
    "Makeup Bases": list(range(0, 5)),  # Items 1-5 (indices 0-4)
    "Powders": list(range(5, 9)),  # Items 6-9 (indices 5-8)
    "Foundations": list(range(9, 20)),  # Items 10-20 (indices 9-19)
    "Lipsticks": list(range(20, 30)),  # Items 21-30 (indices 20-29)
    "Eyeshadows": list(range(30, 45)),  # Items 31-45 (indices 30-44)
    "Eyeliners": list(range(45, 50)),  # Items 46-50 (indices 45-49)
}

USER_IDS = ["A", "B", "C", "D"]

REVISED_CATEGORY_COUNTS = {
    "Makeup Bases": 5,
    "Powders": 4,
    "Foundations": 11,
    "Lipsticks/Lip Gloss": 10,
    "Eyeshadows": 15,
    "Eyeliners": 5,
}

REVISED_USER_BIASES = {
    "A": 0.5,
    "B": 0.3,
    "C": 0.0,
    "D": -0.5,
}


def load_cosmetics_dataset(filepath: str, format: str = "csv") -> pd.DataFrame:
    """
    Load cosmetics evaluation data from file.

    Parameters
    ----------
    filepath : str
        Path to the data file
    format : {'csv', 'excel', 'json'}, default='csv'
        File format

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: ['user_id', 'item_id', 'rating', 'category']

    Examples
    --------
    >>> df = load_cosmetics_dataset('data/cosmetics_ratings.csv')
    >>> print(df.head())
       user_id  item_id  rating      category
    0        A        0     8.5  Makeup Bases
    1        A        1     7.0  Makeup Bases
    ...
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Load based on format
    if format == "csv":
        df = pd.read_csv(filepath)
    elif format == "excel":
        df = pd.read_excel(filepath)
    elif format == "json":
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Validate required columns
    required_columns = ["user_id", "item_id", "rating"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Add category if not present
    if "category" not in df.columns:
        df["category"] = df["item_id"].apply(get_item_category)

    return df


def create_user_item_matrix(
    df: pd.DataFrame, n_users: int = 4, n_items: int = 50
) -> np.ndarray:
    """
    Create User-Item rating matrix from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: ['user_id', 'item_id', 'rating']
    n_users : int, default=4
        Number of users
    n_items : int, default=50
        Number of items

    Returns
    -------
    matrix : np.ndarray, shape (n_users, n_items)
        User-item rating matrix (0 means not rated)

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'user_id': ['A', 'A', 'B', 'B'],
    ...     'item_id': [0, 1, 0, 1],
    ...     'rating': [8.5, 7.0, 9.0, 6.5]
    ... })
    >>> matrix = create_user_item_matrix(df)
    >>> print(matrix.shape)
    (4, 50)
    """
    # Initialize matrix with zeros
    matrix = np.zeros((n_users, n_items), dtype=np.float64)

    # Map user IDs to indices
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(USER_IDS)}

    # Fill matrix
    for _, row in df.iterrows():
        user_id = row["user_id"]
        item_id = row["item_id"]
        rating = row["rating"]

        if user_id not in user_id_to_idx:
            continue

        user_idx = user_id_to_idx[user_id]

        if 0 <= item_id < n_items:
            matrix[user_idx, item_id] = rating

    return matrix


def get_item_category(item_id: int) -> str:
    """
    Get category name for an item.

    Parameters
    ----------
    item_id : int
        Item index (0-indexed)

    Returns
    -------
    category : str
        Category name

    Examples
    --------
    >>> get_item_category(0)
    'Makeup Bases'
    >>> get_item_category(25)
    'Lipsticks'
    """
    for category, item_indices in COSMETICS_CATEGORIES.items():
        if item_id in item_indices:
            return category

    return "Unknown"


def get_category_items(category: str) -> List[int]:
    """
    Get item indices for a category.

    Parameters
    ----------
    category : str
        Category name

    Returns
    -------
    item_indices : List[int]
        List of item indices in the category

    Examples
    --------
    >>> get_category_items('Makeup Bases')
    [0, 1, 2, 3, 4]
    """
    return COSMETICS_CATEGORIES.get(category, [])


def generate_synthetic_data(
    n_users: int = 4,
    n_items: int = 50,
    sparsity: float = 0.3,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic cosmetics rating data for testing.

    Parameters
    ----------
    n_users : int, default=4
        Number of users
    n_items : int, default=50
        Number of items
    sparsity : float, default=0.3
        Proportion of missing ratings (0-1)
    random_seed : Optional[int], default=None
        Random seed for reproducibility

    Returns
    -------
    df : pd.DataFrame
        Synthetic rating data

    Examples
    --------
    >>> df = generate_synthetic_data(n_users=4, n_items=50, random_seed=42)
    >>> print(df.shape[0])  # Number of ratings
    140  # Approximately (1 - 0.3) * 4 * 50 = 140
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    data = []

    for user_idx in range(n_users):
        user_id = USER_IDS[user_idx] if user_idx < len(USER_IDS) else f"User_{user_idx}"

        for item_id in range(n_items):
            # Skip based on sparsity
            if np.random.random() < sparsity:
                continue

            # Generate rating (0-10 scale)
            # Add some structure: users prefer certain categories
            category = get_item_category(item_id)
            base_rating = np.random.uniform(3, 9)

            # Add user-specific bias
            user_bias = np.random.normal(0, 1)
            rating = np.clip(base_rating + user_bias, 0, 10)

            data.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": round(rating, 1),
                    "category": category,
                }
            )

    df = pd.DataFrame(data)
    return df


def revised_category_ids(
    category_counts: Dict[str, int]
) -> Tuple[np.ndarray, List[str]]:
    """Return deterministic item-to-category IDs for the revised experiment."""
    names = list(category_counts)
    counts = [int(category_counts[name]) for name in names]
    if len(names) != 6 or sum(counts) != 50 or any(count <= 0 for count in counts):
        raise ValueError(
            "Revised categories must contain six positive counts totaling 50"
        )
    return np.repeat(np.arange(len(names), dtype=np.int64), counts), names


def generate_revised_synthetic_data(
    rng: np.random.Generator,
    *,
    category_counts: Optional[Dict[str, int]] = None,
    user_biases: Optional[Dict[str, float]] = None,
    attractiveness_low: float = 2.0,
    attractiveness_high: float = 8.0,
    category_preference_std: float = 1.0,
    idiosyncratic_noise_std: float = 0.5,
    rating_min: float = 0.0,
    rating_max: float = 10.0,
) -> Dict[str, np.ndarray]:
    """Generate one independent dataset for the revised experiment.

    Draw order is deliberately fixed: item attractiveness, user-category
    preferences, then user-item disturbances. All 200 values are observed;
    missingness for User A is introduced later with an explicit Boolean mask.
    """
    category_counts = category_counts or REVISED_CATEGORY_COUNTS
    user_biases = user_biases or REVISED_USER_BIASES
    category_ids, category_names = revised_category_ids(category_counts)

    if list(user_biases) != USER_IDS:
        raise ValueError(f"user_biases must be ordered as {USER_IDS}")
    if attractiveness_low >= attractiveness_high:
        raise ValueError("attractiveness_low must be less than attractiveness_high")

    item_attractiveness = rng.uniform(attractiveness_low, attractiveness_high, 50)
    category_preferences = rng.normal(0.0, category_preference_std, (4, 6))
    idiosyncratic_noise = rng.normal(0.0, idiosyncratic_noise_std, (4, 50))
    bias_vector = np.asarray([user_biases[user] for user in USER_IDS], dtype=np.float64)

    latent_ratings = (
        item_attractiveness[np.newaxis, :]
        + bias_vector[:, np.newaxis]
        + category_preferences[:, category_ids]
        + idiosyncratic_noise
    )
    ratings = np.clip(latent_ratings, rating_min, rating_max).astype(np.float64)
    observed_mask = np.ones(ratings.shape, dtype=bool)

    return {
        "ratings": ratings,
        "observed_mask": observed_mask,
        "item_attractiveness": item_attractiveness,
        "category_preferences": category_preferences,
        "idiosyncratic_noise": idiosyncratic_noise,
        "category_ids": category_ids,
        "category_names": np.asarray(category_names, dtype=object),
        "user_biases": bias_vector,
    }


def split_revised_user_observations(
    ratings: np.ndarray,
    rng: np.random.Generator,
    *,
    user_idx: int = 0,
    n_train: int = 30,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Create a deterministic random train/test observation mask for one user."""
    if ratings.ndim != 2 or not 0 <= user_idx < ratings.shape[0]:
        raise ValueError("ratings must be 2D and user_idx must be valid")
    if not 0 < n_train < ratings.shape[1]:
        raise ValueError("n_train must be between zero and the number of items")

    train_ids = sorted(
        int(item_id)
        for item_id in rng.choice(ratings.shape[1], size=n_train, replace=False)
    )
    train_set = set(train_ids)
    test_ids = [
        item_id for item_id in range(ratings.shape[1]) if item_id not in train_set
    ]
    observed_mask = np.ones(ratings.shape, dtype=bool)
    observed_mask[user_idx, test_ids] = False
    return observed_mask, train_ids, test_ids


def save_dataset(
    df: pd.DataFrame, filepath: str, format: str = "csv", include_metadata: bool = True
) -> None:
    """
    Save dataset to file.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to save
    filepath : str
        Output file path
    format : {'csv', 'excel', 'json'}, default='csv'
        File format
    include_metadata : bool, default=True
        If True, save metadata JSON alongside the data

    Examples
    --------
    >>> df = generate_synthetic_data(random_seed=42)
    >>> save_dataset(df, 'data/cosmetics_ratings.csv')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save data
    if format == "csv":
        df.to_csv(filepath, index=False)
    elif format == "excel":
        df.to_excel(filepath, index=False)
    elif format == "json":
        df.to_json(filepath, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Save metadata
    if include_metadata:
        metadata = {
            "n_users": df["user_id"].nunique(),
            "n_items": df["item_id"].nunique(),
            "n_ratings": len(df),
            "rating_scale": [0, 10],
            "categories": list(COSMETICS_CATEGORIES.keys()),
            "sparsity": 1.0
            - (len(df) / (df["user_id"].nunique() * df["item_id"].nunique())),
        }

        metadata_path = filepath.with_suffix(".meta.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def split_user_data(
    user_item_matrix: np.ndarray,
    user_idx: int,
    n_rated: int = 30,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Split user's data into rated and unrated items.

    Used to simulate the scenario in the paper where User A
    evaluates 30 items and system recommends remaining 20.

    Parameters
    ----------
    user_item_matrix : np.ndarray, shape (n_users, n_items)
        User-item rating matrix
    user_idx : int
        User index
    n_rated : int, default=30
        Number of items to keep as rated
    random_seed : Optional[int], default=None
        Random seed for reproducibility

    Returns
    -------
    modified_matrix : np.ndarray, shape (n_users, n_items)
        Matrix with some of user's ratings removed
    rated_indices : List[int]
        Indices of items kept as rated
    unrated_indices : List[int]
        Indices of items removed (to be predicted)

    Examples
    --------
    >>> matrix = np.random.rand(4, 50) * 10
    >>> modified, rated, unrated = split_user_data(matrix, user_idx=0, n_rated=30, random_seed=42)
    >>> print(len(rated), len(unrated))
    30 20
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    modified_matrix = user_item_matrix.copy()

    # Get user's rated items
    user_ratings = user_item_matrix[user_idx]
    rated_mask = user_ratings > 0
    rated_item_ids = np.where(rated_mask)[0]

    if len(rated_item_ids) < n_rated:
        raise ValueError(
            f"User has only {len(rated_item_ids)} rated items, "
            f"but n_rated={n_rated} requested"
        )

    # Randomly select items to keep
    np.random.shuffle(rated_item_ids)
    rated_indices = sorted(rated_item_ids[:n_rated].tolist())
    unrated_indices = sorted(rated_item_ids[n_rated:].tolist())

    # Remove ratings for unrated items
    modified_matrix[user_idx, unrated_indices] = 0

    return modified_matrix, rated_indices, unrated_indices


def create_paper_experiment_data(
    output_dir: str = "data", random_seed: int = 42
) -> Dict[str, any]:
    """
    Create datasets matching the paper's experimental setup.

    Creates:
    - Full dataset: 4 users × 50 items
    - User A with 30 items rated (20 hidden for evaluation)

    Parameters
    ----------
    output_dir : str, default='data'
        Directory to save data files
    random_seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    data : Dict
        Dictionary containing:
        - 'full_matrix': Full user-item matrix
        - 'training_matrix': Matrix with User A having only 30 items
        - 'user_a_rated_indices': Indices of 30 rated items for User A
        - 'user_a_test_indices': Indices of 20 test items for User A

    Examples
    --------
    >>> data = create_paper_experiment_data(output_dir='data', random_seed=42)
    >>> print(data['full_matrix'].shape)
    (4, 50)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate full dataset
    df_full = generate_synthetic_data(
        n_users=4,
        n_items=50,
        sparsity=0.0,  # All users rate all items initially
        random_seed=random_seed,
    )

    # Create full matrix
    full_matrix = create_user_item_matrix(df_full)

    # Split User A's data (index 0)
    training_matrix, rated_indices, test_indices = split_user_data(
        full_matrix, user_idx=0, n_rated=30, random_seed=random_seed
    )

    # Save datasets
    save_dataset(df_full, output_path / "cosmetics_full.csv")

    # Create training DataFrame
    df_training = []
    for user_idx, user_id in enumerate(USER_IDS):
        for item_id in range(50):
            rating = training_matrix[user_idx, item_id]
            if rating > 0:
                df_training.append(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "rating": rating,
                        "category": get_item_category(item_id),
                    }
                )

    df_training = pd.DataFrame(df_training)
    save_dataset(df_training, output_path / "cosmetics_training.csv")

    # Save split indices
    split_info = {
        "user_a_rated_indices": rated_indices,
        "user_a_test_indices": test_indices,
    }

    with open(output_path / "user_a_split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    return {
        "full_matrix": full_matrix,
        "training_matrix": training_matrix,
        "user_a_rated_indices": rated_indices,
        "user_a_test_indices": test_indices,
        "df_full": df_full,
        "df_training": df_training,
    }
