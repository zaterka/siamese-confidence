"""
Utility functions for data loading and preprocessing.

This module contains helper functions for:
- Loading embeddings from JSON files
- Generating training pairs for Siamese learning
"""

import json
import random
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_embeddings(train_path: str) -> List[Dict]:
    """
    Load training embeddings from JSON file.

    Expected format:
    {
        "<doc_name>": {
            "embedding": [<floats>],
            "Page": <int>,
            "Class": "<class_name>"
        },
        ...
    }

    Args:
        train_path: Path to the JSON file containing embeddings

    Returns:
        List of dicts with keys: 'doc_name', 'class', 'embedding'
    """
    with open(train_path, "r") as f:
        raw_train = json.load(f)

    train_records = []
    for doc_key, info in raw_train.items():
        emb = info.get("embedding", None)
        cls = info.get("Class", None)
        if emb is None or cls is None:
            continue
        arr = np.array(emb, dtype=float)
        train_records.append({"doc_name": doc_key, "class": cls, "embedding": arr})

    return train_records


def generate_training_pairs(
    train_records: List[Dict],
    max_pairs_per_class: int = 200,
    neg_ratio: float = 1.0,
    random_seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate pairs of embeddings for Siamese training with advanced feature engineering.
    Uses vector differences for richer representation.
    Ensures no pairs are created from pages of the same document to prevent data leakage.

    Args:
        train_records: List of training records with embeddings
        max_pairs_per_class: Maximum number of positive pairs per class
        neg_ratio: Ratio of negative pairs to positive pairs
        random_seed: Random seed for reproducibility
        verbose: Whether to print filtering statistics

    Returns:
        features: Array of engineered features (vector differences)
        labels: Array of labels (1 = same class, 0 = different class)
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Group records by class
    class_to_indices = {}
    for idx, record in enumerate(train_records):
        cls = record["class"]
        class_to_indices.setdefault(cls, []).append(idx)

    # Generate positive pairs (same class, different documents)
    pos_pairs = []
    for cls, indices in class_to_indices.items():
        if len(indices) < 2:
            continue
        # All possible combinations within the class
        all_combos = list(combinations(indices, 2))

        # Filter out pairs from the same document
        valid_combos = []
        for i, j in all_combos:
            doc_i = train_records[i]["doc_name"]
            doc_j = train_records[j]["doc_name"]
            if doc_i != doc_j:  # Only keep pairs from different documents
                valid_combos.append((i, j))

        random.shuffle(valid_combos)
        selected = valid_combos[:max_pairs_per_class]
        pos_pairs.extend(selected)

    # Generate negative pairs (different classes)
    neg_pairs = []
    all_indices = list(range(len(train_records)))
    num_pos = len(pos_pairs)
    num_neg = int(num_pos * neg_ratio)

    attempts = 0
    max_attempts = num_neg * 10  # Prevent infinite loop

    while len(neg_pairs) < num_neg and attempts < max_attempts:
        i, j = random.sample(all_indices, 2)
        if (
            train_records[i]["class"] != train_records[j]["class"]
            and train_records[i]["doc_name"] != train_records[j]["doc_name"]
        ):
            neg_pairs.append((i, j))
        attempts += 1

    # Generate vector differences
    def create_features(emb_i, emb_j):
        """Create vector differences from embedding pairs."""
        return emb_i - emb_j

    features = []
    labels = []

    # Positive pairs
    for i, j in pos_pairs:
        emb_i = train_records[i]["embedding"]
        emb_j = train_records[j]["embedding"]
        feature = create_features(emb_i, emb_j)
        features.append(feature)
        labels.append(1)  # Same class

    # Negative pairs
    for i, j in neg_pairs:
        emb_i = train_records[i]["embedding"]
        emb_j = train_records[j]["embedding"]
        feature = create_features(emb_i, emb_j)
        features.append(feature)
        labels.append(0)  # Different class

    if verbose:
        feature_dim = len(features[0]) if features else 0
        print(f"Generated {len(features)} pairs:")
        print(f"  - Positive pairs (same class): {np.sum(labels)}")
        print(f"  - Negative pairs (different class): {len(labels) - np.sum(labels)}")
        print(f"  - Feature dimension: {feature_dim}")

    return np.array(features), np.array(labels)
