"""
Core Siamese model implementation and training functions.

This module contains the main functionality for:
- SiameseModel: Combined StandardScaler + MLP pipeline
- Training functions for Siamese models
- Confidence computation for embeddings
- Model persistence (save/load)
"""

import json
from typing import List, Optional

import numpy as np

from .models import MLPClassifier, StandardScaler
from .utils import generate_training_pairs, load_embeddings


class SiameseModel:
    """Siamese model using MLP with StandardScaler pipeline."""

    def __init__(
        self,
        hidden_layer_sizes=(100, 50),
        activation="relu",
        learning_rate=0.001,
        max_iter=200,
        alpha=0.0001,
        dropout_rate=0.1,
        early_stopping=True,
        random_state=42,
    ):
        self.scaler = StandardScaler()
        self.mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            max_iter=max_iter,
            alpha=alpha,
            dropout_rate=dropout_rate,
            early_stopping=early_stopping,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        """Fit the pipeline: scale then train MLP."""
        if verbose:
            print("Fitting StandardScaler...")
        X_scaled = self.scaler.fit_transform(X)

        if verbose:
            print("Training MLP...")
        self.mlp.fit(X_scaled, y, verbose=verbose)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using the fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.mlp.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)

    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "mlp_weights": [w.tolist() for w in self.mlp.weights_],
            "mlp_biases": [b.tolist() for b in self.mlp.biases_],
            "mlp_params": {
                "hidden_layer_sizes": self.mlp.hidden_layer_sizes,
                "activation": self.mlp.activation,
                "learning_rate": self.mlp.learning_rate,
                "max_iter": self.mlp.max_iter,
                "alpha": self.mlp.alpha,
                "dropout_rate": self.mlp.dropout_rate,
                "early_stopping": self.mlp.early_stopping,
                "random_state": self.mlp.random_state,
            },
            "is_fitted": self.is_fitted,
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

    def load_model(self, filepath: str):
        """Load a trained model."""
        with open(filepath, "r") as f:
            model_data = json.load(f)

        # Restore scaler
        self.scaler.mean_ = np.array(model_data["scaler_mean"])
        self.scaler.scale_ = np.array(model_data["scaler_scale"])
        self.scaler.is_fitted = True

        # Restore MLP parameters
        params = model_data["mlp_params"]
        self.mlp = MLPClassifier(**params)

        # Restore weights and biases
        self.mlp.weights_ = [np.array(w) for w in model_data["mlp_weights"]]
        self.mlp.biases_ = [np.array(b) for b in model_data["mlp_biases"]]
        self.mlp.n_layers_ = len(self.mlp.weights_) + 1
        self.mlp.is_fitted = True

        self.is_fitted = model_data["is_fitted"]
        return self


def train_siamese_model(
    train_path: str,
    model_save_path: str,
    hidden_layer_sizes: tuple = (100, 50),
    learning_rate: float = 0.001,
    max_pairs_per_class: int = 200,
    neg_ratio: float = 1.0,
    alpha: float = 0.0001,
    dropout_rate: float = 0.1,
    max_iter: int = 200,
    early_stopping: bool = True,
    random_seed: Optional[int] = 42,
    verbose: bool = True,
) -> SiameseModel:
    """
    Train a Siamese model for embedding similarity classification.

    Args:
        train_path: Path to training embeddings JSON file
        model_save_path: Path to save the trained model
        hidden_layer_sizes: Architecture of hidden layers
        learning_rate: Learning rate for gradient descent
        max_pairs_per_class: Maximum positive pairs per class
        neg_ratio: Ratio of negative to positive pairs
        alpha: L2 regularization strength
        dropout_rate: Dropout rate for regularization
        max_iter: Maximum training iterations
        early_stopping: Whether to use early stopping
        random_seed: Random seed for reproducibility
        verbose: Whether to print training progress

    Returns:
        Trained SiameseModel
    """
    if verbose:
        print("Loading training embeddings...")

    train_records = load_embeddings(train_path)

    if verbose:
        print(f"Loaded {len(train_records)} training records")
        classes = set(record["class"] for record in train_records)
        print(f"Found {len(classes)} classes: {sorted(classes)}")

    if verbose:
        print("Generating training pairs...")

    features, labels = generate_training_pairs(
        train_records,
        max_pairs_per_class=max_pairs_per_class,
        neg_ratio=neg_ratio,
        random_seed=random_seed,
        verbose=verbose,
    )

    if verbose:
        print(f"Generated {len(features)} pairs:")
        print(f"  - Positive pairs (same class): {np.sum(labels)}")
        print(f"  - Negative pairs (different class): {len(labels) - np.sum(labels)}")
        print(f"  - Feature shape: {features.shape}")

    if verbose:
        print("Training Siamese MLP...")

    model = SiameseModel(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate=learning_rate,
        alpha=alpha,
        dropout_rate=dropout_rate,
        max_iter=max_iter,
        early_stopping=early_stopping,
        random_state=random_seed,
    )
    model.fit(features, labels, verbose=verbose)

    if verbose:
        print("Evaluating model performance...")
        predictions = model.predict(features)
        accuracy = np.mean(predictions == labels)
        print(f"Training accuracy: {accuracy:.4f}")

    if verbose:
        print(f"Saving model to {model_save_path}...")

    model.save_model(model_save_path)

    if verbose:
        print("Training completed successfully!")

    return model


def compute_confidence(
    page_embedding: np.ndarray,
    reference_embeddings: List[np.ndarray],
    model: SiameseModel,
) -> float:
    """
    Compute confidence score for a page embedding against reference embeddings.

    Args:
        page_embedding: The embedding to score
        reference_embeddings: List of reference embeddings from the same predicted class
        model: Trained Siamese MLP model

    Returns:
        Confidence score between 0 and 100
    """
    if not reference_embeddings:
        return 0.0

    # Compute vector differences to all reference embeddings
    differences = []
    for ref_emb in reference_embeddings:
        diff = page_embedding - ref_emb
        differences.append(diff)

    # Predict probabilities
    probs = []
    for diff in differences:
        prob_same_class = model.predict_proba([diff])[0, 1]
        probs.append(prob_same_class)

    # Use maximum probability (most confident prediction)
    max_prob = max(probs)

    # Convert to confidence percentage
    return float(max_prob * 100.0)


def load_trained_model(model_path: str) -> SiameseModel:
    """Load a trained Siamese model from file."""
    model = SiameseModel()
    model.load_model(model_path)
    return model
