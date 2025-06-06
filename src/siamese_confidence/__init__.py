"""
Siamese Confidence Scoring

A pure NumPy implementation of neural network-based Siamese confidence
scoring for embedding similarity classification.

This library provides a robust, regularized approach to computing confidence
scores for document classification tasks using high-dimensional vector embeddings.
"""

from .core import (
    SiameseModel,
    compute_confidence,
    load_trained_model,
    train_siamese_model,
)
from .models import MLPClassifier, StandardScaler
from .utils import generate_training_pairs, load_embeddings

__version__ = "0.1.0"
__author__ = "Pedro Zaterka"
__email__ = "pedrozaterka@gmail.com"

__all__ = [
    # Core functionality
    "SiameseModel",
    "compute_confidence",
    "load_trained_model",
    "train_siamese_model",
    # Model components
    "MLPClassifier",
    "StandardScaler",
    # Utilities
    "generate_training_pairs",
    "load_embeddings",
    # Version info
    "__version__",
]


def get_version() -> str:
    """Get the current version of the package."""
    return __version__


def get_info() -> dict:
    """Get package information."""
    return {
        "name": "siamese-confidence",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "Neural network-based Siamese confidence scoring for embeddings",
    }
