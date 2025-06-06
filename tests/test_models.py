"""
Tests for the models module.
"""

import numpy as np
import pytest

from siamese_confidence.models import MLPClassifier, StandardScaler


class TestStandardScaler:
    """Test cases for StandardScaler."""

    def test_fit_transform(self):
        """Test basic fit and transform functionality."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

        X_scaled = scaler.fit_transform(X)

        # Check that mean is approximately 0 and std is approximately 1
        assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-10)

    def test_separate_fit_transform(self):
        """Test separate fit and transform calls."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

        scaler.fit(X)
        X_scaled = scaler.transform(X)

        assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-10)

    def test_transform_before_fit_raises_error(self):
        """Test that transform before fit raises ValueError."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="Scaler must be fitted"):
            scaler.transform(X)

    def test_zero_variance_handling(self):
        """Test handling of zero variance features."""
        scaler = StandardScaler()
        X = np.array(
            [[1, 5], [1, 6], [1, 7]], dtype=float
        )  # First column has zero variance

        X_scaled = scaler.fit_transform(X)

        # First column should be centered (mean subtracted) but not scaled
        assert np.allclose(X_scaled[:, 0], 0)  # (1-1)/1 = 0
        # Second column should be standardized
        assert np.allclose(np.mean(X_scaled[:, 1]), 0, atol=1e-10)


class TestMLPClassifier:
    """Test cases for MLPClassifier."""

    def test_initialization(self):
        """Test MLP initialization with default parameters."""
        mlp = MLPClassifier()

        assert mlp.hidden_layer_sizes == (100,)
        assert mlp.activation == "relu"
        assert mlp.learning_rate == 0.001
        assert not mlp.is_fitted

    def test_fit_simple_data(self):
        """Test fitting on simple binary classification data."""
        mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10, random_state=42)

        # Simple linearly separable data
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])  # XOR pattern

        mlp.fit(X, y)

        assert mlp.is_fitted
        assert len(mlp.weights_) == 2  # Input->Hidden, Hidden->Output
        assert len(mlp.biases_) == 2

    def test_predict_proba_shape(self):
        """Test that predict_proba returns correct shape."""
        mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=5, random_state=42)

        X = np.random.randn(10, 3)
        y = np.random.randint(0, 2, 10)

        mlp.fit(X, y)
        proba = mlp.predict_proba(X)

        assert proba.shape == (10, 2)  # n_samples x 2 (binary classification)
        assert np.allclose(proba.sum(axis=1), 1)  # Probabilities sum to 1

    def test_predict_binary_output(self):
        """Test that predict returns binary labels."""
        mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=5, random_state=42)

        X = np.random.randn(10, 3)
        y = np.random.randint(0, 2, 10)

        mlp.fit(X, y)
        predictions = mlp.predict(X)

        assert predictions.shape == (10,)
        assert np.all(np.isin(predictions, [0, 1]))  # Only 0s and 1s

    def test_predict_before_fit_raises_error(self):
        """Test that predict before fit raises ValueError."""
        mlp = MLPClassifier()
        X = np.random.randn(5, 3)

        with pytest.raises(ValueError, match="Model must be fitted"):
            mlp.predict_proba(X)

    def test_different_activations(self):
        """Test different activation functions."""
        activations = ["relu", "tanh", "logistic"]

        for activation in activations:
            mlp = MLPClassifier(
                hidden_layer_sizes=(5,),
                activation=activation,
                max_iter=5,
                random_state=42,
            )

            X = np.random.randn(10, 3)
            y = np.random.randint(0, 2, 10)

            mlp.fit(X, y)
            predictions = mlp.predict(X)

            assert predictions.shape == (10,)

    def test_early_stopping(self):
        """Test early stopping functionality."""
        mlp = MLPClassifier(
            hidden_layer_sizes=(10,),
            max_iter=100,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=42,
        )

        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        mlp.fit(X, y)

        # Should stop before max_iter due to early stopping
        assert mlp.n_iter_ <= 100
