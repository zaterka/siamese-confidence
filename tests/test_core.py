"""
Tests for the core module.
"""

import os
import tempfile

import numpy as np
import pytest

from siamese_confidence.core import (
    SiameseModel,
    compute_confidence,
    load_trained_model,
)


class TestSiameseModel:
    """Test cases for SiameseModel."""

    def test_initialization(self):
        """Test SiameseModel initialization."""
        model = SiameseModel()

        assert not model.is_fitted
        assert model.scaler is not None
        assert model.mlp is not None

    def test_fit_and_predict(self):
        """Test basic fit and predict functionality."""
        model = SiameseModel(hidden_layer_sizes=(10,), max_iter=5, random_state=42)

        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)

        model.fit(X, y)

        assert model.is_fitted

        # Test predictions
        proba = model.predict_proba(X)
        predictions = model.predict(X)

        assert proba.shape == (20, 2)
        assert predictions.shape == (20,)
        assert np.all(np.isin(predictions, [0, 1]))

    def test_save_and_load_model(self):
        """Test model persistence."""
        model = SiameseModel(hidden_layer_sizes=(5,), max_iter=3, random_state=42)

        X = np.random.randn(10, 3)
        y = np.random.randint(0, 2, 10)

        model.fit(X, y)

        # Save model to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            model.save_model(temp_path)

            # Load model
            loaded_model = load_trained_model(temp_path)

            assert loaded_model.is_fitted

            # Test that loaded model gives same predictions
            original_proba = model.predict_proba(X)
            loaded_proba = loaded_model.predict_proba(X)

            assert np.allclose(original_proba, loaded_proba)

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_predict_before_fit_raises_error(self):
        """Test that predict before fit raises ValueError."""
        model = SiameseModel()
        X = np.random.randn(5, 3)

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_proba(X)

    def test_save_before_fit_raises_error(self):
        """Test that save before fit raises ValueError."""
        model = SiameseModel()

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.save_model("dummy_path.json")


class TestComputeConfidence:
    """Test cases for compute_confidence function."""

    def setup_method(self):
        """Set up a trained model for testing."""
        self.model = SiameseModel(hidden_layer_sizes=(5,), max_iter=5, random_state=42)

        # Train on some mock data
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 2, 20)
        self.model.fit(X, y)

    def test_compute_confidence_basic(self):
        """Test basic confidence computation."""
        page_embedding = np.random.randn(10)
        reference_embeddings = [np.random.randn(10) for _ in range(3)]

        confidence = compute_confidence(
            page_embedding, reference_embeddings, self.model
        )

        assert isinstance(confidence, float)
        assert 0 <= confidence <= 100

    def test_compute_confidence_empty_references(self):
        """Test confidence computation with empty references."""
        page_embedding = np.random.randn(10)
        reference_embeddings = []

        confidence = compute_confidence(
            page_embedding, reference_embeddings, self.model
        )

        assert confidence == 0.0

    def test_compute_confidence_single_reference(self):
        """Test confidence computation with single reference."""
        page_embedding = np.random.randn(10)
        reference_embeddings = [np.random.randn(10)]

        confidence = compute_confidence(
            page_embedding, reference_embeddings, self.model
        )

        assert isinstance(confidence, float)
        assert 0 <= confidence <= 100

    def test_compute_confidence_multiple_references(self):
        """Test confidence computation with multiple references."""
        page_embedding = np.random.randn(10)
        reference_embeddings = [np.random.randn(10) for _ in range(5)]

        confidence = compute_confidence(
            page_embedding, reference_embeddings, self.model
        )

        assert isinstance(confidence, float)
        assert 0 <= confidence <= 100

    def test_compute_confidence_deterministic(self):
        """Test that confidence computation is deterministic."""
        np.random.seed(42)
        page_embedding = np.random.randn(10)
        reference_embeddings = [np.random.randn(10) for _ in range(3)]

        confidence1 = compute_confidence(
            page_embedding, reference_embeddings, self.model
        )

        confidence2 = compute_confidence(
            page_embedding, reference_embeddings, self.model
        )

        assert confidence1 == confidence2
