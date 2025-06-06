"""
Neural network models and preprocessing utilities.

This module contains the core ML components used in the Siamese confidence scoring:
- StandardScaler: Feature normalization
- MLPClassifier: Multi-layer perceptron neural network
"""

from typing import List

import numpy as np


class StandardScaler:
    """Numpy implementation of sklearn's StandardScaler."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """Compute the mean and std to be used for later scaling."""
        X = np.array(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero
        self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features using fitted parameters."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        X = np.array(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class MLPClassifier:
    """Numpy implementation of Multi-Layer Perceptron Classifier."""

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        learning_rate=0.001,
        max_iter=200,
        alpha=0.0001,
        batch_size=None,
        dropout_rate=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha  # L2 regularization strength
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

        # Model parameters
        self.weights_ = []
        self.biases_ = []
        self.n_layers_ = 0
        self.n_iter_ = 0
        self.is_fitted = False

        # Training history
        self.loss_curve_ = []
        self.validation_scores_ = []

    def _activation_func(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "logistic":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function."""
        if self.activation == "relu":
            return (x > 0).astype(float)
        elif self.activation == "tanh":
            return 1 - np.tanh(x) ** 2
        elif self.activation == "logistic":
            s = self._activation_func(x)
            return s * (1 - s)

    def _init_weights(self, layer_sizes: List[int]):
        """Initialize weights using He/Xavier initialization."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.weights_ = []
        self.biases_ = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            # He initialization for ReLU, Xavier for others
            if self.activation == "relu":
                std = np.sqrt(2.0 / fan_in)
            else:
                std = np.sqrt(2.0 / (fan_in + fan_out))

            weights = np.random.normal(0, std, (fan_in, fan_out))
            bias = np.zeros(fan_out)

            self.weights_.append(weights)
            self.biases_.append(bias)

    def _forward_pass(self, X: np.ndarray, training: bool = False):
        """Forward pass through the network."""
        activations = [X]
        z_values = []

        for i in range(len(self.weights_)):
            # Linear transformation
            z = np.dot(activations[-1], self.weights_[i]) + self.biases_[i]
            z_values.append(z)

            # Apply activation (except for output layer)
            if i < len(self.weights_) - 1:
                a = self._activation_func(z)

                # Apply dropout during training
                if training and self.dropout_rate > 0:
                    dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, a.shape)
                    a = a * dropout_mask / (1 - self.dropout_rate)

                activations.append(a)
            else:
                # Output layer (sigmoid for binary classification)
                a = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
                activations.append(a)

        return activations, z_values

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary cross-entropy loss with L2 regularization."""
        # Clip predictions to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Binary cross-entropy loss
        ce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        # L2 regularization
        l2_penalty = 0
        for weights in self.weights_:
            l2_penalty += np.sum(weights**2)
        l2_penalty *= self.alpha / 2

        return ce_loss + l2_penalty

    def _backward_pass(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: List[np.ndarray],
        z_values: List[np.ndarray],
    ):
        """Backward pass (backpropagation)."""
        m = X.shape[0]

        # Initialize gradients
        weight_grads = [np.zeros_like(w) for w in self.weights_]
        bias_grads = [np.zeros_like(b) for b in self.biases_]

        # Output layer error
        delta = activations[-1] - y.reshape(-1, 1)

        # Backpropagate errors
        for i in reversed(range(len(self.weights_))):
            # Compute gradients
            weight_grads[i] = (
                np.dot(activations[i].T, delta) / m + self.alpha * self.weights_[i]
            )
            bias_grads[i] = np.mean(delta, axis=0)

            # Propagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights_[i].T) * self._activation_derivative(
                    z_values[i - 1]
                )

        return weight_grads, bias_grads

    def _update_weights(
        self, weight_grads: List[np.ndarray], bias_grads: List[np.ndarray]
    ):
        """Update weights using gradient descent."""
        for i in range(len(self.weights_)):
            self.weights_[i] -= self.learning_rate * weight_grads[i]
            self.biases_[i] -= self.learning_rate * bias_grads[i]

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        """Train the MLP classifier."""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        n_samples, n_features = X.shape

        # Create layer sizes
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [1]
        self.n_layers_ = len(layer_sizes)

        # Initialize weights
        self._init_weights(layer_sizes)

        # Model is now ready to make predictions
        self.is_fitted = True

        # Split data for early stopping
        if self.early_stopping:
            n_val = int(n_samples * self.validation_fraction)
            indices = np.random.permutation(n_samples)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
        else:
            X_train, y_train = X, y
            X_val = y_val = None

        # Determine batch size
        if self.batch_size is None:
            self.batch_size = min(200, X_train.shape[0])

        # Training loop
        best_val_score = -np.inf
        no_improve_count = 0

        for epoch in range(self.max_iter):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Mini-batch training
            epoch_loss = 0
            n_batches = 0

            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train_shuffled[i : i + self.batch_size]
                batch_y = y_train_shuffled[i : i + self.batch_size]

                # Forward pass
                activations, z_values = self._forward_pass(batch_X, training=True)

                # Compute loss
                loss = self._compute_loss(batch_y, activations[-1])
                epoch_loss += loss
                n_batches += 1

                # Backward pass
                weight_grads, bias_grads = self._backward_pass(
                    batch_X, batch_y, activations, z_values
                )

                # Update weights
                self._update_weights(weight_grads, bias_grads)

            avg_loss = epoch_loss / n_batches
            self.loss_curve_.append(avg_loss)

            # Validation and early stopping
            if self.early_stopping and X_val is not None:
                val_predictions = self.predict_proba(X_val)[:, 1]
                val_score = -self._compute_loss(y_val, val_predictions)
                self.validation_scores_.append(val_score)

                if val_score > best_val_score:
                    best_val_score = val_score
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= self.n_iter_no_change:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if verbose and epoch % 20 == 0:
                train_acc = np.mean(
                    (self.predict_proba(X_train)[:, 1] > 0.5) == y_train
                )
                if self.early_stopping and X_val is not None:
                    val_acc = np.mean((self.predict_proba(X_val)[:, 1] > 0.5) == y_val)
                    print(
                        f"Epoch {epoch}: Loss={avg_loss:.4f}, "
                        f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch}: Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}"
                    )

        self.n_iter_ = epoch + 1
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X, dtype=float)
        activations, _ = self._forward_pass(X, training=False)
        proba_1 = activations[-1].flatten()
        proba_0 = 1 - proba_1
        return np.column_stack([proba_0, proba_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)
