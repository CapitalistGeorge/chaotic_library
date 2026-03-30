"""
Enhanced Echo State Network with Fourier Analysis Network (FAN) features.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class EnhancedESN_FAN:
    """
    Enhanced Echo State Network with Fourier Analysis Network features.

    This model combines traditional ESN dynamics with polynomial features
    and explicit Fourier series terms for improved time series modeling.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features per timestep.
    reservoir_size : int, default=100
        Number of neurons in the reservoir.
    spectral_radius : float, default=0.9
        Spectral radius of the reservoir weight matrix.
    sparsity : float, default=0.1
        Effective fraction of non-zero reservoir connections.
    ridge_alpha : float, default=1.0
        Regularization strength for ridge regression.
    leaking_rate : float, default=1.0
        Leaking rate for reservoir state updates (0-1).
    poly_order : int, default=2
        Order of polynomial features.
    fan_terms : int, default=5
        Number of Fourier terms (k in sin(2πkx), cos(2πkx)).
    random_state : int, default=42
        Random seed for reproducibility.
    clip_value : float, default=3.0
        Maximum absolute value for clipping scaled inputs.
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 100,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        ridge_alpha: float = 1.0,
        leaking_rate: float = 1.0,
        poly_order: int = 2,
        fan_terms: int = 5,
        random_state: int = 42,
        clip_value: float = 3.0,
    ) -> None:
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leaking_rate = leaking_rate
        self.poly_order = poly_order
        self.fan_terms = fan_terms
        self.random_state = random_state
        self.clip_value = clip_value
        self.ridge_alpha = ridge_alpha

        self._validate_parameters()

        self.rng = np.random.default_rng(self.random_state)

        self.poly = PolynomialFeatures(degree=self.poly_order, include_bias=False)
        self.ridge = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
        self.scaler = StandardScaler()
        self.input_scaler = StandardScaler()

        self.Win: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None

        self._is_fitted = False
        self.output_dim_: Optional[int] = None
        self.last_state_: Optional[np.ndarray] = None

    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.reservoir_size <= 0:
            raise ValueError("reservoir_size must be positive")
        if not 0 <= self.sparsity <= 1:
            raise ValueError("sparsity must be between 0 and 1")
        if not 0 <= self.leaking_rate <= 1:
            raise ValueError("leaking_rate must be between 0 and 1")
        if self.spectral_radius <= 0:
            raise ValueError("spectral_radius must be positive")
        if self.fan_terms <= 0:
            raise ValueError("fan_terms must be positive")
        if self.poly_order < 1:
            raise ValueError("poly_order must be at least 1")
        if self.clip_value <= 0:
            raise ValueError("clip_value must be positive")
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")

    def _validate_X(self, X: np.ndarray, *, check_min_samples: bool = False) -> np.ndarray:
        """Validate input matrix X."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")

        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected X with {self.input_dim} features, got {X.shape[1]}"
            )

        if check_min_samples and X.shape[0] < 10:
            raise ValueError("Insufficient training data")

        return X

    def _validate_y(self, y: np.ndarray, n_samples: int) -> np.ndarray:
        """Validate target array y."""
        y = np.asarray(y, dtype=float)

        if y.ndim not in (1, 2):
            raise ValueError("y must be 1-dimensional or 2-dimensional")

        if y.shape[0] != n_samples:
            raise ValueError("X and y must have the same number of samples")

        return y

    def _check_is_fitted(self) -> None:
        """Raise if model is not fitted."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

    def initialize_weights(self, X: np.ndarray) -> None:
        """
        Initialize input and reservoir weight matrices.

        Uses input-distribution-aware initialization for input weights
        and creates a sparse reservoir matrix with controlled spectral radius.
        """
        X = self._validate_X(X)

        inp_mean = np.mean(X, axis=0)
        inp_std = np.std(X, axis=0)
        mu = float(np.mean(inp_mean))
        sigma = float(np.mean(inp_std))

        # Avoid degenerate initialization when inputs are nearly constant
        sigma = max(sigma, 1e-6)

        self.Win = self.rng.normal(
            loc=mu,
            scale=sigma,
            size=(self.reservoir_size, self.input_dim + 1),
        )

        W = self.rng.random((self.reservoir_size, self.reservoir_size)) - 0.5

        # Keep approximately `sparsity` fraction of active connections
        mask = self.rng.random(W.shape) > self.sparsity
        W[mask] = 0.0

        try:
            eigenvalues = np.linalg.eigvals(W)
            max_eigenvalue = np.max(np.abs(eigenvalues))

            if max_eigenvalue > 1e-12:
                W *= self.spectral_radius / max_eigenvalue
            else:
                warnings.warn(
                    "Reservoir matrix has near-zero eigenvalues. "
                    "Spectral radius scaling may be ineffective."
                )
        except (np.linalg.LinAlgError, ValueError) as exc:
            warnings.warn(f"Eigenvalue computation failed: {exc}")

        self.W = W

    def _update_reservoir_state(
        self,
        state: np.ndarray,
        input_vector: np.ndarray,
    ) -> np.ndarray:
        """Update reservoir state using ESN dynamics with leaking rate."""
        if self.Win is None or self.W is None:
            raise ValueError("Weights are not initialized. Call fit() first.")

        input_with_bias = np.concatenate(([1.0], input_vector))
        pre_activation = self.Win @ input_with_bias + self.W @ state
        new_state = np.tanh(pre_activation)

        return (1.0 - self.leaking_rate) * state + self.leaking_rate * new_state

    def _compute_states(
        self,
        X: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute reservoir states for a full input sequence.

        Returns
        -------
        states : ndarray of shape (n_timesteps, reservoir_size)
        last_state : ndarray of shape (reservoir_size,)
        """
        X = self._validate_X(X)

        n_timesteps = X.shape[0]
        states = np.zeros((n_timesteps, self.reservoir_size), dtype=float)

        if initial_state is None:
            current_state = np.zeros(self.reservoir_size, dtype=float)
        else:
            current_state = np.asarray(initial_state, dtype=float).copy()
            if current_state.shape != (self.reservoir_size,):
                raise ValueError(
                    "initial_state must have shape "
                    f"({self.reservoir_size},), got {current_state.shape}"
                )

        for t in range(n_timesteps):
            current_state = self._update_reservoir_state(current_state, X[t])
            states[t] = current_state

        return states, current_state

    def _scale_input(self, X: np.ndarray, *, fit: bool) -> np.ndarray:
        """Scale raw inputs and clip extreme standardized values."""
        if fit:
            X_scaled = self.input_scaler.fit_transform(X)
        else:
            X_scaled = self.input_scaler.transform(X)

        return np.clip(X_scaled, -self.clip_value, self.clip_value)

    def _compute_fourier_features(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Fourier Analysis Network features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, input_dim)

        Returns
        -------
        ndarray of shape (n_samples, 2 * fan_terms * input_dim)
        """
        features = []

        for k in range(1, self.fan_terms + 1):
            features.append(np.sin(2.0 * np.pi * k * X))
            features.append(np.cos(2.0 * np.pi * k * X))

        return np.hstack(features)

    def _build_feature_matrix(
        self,
        X: np.ndarray,
        states: np.ndarray,
        *,
        fit_transformers: bool,
    ) -> np.ndarray:
        """
        Build the full feature matrix using a single, consistent pipeline.

        Pipeline:
        1) scale raw X with input_scaler
        2) clip scaled inputs
        3) build polynomial features
        4) build Fourier features
        5) concatenate with reservoir states
        6) scale combined feature matrix with scaler
        """
        X_scaled = self._scale_input(X, fit=fit_transformers)

        if fit_transformers:
            poly_features = self.poly.fit_transform(X_scaled)
        else:
            poly_features = self.poly.transform(X_scaled)

        fourier_features = self._compute_fourier_features(X_scaled)
        combined_features = np.hstack([states, poly_features, fourier_features])

        if fit_transformers:
            self.scaler.fit(combined_features)
            self.scaler.scale_[self.scaler.scale_ == 0.0] = 1.0

        return self.scaler.transform(combined_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnhancedESN_FAN":
        """
        Fit the Enhanced ESN model to training data.

        Parameters
        ----------
        X : ndarray of shape (n_timesteps, input_dim)
            Input time series data.
        y : ndarray of shape (n_timesteps,) or (n_timesteps, output_dim)
            Target values.

        Returns
        -------
        self : EnhancedESN_FAN
            Fitted model.
        """
        X = self._validate_X(X, check_min_samples=True)
        y = self._validate_y(y, n_samples=X.shape[0])

        self.initialize_weights(X)

        states, last_state = self._compute_states(X)
        scaled_features = self._build_feature_matrix(
            X,
            states,
            fit_transformers=True,
        )

        self.ridge.fit(scaled_features, y)

        self.output_dim_ = 1 if y.ndim == 1 else y.shape[1]
        self.last_state_ = last_state.copy()
        self._is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        generative_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate predictions using the fitted model.

        Parameters
        ----------
        X : ndarray of shape (n_timesteps, input_dim)
            Input data for prediction.
        generative_steps : int, optional
            If provided, perform recursive/generative forecasting for this many steps.
            If None, perform teacher-forcing / open-loop prediction.

        Returns
        -------
        ndarray
            Predictions.
        """
        self._check_is_fitted()
        X = self._validate_X(X)

        if generative_steps is None:
            return self._predict_open_loop(X)

        if generative_steps <= 0:
            raise ValueError("generative_steps must be positive")

        return self._predict_generative(X, generative_steps)

    def _predict_open_loop(self, X: np.ndarray) -> np.ndarray:
        """One-step prediction (teacher forcing)."""
        states, _ = self._compute_states(X)
        scaled_features = self._build_feature_matrix(
            X,
            states,
            fit_transformers=False,
        )
        return self.ridge.predict(scaled_features)

    def _predict_generative(self, X: np.ndarray, steps: int) -> np.ndarray:
        """
        Generative forecasting (recursive prediction).

        Notes
        -----
        This mode requires output_dim == input_dim because each prediction is
        fed back as the next input.
        """
        if self.output_dim_ != self.input_dim:
            raise ValueError(
                "Generative forecasting requires output_dim == input_dim "
                f"(got output_dim={self.output_dim_}, input_dim={self.input_dim})"
            )

        # Use the provided seed sequence to move reservoir state to the correct context
        _, current_state = self._compute_states(X)
        current_input = X[-1].astype(float).copy()

        predictions = []

        for _ in range(steps):
            current_state = self._update_reservoir_state(current_state, current_input)

            scaled_feature_vector = self._build_feature_matrix(
                current_input.reshape(1, -1),
                current_state.reshape(1, -1),
                fit_transformers=False,
            )

            next_pred = self.ridge.predict(scaled_feature_vector)
            next_input = np.asarray(next_pred, dtype=float).reshape(-1)

            predictions.append(next_input.copy())
            current_input = next_input

        return np.vstack(predictions)

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._is_fitted

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            "input_dim": self.input_dim,
            "reservoir_size": self.reservoir_size,
            "spectral_radius": self.spectral_radius,
            "sparsity": self.sparsity,
            "ridge_alpha": self.ridge_alpha,
            "leaking_rate": self.leaking_rate,
            "poly_order": self.poly_order,
            "fan_terms": self.fan_terms,
            "random_state": self.random_state,
            "clip_value": self.clip_value,
        }

    def set_params(self, **params) -> "EnhancedESN_FAN":
        """
        Set parameters for this estimator.

        Structural parameter changes invalidate the fitted state.
        """
        valid_params = {
            "input_dim",
            "reservoir_size",
            "spectral_radius",
            "sparsity",
            "ridge_alpha",
            "leaking_rate",
            "poly_order",
            "fan_terms",
            "random_state",
            "clip_value",
        }

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)

        self._validate_parameters()

        # Recreate dependent objects
        self.rng = np.random.default_rng(self.random_state)
        self.poly = PolynomialFeatures(degree=self.poly_order, include_bias=False)
        self.ridge = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
        self.scaler = StandardScaler()
        self.input_scaler = StandardScaler()

        self.Win = None
        self.W = None
        self.last_state_ = None
        self.output_dim_ = None
        self._is_fitted = False

        return self