"""
Enhanced Echo State Network with Fourier Analysis Network (FAN) features.
"""

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
        Dimensionality of input features per timestep
    reservoir_size : int, default=100
        Number of neurons in the reservoir
    spectral_radius : float, default=0.9
        Spectral radius of the reservoir weight matrix
    sparsity : float, default=0.1
        Sparsity proportion of reservoir connections (0-1)
    ridge_alpha : float, default=1.0
        Regularization strength for ridge regression
    leaking_rate : float, default=1.0
        Leaking rate for reservoir state updates (0-1)
    poly_order : int, default=2
        Order of polynomial features
    fan_terms : int, default=5
        Number of Fourier terms (k in sin(2πkx), cos(2πkx))
    random_state : int, default=42
        Random seed for reproducibility
    clip_value : float, default=3.0
        Maximum absolute value for clipping scaled inputs

    Attributes
    ----------
    Win : ndarray
        Input weight matrix of shape (reservoir_size, input_dim + 1)
    W : ndarray
        Reservoir weight matrix of shape (reservoir_size, reservoir_size)
    ridge : Ridge
        Ridge regression model for readout layer
    scaler : StandardScaler
        Scaler for combined features
    input_scaler : StandardScaler
        Scaler for input data
    poly : PolynomialFeatures
        Polynomial feature generator

    Examples
    --------
    >>> from chaotic_library.enhanced_esn_fan import EnhancedESN_FAN
    >>> import numpy as np
    >>>
    >>> # Generate sample data
    >>> X = np.random.randn(1000, 5)
    >>> y = np.random.randn(1000)
    >>>
    >>> # Initialize and fit model
    >>> esn = EnhancedESN_FAN(input_dim=5, reservoir_size=200)
    >>> esn.fit(X, y)
    >>>
    >>> # Predict
    >>> predictions = esn.predict(X[:10])
    >>>
    >>> # Generative forecasting
    >>> future_predictions = esn.predict(X[-1:], generative_steps=50)
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
    ):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leaking_rate = leaking_rate
        self.poly_order = poly_order
        self.fan_terms = fan_terms
        self.random_state = random_state
        self.clip_value = clip_value

        # Validate parameters
        self._validate_parameters()

        # Set random seed
        np.random.seed(self.random_state)

        # Initialize components
        self.poly = PolynomialFeatures(degree=self.poly_order, include_bias=False)

        self.ridge = Ridge(alpha=ridge_alpha, random_state=random_state)
        self.scaler = StandardScaler()
        self.input_scaler = StandardScaler()

        # Weight matrices (initialized during fit)
        self.Win = None
        self.W = None

        # Model state
        self._is_fitted = False

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

    def initialize_weights(self, X: np.ndarray) -> None:
        """
        Initialize input and reservoir weight matrices.

        Uses input-distribution aware initialization for input weights
        and creates a sparse reservoir matrix with controlled spectral radius.

        Parameters
        ----------
        X : ndarray of shape (n_samples, input_dim)
            Input data for weight initialization
        """
        n_samples, input_dim = X.shape

        if input_dim != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, " f"but got {input_dim}"
            )

        # Input-distribution aware Win initialization
        inp_mean = np.mean(X, axis=0)
        inp_std = np.std(X, axis=0)
        mu, sigma = np.mean(inp_mean), np.mean(inp_std)

        # Input weights with bias term (first column is for bias)
        self.Win = np.random.normal(mu, sigma, (self.reservoir_size, input_dim + 1))

        # Sparse reservoir weights
        W = np.random.rand(self.reservoir_size, self.reservoir_size) - 0.5

        # Apply sparsity
        mask = np.random.rand(*W.shape) > self.sparsity
        W[mask] = 0.0

        # Scale by spectral radius
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
        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f"Eigenvalue computation failed: {e}")

        self.W = W

    def _update_reservoir_state(
        self, state: np.ndarray, input_vector: np.ndarray
    ) -> np.ndarray:
        """
        Update reservoir state using ESN dynamics with leaking rate.

        Parameters
        ----------
        state : ndarray of shape (reservoir_size,)
            Current reservoir state
        input_vector : ndarray of shape (input_dim,)
            Input vector at current time step

        Returns
        -------
        ndarray of shape (reservoir_size,)
            Updated reservoir state
        """
        # Add bias term to input
        input_with_bias = np.concatenate(([1.0], input_vector))

        # Reservoir update
        pre_activation = self.Win @ input_with_bias + self.W @ state

        # Nonlinear activation
        new_state = np.tanh(pre_activation)

        # Apply leaking rate
        updated_state = (1 - self.leaking_rate) * state + self.leaking_rate * new_state

        return updated_state

    def _compute_fourier_features(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Fourier Analysis Network features.

        Generates sin(2πkx) and cos(2πkx) terms for k=1 to fan_terms.

        Parameters
        ----------
        X : ndarray of shape (n_samples, input_dim)
            Input data

        Returns
        -------
        ndarray of shape (n_samples, 2 * fan_terms * input_dim)
            Fourier features (sin and cos terms)
        """
        features = []

        for k in range(1, self.fan_terms + 1):
            # sin(2πkX) and cos(2πkX) terms
            features.append(np.sin(2 * np.pi * k * X))
            features.append(np.cos(2 * np.pi * k * X))

        return np.hstack(features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnhancedESN_FAN":
        """
        Fit the Enhanced ESN model to training data.

        Parameters
        ----------
        X : ndarray of shape (n_timesteps, input_dim)
            Input time series data
        y : ndarray of shape (n_timesteps,) or (n_timesteps, output_dim)
            Target values

        Returns
        -------
        self : EnhancedESN_FAN
            Fitted model

        Raises
        ------
        ValueError
            If input dimensions are inconsistent or data is insufficient
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if X.shape[0] < 10:
            raise ValueError("Insufficient training data")

        # Initialize reservoir weights
        self.initialize_weights(X)

        # Collect reservoir states through time
        n_timesteps = X.shape[0]
        states = np.zeros((n_timesteps, self.reservoir_size))
        current_state = np.zeros(self.reservoir_size)

        for t in range(n_timesteps):
            current_state = self._update_reservoir_state(current_state, X[t])
            states[t] = current_state

        # Prepare features
        X_scaled = self.input_scaler.fit_transform(X)

        # Polynomial features
        poly_features = self.poly.fit_transform(X_scaled)

        # Fourier features
        fourier_features = self._compute_fourier_features(X_scaled)

        # Combine all features
        combined_features = np.hstack([states, poly_features, fourier_features])

        # Scale combined features with zero-division protection
        self.scaler.fit(combined_features)
        self.scaler.scale_[self.scaler.scale_ == 0.0] = 1.0
        scaled_features = self.scaler.transform(combined_features)

        # Fit readout layer
        self.ridge.fit(scaled_features, y)

        self._is_fitted = True
        return self

    def predict(
        self, X: np.ndarray, generative_steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate predictions using the fitted model.

        Parameters
        ----------
        X : ndarray of shape (n_timesteps, input_dim)
            Input data for prediction
        generative_steps : int, optional
            If provided, perform generative forecasting for this many steps.
            If None, perform one-step prediction (teacher forcing).

        Returns
        -------
        ndarray
            Predictions. Shape depends on generative_steps:
            - If generative_steps is None: (n_timesteps, output_dim)
            - If generative_steps is provided: (generative_steps, output_dim)

        Raises
        ------
        ValueError
            If model is not fitted or input dimensions are invalid
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")

        if generative_steps is None:
            return self._predict_open_loop(X)
        else:
            if generative_steps <= 0:
                raise ValueError("generative_steps must be positive")
            return self._predict_generative(X, generative_steps)

    def _predict_open_loop(self, X: np.ndarray) -> np.ndarray:
        """One-step prediction (teacher forcing)."""
        n_timesteps = X.shape[0]
        states = np.zeros((n_timesteps, self.reservoir_size))
        current_state = np.zeros(self.reservoir_size)

        # Update reservoir states
        for t in range(n_timesteps):
            current_state = self._update_reservoir_state(current_state, X[t])
            states[t] = current_state

        # Prepare features (using original X for polynomial/Fourier)
        poly_features = self.poly.transform(X)
        fourier_features = self._compute_fourier_features(X)
        combined_features = np.hstack([states, poly_features, fourier_features])

        # Scale and predict
        scaled_features = self.scaler.transform(combined_features)
        return self.ridge.predict(scaled_features)

    def _predict_generative(self, X: np.ndarray, steps: int) -> np.ndarray:
        """Generative forecasting (recursive prediction)."""
        # Start from last input
        current_input = X[-1].copy()
        current_state = np.zeros(self.reservoir_size)
        predictions = []

        for _ in range(steps):
            # Update reservoir state
            current_state = self._update_reservoir_state(current_state, current_input)

            # Prepare features for prediction
            input_scaled = self.input_scaler.transform(current_input.reshape(1, -1))

            # Clip to prevent extreme values
            input_scaled = np.clip(input_scaled, -self.clip_value, self.clip_value)

            poly_features = self.poly.transform(input_scaled)
            fourier_features = self._compute_fourier_features(input_scaled)

            # Combine features
            feature_vector = np.hstack(
                [current_state, poly_features.ravel(), fourier_features.ravel()]
            ).reshape(1, -1)

            # Scale and predict
            scaled_features = self.scaler.transform(feature_vector)
            next_input = self.ridge.predict(scaled_features)

            predictions.append(next_input.ravel())
            current_input = next_input  # Feedback for next step

        return np.vstack(predictions)

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this model."""
        return {
            "input_dim": self.input_dim,
            "reservoir_size": self.reservoir_size,
            "spectral_radius": self.spectral_radius,
            "sparsity": self.sparsity,
            "ridge_alpha": self.ridge.alpha,
            "leaking_rate": self.leaking_rate,
            "poly_order": self.poly_order,
            "fan_terms": self.fan_terms,
            "random_state": self.random_state,
            "clip_value": self.clip_value,
        }

    def set_params(self, **params) -> "EnhancedESN_FAN":
        """Set the parameters of this model."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key == "ridge_alpha":
                self.ridge.alpha = value
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self
