"""
Chaos Analysis Library
=====================

A comprehensive library for nonlinear time series analysis including
Hurst exponent, Lyapunov exponents, entropy measures, dimensionality analysis,
and enhanced echo state networks.
"""

from .chaotic_measures import (embedding_dimension, fourier_harmonic_count,
                               hurst_trajectory, ks_entropy,
                               max_lyapunov_exponent, moving_least_squares,
                               noise_factor, normalize_01)
from .enhanced_esn_fan import EnhancedESN_FAN

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # Chaos analysis functions
    "moving_least_squares",
    "hurst_trajectory",
    "normalize_01",
    "noise_factor",
    "embedding_dimension",
    "max_lyapunov_exponent",
    "ks_entropy",
    "fourier_harmonic_count",
    # ESN models
    "EnhancedESN_FAN",
]
