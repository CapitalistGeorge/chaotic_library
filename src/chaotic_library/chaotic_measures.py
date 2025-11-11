"""
Chaos Analysis Library
A collection of tools for nonlinear time series analysis.
"""

import warnings
from itertools import combinations
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.fft import fft
from scipy.linalg import hankel
from scipy.spatial import cKDTree


def moving_least_squares(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Moving Least Squares regression for single variable.

    Parameters
    ----------
    x : array-like
        Independent variable values
    y : array-like
        Dependent variable values

    Returns
    -------
    tuple
        (slope, intercept) of the regression line
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    n = len(x)
    sum_x, sum_y = np.sum(x), np.sum(y)
    sum_x2 = np.sum(x * x)
    sum_xy = np.sum(x * y)

    denominator = n * sum_x2 - sum_x * sum_x
    if abs(denominator) < 1e-12:
        warnings.warn("Denominator close to zero in MLS calculation")
        return 0.0, np.mean(y)

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    return slope, intercept


def hurst_trajectory(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Calculate Hurst exponent using R/S analysis.

    Parameters
    ----------
    series : array-like
        Time series data

    Returns
    -------
    tuple
        (log_tau, rs_trajectory, hurst_exponent, memory_point)
    """
    series = np.array(series, dtype=float)
    if len(series) < 10:
        raise ValueError("Series too short for Hurst analysis")

    # Normalize series
    series_norm, _, _ = normalize_01(series)

    # Calculate R/S trajectory
    tau_values = np.arange(3, len(series_norm))
    rs_values = []

    for tau in tau_values:
        segment = series_norm[:tau]
        mean_val = np.mean(segment)
        std_val = np.std(segment, ddof=1)

        if std_val < 1e-12:
            rs_values.append(0.0)
            continue

        cumulative_deviation = np.cumsum(segment - mean_val)
        range_val = np.max(cumulative_deviation) - np.min(cumulative_deviation)
        rs_values.append(np.log(range_val / std_val) if range_val > 0 else 0.0)

    rs_values = np.array(rs_values)
    log_tau = np.concatenate([[0.0], np.log(tau_values[1:] / 2)])

    # Calculate Hurst exponent from initial segment
    initial_length = max(1, len(log_tau) // 50)
    hurst_exp, _ = moving_least_squares(
        log_tau[:initial_length], rs_values[:initial_length]
    )

    # Find memory point (first negative difference)
    differences = np.diff(rs_values)
    memory_points = np.where(differences < 0.0)[0]
    memory_point = memory_points[0] if len(memory_points) > 0 else 0

    return log_tau, rs_values, hurst_exp, memory_point


def normalize_01(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalize data to [0, 1] range.

    Parameters
    ----------
    data : array-like
        Input data

    Returns
    -------
    tuple
        (normalized_data, min_value, scale_factor)
    """
    data = np.array(data, dtype=float)
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    scale = max_val - min_val

    if scale < 1e-12:
        warnings.warn("Constant or near-constant input data")
        return np.zeros_like(data), min_val, 1.0

    normalized = (data - min_val) / scale
    return normalized, min_val, scale


def noise_factor(data: np.ndarray, axis: int = 0, ddof: int = 1) -> float:
    """
    Calculate noise factor (1 - std(diff)/std(data)).

    Parameters
    ----------
    data : array-like
        Input data
    axis : int
        Axis along which to calculate
    ddof : int
        Delta degrees of freedom for std calculation

    Returns
    -------
    float
        Noise factor (higher value = less noise)
    """
    data_norm, _, _ = normalize_01(data)

    if isinstance(data_norm, pd.Series):
        differences = data_norm.diff().dropna().abs()
    else:
        differences = np.abs(np.diff(data_norm, axis=axis))

    std_diff = np.std(differences, axis=axis, ddof=ddof)
    std_data = np.std(data_norm, axis=axis, ddof=ddof)

    if std_data < 1e-12:
        return 0.0

    return 1.0 - float(std_diff / std_data)


def embedding_dimension(
    time_series: np.ndarray, epsilon: float = 0.1, max_dim: Optional[int] = None
) -> Tuple[int, float, float]:
    """
    Estimate embedding dimension using correlation integral.

    Parameters
    ----------
    time_series : array-like
        Input time series
    epsilon : float
        Convergence threshold
    max_dim : int, optional
        Maximum embedding dimension to test

    Returns
    -------
    tuple
        (embedding_dim, correlation_dim, entropy_estimate)
    """
    series_norm, _, _ = normalize_01(time_series)
    n = len(series_norm)

    if max_dim is None:
        max_dim = n // 2

    hankel_matrix = hankel(series_norm)
    correlation_sum = [1.0]
    prev_dimension = 0.0

    for k in range(2, max_dim + 1):
        total_correlation = sum(correlation_sum)
        embedding = hankel_matrix[: n - k, :k]

        n_points = n - k
        if n_points < 2:
            break

        distances = np.zeros((n_points, n_points))

        for i, j in combinations(range(n_points), 2):
            norm_val = np.linalg.norm(embedding[i] - embedding[j])
            distances[i, j] = norm_val
            distances[j, i] = norm_val

        non_zero_distances = distances[distances != 0]
        if len(non_zero_distances) == 0:
            break

        distance_range = np.linspace(
            np.min(non_zero_distances), np.max(non_zero_distances), num=20
        )

        correlation_values = []
        log_correlation = []

        for dist_threshold in distance_range:
            correlation = (
                np.sum(np.heaviside(dist_threshold - distances - np.eye(n_points), 0))
                // 2
            )
            normalized_correlation = correlation / (n_points**2)
            correlation_values.append(normalized_correlation)
            log_correlation.append(
                np.log(normalized_correlation) if normalized_correlation > 0 else 0
            )

        # Calculate correlation dimension
        if len(log_correlation) > 1 and len(distance_range) > 1:
            current_dimension = (log_correlation[1] - log_correlation[0]) / (
                np.log(distance_range[1]) - np.log(distance_range[0])
            )
        else:
            current_dimension = prev_dimension

        if abs(current_dimension - prev_dimension) > epsilon:
            prev_dimension = current_dimension
        else:
            k -= 1
            break

    embedding_dim = max(2, k)
    entropy_est = (
        sum(correlation_values) / total_correlation if total_correlation > 0 else 0
    )

    return embedding_dim, prev_dimension, entropy_est


def max_lyapunov_exponent(
    x: np.ndarray, emb_dim: int = 10, lag: int = 1, fit_len: int = 20
) -> float:
    """
    Estimate maximal Lyapunov exponent using Rosenstein's algorithm.

    Parameters
    ----------
    x : array-like
        1-dimensional time series
    emb_dim : int
        Embedding dimension
    lag : int
        Time delay for embedding
    fit_len : int
        Number of steps for divergence tracking

    Returns
    -------
    float
        Maximal Lyapunov exponent estimate
    """
    x = np.asarray(x, dtype=float)

    if len(x) < emb_dim * lag + fit_len:
        raise ValueError(
            f"Time series too short. Need at least {emb_dim * lag + fit_len} points, "
            f"got {len(x)}"
        )

    # Phase-space reconstruction
    n_points = len(x) - (emb_dim - 1) * lag
    embedding = np.column_stack(
        [x[i * lag : i * lag + n_points] for i in range(emb_dim)]
    )

    # Find nearest neighbors
    tree = cKDTree(embedding)
    distances, indices = tree.query(embedding, k=2)
    neighbor_indices = indices[:, 1]

    # Track divergence
    divergence = np.zeros(fit_len)
    counts = np.zeros(fit_len)

    for i in range(n_points):
        j = neighbor_indices[i]
        for k in range(fit_len):
            if i + k < n_points and j + k < n_points:
                distance = np.abs(x[i + k] - x[j + k])
                divergence[k] += np.log(distance + 1e-12)
                counts[k] += 1

    # Calculate average divergence
    valid_mask = counts > 0
    if not np.any(valid_mask):
        raise ValueError("No valid divergence pairs found")

    avg_divergence = divergence[valid_mask] / counts[valid_mask]
    time_steps = np.arange(fit_len)[valid_mask]

    # Linear fit
    slope, _ = np.polyfit(time_steps, avg_divergence, 1)
    return float(slope)


def ks_entropy(x: np.ndarray, n_bins: int = 10) -> float:
    """
    Estimate Kolmogorov-Sinai entropy using partition-based method.

    Parameters
    ----------
    x : array-like
        Time series data
    n_bins : int
        Number of quantile-based bins

    Returns
    -------
    float
        KS entropy estimate (nats per sample)
    """
    x = np.asarray(x, dtype=float)

    if len(x) < n_bins * 2:
        raise ValueError(f"Time series too short for {n_bins} bins")

    # Create quantile bins
    try:
        categories, bins = pd.qcut(
            x, q=n_bins, labels=False, retbins=True, duplicates="drop"
        )
    except ValueError as e:
        raise ValueError(f"Failed to create bins: {e}")

    symbols = np.array(categories, dtype=int)
    n_symbols = len(bins) - 1

    if n_symbols < 2:
        warnings.warn("Too few unique symbols for entropy calculation")
        return 0.0

    # Build transition matrix
    transition_counts = np.zeros((n_symbols, n_symbols), dtype=int)

    for t in range(len(symbols) - 1):
        i, j = symbols[t], symbols[t + 1]
        if i < n_symbols and j < n_symbols:  # Safety check
            transition_counts[i, j] += 1

    # Normalize to probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        transition_probs = transition_counts / row_sums
    transition_probs[np.isnan(transition_probs)] = 0.0

    # Stationary distribution
    stationary_probs = row_sums.flatten() / row_sums.sum()

    # Calculate entropy
    entropy = 0.0
    for i in range(n_symbols):
        for j in range(n_symbols):
            if transition_probs[i, j] > 0:
                entropy -= (
                    stationary_probs[i]
                    * transition_probs[i, j]
                    * np.log(transition_probs[i, j])
                )

    return float(entropy)


def fourier_harmonic_count(x: np.ndarray, sampling_interval: float = 0.1) -> int:
    """
    Count Fourier harmonics with amplitude above mean.

    Parameters
    ----------
    x : array-like
        Time series data
    sampling_interval : float
        Sampling time interval

    Returns
    -------
    int
        Number of significant harmonics
    """
    x = np.asarray(x, dtype=float)

    if len(x) < 2:
        return 0

    frequencies = np.fft.fftfreq(len(x), d=sampling_interval)
    magnitudes = 2 * np.abs(fft(x)) / len(x)

    # Consider only positive frequencies
    positive_freq_idx = np.where(frequencies > 0)[0]
    if len(positive_freq_idx) == 0:
        return 0

    positive_magnitudes = magnitudes[positive_freq_idx]
    mean_magnitude = np.mean(positive_magnitudes)

    return int(np.sum(positive_magnitudes > mean_magnitude))
