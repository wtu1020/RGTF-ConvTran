"""RR interval and HRV feature extraction used by RGTF-ConvTran."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.signal import butter, filtfilt, find_peaks
except ImportError as exc:  # pragma: no cover - handled when detection is called
    butter = filtfilt = find_peaks = None
    SCIPY_IMPORT_ERROR: Exception | None = exc
else:
    SCIPY_IMPORT_ERROR = None


FEATURE_NAMES = (
    "mean_rr",
    "std_rr",
    "rmssd",
    "sdsd",
    "pnn50",
    "mean_hr",
    "std_hr",
    "min_rr",
    "max_rr",
    "rr_range",
    "cvrr",
    "median_rr",
    "iqr_rr",
    "num_r_peaks",
)

PROMINENCE_CANDIDATES = (0.25, 0.35, 0.50, 0.75, 1.00)
MIN_RR_SECONDS = 0.30
MAX_RR_SECONDS = 2.00


def _as_time_channels(ecg: np.ndarray) -> np.ndarray:
    """Convert a generic ECG array to ``[time, channels]`` for peak detection."""
    value = np.asarray(ecg, dtype=np.float32)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2:
        raise ValueError(f"ECG must be a 1D or 2D array, got shape={value.shape}")
    if value.shape[0] <= 12 and value.shape[1] > value.shape[0]:
        value = value.T
    return value.astype(np.float32)


def zscore_signal(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    value = np.asarray(signal, dtype=np.float32)
    return (
        (value - np.nanmean(value)) / (np.nanstd(value) + eps)
    ).astype(np.float32)


def bandpass_qrs(
    signal: np.ndarray,
    sampling_rate_hz: float,
    low_hz: float = 5.0,
    high_hz: float = 20.0,
) -> np.ndarray:
    """Second-order QRS-band filter, falling back to normalized raw ECG."""
    value = zscore_signal(signal)
    if butter is None or filtfilt is None:
        return value
    nyquist = 0.5 * sampling_rate_hz
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 0.99)
    if not 0 < low < high < 1:
        return value
    try:
        numerator, denominator = butter(2, [low, high], btype="bandpass")
        filtered = filtfilt(numerator, denominator, value).astype(np.float32)
        return zscore_signal(filtered)
    except (ValueError, RuntimeError, FloatingPointError):
        return value


def _candidate_score(
    peaks: np.ndarray,
    signal: np.ndarray,
    sampling_rate_hz: float,
) -> tuple[float, int, float]:
    if peaks is None or len(peaks) < 2:
        return -1e9, 0, 0.0
    rr = np.diff(peaks) / sampling_rate_hz
    valid = (rr >= MIN_RR_SECONDS) & (rr <= MAX_RR_SECONDS)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return -1e9, 0, 0.0
    valid_rr = rr[valid]
    median_rr = float(np.median(valid_rr))
    coefficient_of_variation = float(
        np.std(valid_rr) / (np.mean(valid_rr) + 1e-8)
    )
    prominence_proxy = float(np.median(np.abs(signal[peaks])))
    score = (
        valid_count
        + 0.20 * prominence_proxy
        - 0.15 * coefficient_of_variation
        - 0.05 * abs(median_rr - 0.8)
    )
    return score, valid_count, coefficient_of_variation


def detect_r_peaks_auto(
    ecg: np.ndarray,
    sampling_rate_hz: float,
    lead_mode: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Try candidate leads, polarities, and prominences; return the best peaks."""
    if find_peaks is None:
        raise ImportError(
            "scipy.signal.find_peaks is required for R-peak detection"
        ) from SCIPY_IMPORT_ERROR
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    value = _as_time_channels(ecg)
    number_of_leads = value.shape[1]
    if lead_mode == "first":
        lead_indices = [0]
    elif lead_mode == "second":
        lead_indices = [1 if number_of_leads > 1 else 0]
    elif lead_mode == "mean":
        lead_indices = [-1]
    elif lead_mode == "auto":
        lead_indices = list(range(number_of_leads))
        if number_of_leads > 1:
            lead_indices.append(-1)
    else:
        raise ValueError(f"Unsupported lead_mode: {lead_mode}")

    best: dict[str, Any] = {
        "score": -1e18,
        "peaks": np.asarray([], dtype=np.int64),
        "lead_index": None,
        "inverted": False,
        "prominence": None,
        "valid_rr_count": 0,
        "rr_cv": None,
    }
    minimum_distance = max(1, int(round(0.25 * sampling_rate_hz)))
    for lead_index in lead_indices:
        if lead_index == -1:
            raw_signal = value.mean(axis=1)
            lead_name = "mean"
        else:
            raw_signal = value[:, lead_index]
            lead_name = str(lead_index)
        base_signal = bandpass_qrs(raw_signal, sampling_rate_hz)
        for inverted in (False, True):
            candidate_signal = -base_signal if inverted else base_signal
            for prominence in PROMINENCE_CANDIDATES:
                peaks, _ = find_peaks(
                    candidate_signal,
                    distance=minimum_distance,
                    prominence=prominence,
                )
                score, valid_count, rr_cv = _candidate_score(
                    peaks, candidate_signal, sampling_rate_hz
                )
                if score > best["score"]:
                    best.update(
                        {
                            "score": float(score),
                            "peaks": peaks.astype(np.int64),
                            "lead_index": lead_name,
                            "inverted": bool(inverted),
                            "prominence": float(prominence),
                            "valid_rr_count": int(valid_count),
                            "rr_cv": float(rr_cv),
                        }
                    )
    peaks = best["peaks"]
    metadata = {
        "detected_lead": best["lead_index"],
        "inverted": best["inverted"],
        "prominence": best["prominence"],
        "r_peak_count": int(len(peaks)),
        "valid_rr_count": int(best["valid_rr_count"]),
        "rr_cv_detect": best["rr_cv"],
        "r_peak_indices": peaks.tolist(),
    }
    return peaks, metadata


def compute_hrv_features_from_peaks(
    peaks: np.ndarray,
    sampling_rate_hz: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute the 14 RR/HRV values from R-peak sample indices."""
    peaks = np.asarray(peaks, dtype=np.int64)
    number_of_peaks = int(len(peaks))
    if number_of_peaks < 3:
        features = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
        features[-1] = float(number_of_peaks)
        return features, {"valid_hrv": 0, "usable_rr_count": 0}
    rr = np.diff(peaks) / float(sampling_rate_hz)
    rr = rr[(rr >= MIN_RR_SECONDS) & (rr <= MAX_RR_SECONDS)]
    if len(rr) < 2:
        features = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
        features[-1] = float(number_of_peaks)
        return features, {"valid_hrv": 0, "usable_rr_count": int(len(rr))}
    rr_differences = np.diff(rr)
    heart_rate = 60.0 / np.clip(rr, 1e-6, None)
    mean_rr = float(np.mean(rr))
    std_rr = float(np.std(rr, ddof=1))
    rmssd = float(np.sqrt(np.mean(rr_differences**2))) if len(rr_differences) else 0.0
    sdsd = (
        float(np.std(rr_differences, ddof=1))
        if len(rr_differences) > 1
        else 0.0
    )
    pnn50 = (
        float(np.mean(np.abs(rr_differences) > 0.05))
        if len(rr_differences)
        else 0.0
    )
    minimum_rr = float(np.min(rr))
    maximum_rr = float(np.max(rr))
    percentile_75, percentile_25 = np.percentile(rr, [75, 25])
    features = np.asarray(
        [
            mean_rr,
            std_rr,
            rmssd,
            sdsd,
            pnn50,
            float(np.mean(heart_rate)),
            float(np.std(heart_rate, ddof=1)),
            minimum_rr,
            maximum_rr,
            maximum_rr - minimum_rr,
            std_rr / (mean_rr + 1e-8),
            float(np.median(rr)),
            float(percentile_75 - percentile_25),
            float(number_of_peaks),
        ],
        dtype=np.float32,
    )
    return features, {"valid_hrv": 1, "usable_rr_count": int(len(rr))}


def extract_rr_hrv_features(
    ecg: np.ndarray,
    sampling_rate_hz: float,
    lead_mode: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect R peaks and compute the raw 14-dimensional feature vector."""
    peaks, detection_metadata = detect_r_peaks_auto(
        ecg, sampling_rate_hz, lead_mode
    )
    features, hrv_metadata = compute_hrv_features_from_peaks(
        peaks, sampling_rate_hz
    )
    return features, {**detection_metadata, **hrv_metadata}


def fit_feature_normalizer(
    raw_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit median imputation and z-score statistics on a reference split."""
    value = np.asarray(raw_features, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected [N,{len(FEATURE_NAMES)}] features")
    if value.shape[0] == 0:
        raise ValueError("At least one reference sample is required")
    fill_values = np.nanmedian(value, axis=0)
    fill_values = np.where(np.isfinite(fill_values), fill_values, 0.0).astype(
        np.float32
    )
    filled = value.copy()
    invalid = ~np.isfinite(filled)
    if invalid.any():
        filled[invalid] = np.take(fill_values, np.where(invalid)[1])
    mean = filled.mean(axis=0).astype(np.float32)
    std = filled.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return fill_values, mean, std


def transform_features(
    raw_features: np.ndarray,
    fill_values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply stored imputation and normalization statistics."""
    value = np.asarray(raw_features, dtype=np.float32).copy()
    single_sample = value.ndim == 1
    if single_sample:
        value = value[None, :]
    if value.ndim != 2 or value.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected features ending in dimension {len(FEATURE_NAMES)}")
    fill_values = np.asarray(fill_values, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    expected = (len(FEATURE_NAMES),)
    if fill_values.shape != expected or mean.shape != expected or std.shape != expected:
        raise ValueError(f"Normalizer vectors must have shape {expected}")
    invalid = ~np.isfinite(value)
    if invalid.any():
        value[invalid] = np.take(fill_values, np.where(invalid)[1])
    normalized = (value - mean) / std
    value = value.astype(np.float32)
    normalized = normalized.astype(np.float32)
    if single_sample:
        return value[0], normalized[0]
    return value, normalized
