"""Public RGTF-ConvTran model and RR/HRV feature APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .rr_hrv import (
    FEATURE_NAMES,
    detect_r_peaks_auto,
    extract_rr_hrv_features,
    fit_feature_normalizer,
    transform_features,
)

if TYPE_CHECKING:
    from .model import RGTFConvTran


def __getattr__(name: str) -> Any:
    if name == "RGTFConvTran":
        from .model import RGTFConvTran

        return RGTFConvTran
    raise AttributeError(name)

__all__ = [
    "FEATURE_NAMES",
    "RGTFConvTran",
    "detect_r_peaks_auto",
    "extract_rr_hrv_features",
    "fit_feature_normalizer",
    "transform_features",
]
