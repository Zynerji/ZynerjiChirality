"""Property enrichment functions for chiral materials.

Derives material-level physical properties (circular dichroism, spin selectivity,
optical rotation, band gap shifts) from the dual-helix spectral decomposition.
All functions operate on SpectralCoords — no new external dependencies.
"""

from __future__ import annotations

import numpy as np

from zynerji_chirality.core.dual_helix import SpectralCoords


def estimate_cd_activity(spectral_coords: SpectralCoords) -> tuple[float, str]:
    """Estimate circular dichroism (CD) intensity and sign from eigenvalue asymmetry.

    CD activity correlates with the magnitude of left-right spectral asymmetry.
    The sign indicates which circularly polarized light is preferentially absorbed.

    Parameters
    ----------
    spectral_coords : SpectralCoords
        Spectral coordinates from dual-helix decomposition.

    Returns
    -------
    tuple[float, str]
        (cd_intensity, cd_sign) where intensity >= 0 and sign is
        "positive", "negative", or "inactive".
    """
    ecos = spectral_coords.eigenvalues_cos
    esin = spectral_coords.eigenvalues_sin

    if len(ecos) == 0 or len(esin) == 0:
        return 0.0, "inactive"

    k = min(len(ecos), len(esin))
    diff = ecos[:k] - esin[:k]

    # CD intensity: RMS of eigenvalue differences, normalized by mean magnitude
    denom = np.mean(np.abs(ecos[:k]) + np.abs(esin[:k]))
    if denom < 1e-12:
        return 0.0, "inactive"

    intensity = float(np.sqrt(np.mean(diff ** 2)) / denom)

    # Sign: weighted sum of differences (positive = right-handed preference)
    weighted_diff = float(np.sum(diff * (np.abs(ecos[:k]) + np.abs(esin[:k]))))

    if abs(intensity) < 1e-6:
        return 0.0, "inactive"
    elif weighted_diff > 0:
        return intensity, "positive"
    else:
        return intensity, "negative"


def estimate_ciss_score(spectral_coords: SpectralCoords) -> float:
    """Estimate chirality-induced spin selectivity (CISS) score.

    The CISS effect arises from the helical potential experienced by electrons
    traversing chiral structures. We approximate it from the cos-sin eigenvalue
    gap — larger gaps indicate stronger helical asymmetry.

    Parameters
    ----------
    spectral_coords : SpectralCoords
        Spectral coordinates from dual-helix decomposition.

    Returns
    -------
    float
        CISS score normalized to [0, 1]. Higher = stronger spin selectivity.
    """
    ecos = spectral_coords.eigenvalues_cos
    esin = spectral_coords.eigenvalues_sin

    if len(ecos) == 0 or len(esin) == 0:
        return 0.0

    k = min(len(ecos), len(esin))
    gap = np.abs(ecos[:k] - esin[:k])

    # Normalize by total spectral spread
    total = np.sum(np.abs(ecos[:k])) + np.sum(np.abs(esin[:k]))
    if total < 1e-12:
        return 0.0

    raw_score = float(np.sum(gap) / total)
    # Sigmoid normalization to [0, 1] with saturation around 0.5 raw
    return float(1.0 / (1.0 + np.exp(-10.0 * (raw_score - 0.1))))


def classify_optical_rotation(chirality_sign: float, chirality_score: float) -> str:
    """Classify optical rotation direction from chirality detection results.

    Parameters
    ----------
    chirality_sign : float
        +1.0 for R-dominant, -1.0 for S-dominant, 0.0 for achiral.
    chirality_score : float
        Magnitude of chirality detection score.

    Returns
    -------
    str
        "dextrorotatory (+)", "levorotatory (-)", or "inactive".
    """
    if chirality_score < 1e-6 or chirality_sign == 0.0:
        return "inactive"
    elif chirality_sign > 0:
        return "dextrorotatory (+)"
    else:
        return "levorotatory (-)"


def band_gap_chirality_shift(spectral_coords: SpectralCoords) -> float:
    """Predict band gap shift due to chirality (relevant for perovskites).

    Chiral distortions in crystal structures induce Rashba/Dresselhaus
    spin-orbit coupling, which modifies the electronic band gap. We
    approximate the shift magnitude from spectral asymmetry.

    Parameters
    ----------
    spectral_coords : SpectralCoords
        Spectral coordinates from dual-helix decomposition.

    Returns
    -------
    float
        Predicted band gap shift in arbitrary units (positive = widening).
        Scale: typical organic chiral molecules ~0.01-0.1, strong chirality ~0.1-0.5.
    """
    ecos = spectral_coords.eigenvalues_cos
    esin = spectral_coords.eigenvalues_sin

    if len(ecos) == 0 or len(esin) == 0:
        return 0.0

    k = min(len(ecos), len(esin))

    # Use the spectral gap at the lowest eigenvalues (closest to band edge)
    # Weight lower eigenvalues more heavily (they correspond to band-edge states)
    weights = np.exp(-np.arange(k, dtype=np.float64))
    diff = np.abs(ecos[:k] - esin[:k])

    return float(np.sum(weights * diff) / np.sum(weights))
