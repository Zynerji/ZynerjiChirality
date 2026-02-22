"""Feature engineering pipeline combining spectral fingerprints with molecular descriptors.

Builds combined feature vectors for ML models that are chirality-aware.
"""

from __future__ import annotations

import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from zynerji_chirality.chirality.fingerprint import chirality_fingerprint
from zynerji_chirality.chirality.detector import HelixChiralityDetector
from zynerji_chirality.core.dual_helix import HelixParams


# Default RDKit descriptors to compute
_DESCRIPTOR_FUNCS = [
    ("MW", Descriptors.MolWt),
    ("LogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumRings", Descriptors.RingCount),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
]


def _rdkit_descriptors(smiles: str) -> np.ndarray:
    """Compute standard RDKit molecular descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(_DESCRIPTOR_FUNCS))

    values = []
    for name, func in _DESCRIPTOR_FUNCS:
        try:
            values.append(float(func(mol)))
        except Exception:
            values.append(0.0)

    return np.array(values, dtype=np.float64)


def build_feature_vector(
    smiles: str,
    params: HelixParams | None = None,
    nbits: int = 128,
    include_per_center: bool = False,
    include_ensemble: bool = False,
    n_conformers: int = 5,
    max_centers: int = 8,
) -> dict[str, np.ndarray]:
    """Build a combined feature vector for ML models.

    Combines spectral fingerprint, RDKit descriptors, and optionally
    per-center scores and ensemble statistics.

    Parameters
    ----------
    smiles : str
        SMILES string.
    params : HelixParams, optional
        Helix parameters.
    nbits : int
        Spectral fingerprint length.
    include_per_center : bool
        Include per-center chirality scores.
    include_ensemble : bool
        Include ensemble conformer statistics.
    n_conformers : int
        Number of conformers for ensemble (if enabled).
    max_centers : int
        Maximum chiral centers to include features for.

    Returns
    -------
    dict[str, np.ndarray]
        Named feature arrays plus a 'combined' flat vector.
    """
    detector = HelixChiralityDetector(params=params)

    # Spectral fingerprint (128d by default)
    fp = chirality_fingerprint(smiles, params=params, nbits=nbits)

    # RDKit descriptors (10d)
    desc = _rdkit_descriptors(smiles)

    # Global chirality result (2d: score, sign)
    result = detector.detect(smiles)
    global_chirality = np.array([result.chirality_score, result.chirality_sign])

    features = {
        "spectral_fingerprint": fp,
        "rdkit_descriptors": desc,
        "global_chirality": global_chirality,
    }

    # Per-center scores (optional, 2 * max_centers dim)
    if include_per_center:
        per_center = detector.detect_per_center(smiles)
        center_features = np.zeros(max_centers * 2)
        for i, (idx, res) in enumerate(sorted(per_center.items())):
            if i >= max_centers:
                break
            center_features[i * 2] = res.chirality_score
            center_features[i * 2 + 1] = res.chirality_sign
        features["per_center_scores"] = center_features

    # Ensemble statistics (optional, 4d)
    if include_ensemble:
        ens = detector.detect_ensemble(smiles, n_conformers=n_conformers)
        ensemble_features = np.array([
            ens.mean_score, ens.std_score, ens.min_score, ens.max_score,
        ])
        features["ensemble_stats"] = ensemble_features

    # Combined flat vector
    features["combined"] = np.concatenate(list(features.values()))

    return features


def descriptor_names() -> list[str]:
    """Return ordered list of RDKit descriptor names."""
    return [name for name, _ in _DESCRIPTOR_FUNCS]
