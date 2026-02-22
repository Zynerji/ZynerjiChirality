"""Axial chirality benchmark molecules — biaryls and atropisomers.

Note: Axial chirality in SMILES is not standardized. These use
extended SMILES notation or explicit 3D stereochemistry.
"""

from __future__ import annotations

# Biaryl atropisomer pairs
# BINAP and simpler atropisomeric biaryls
# Note: Many of these rely on 3D conformer generation for dihedral detection
AXIAL_PAIRS: list[tuple[str, str, str, str]] = [
    (
        "1,1'-Binaphthyl",
        # (R)-1,1'-Binaphthyl: two naphthalene rings connected at 1-position
        "c1ccc2c(c1)-c1ccc3ccccc3c1C2",  # simplified — detection relies on ortho bulk
        "c1ccc2c(c1)-c1ccc3ccccc3c1C2",  # same SMILES — 3D differentiates
        "Prototype atropisomer, restricted rotation",
    ),
]

# Single-molecule axial chirality tests
AXIAL_MOLECULES: list[tuple[str, str, str]] = [
    (
        "2,2'-Dimethylbiphenyl",
        "Cc1ccccc1-c1ccccc1C",
        "Ortho-substituted biphenyl — potential atropisomer",
    ),
    (
        "2,2'-Dihydroxybiphenyl",
        "Oc1ccccc1-c1ccccc1O",
        "Ortho-hydroxyl groups restrict rotation",
    ),
]


def get_axial_pairs() -> list[tuple[str, str, str, str]]:
    """Return axial chirality pairs: [(name, r_smiles, s_smiles, note)]."""
    return list(AXIAL_PAIRS)


def get_axial_molecules() -> list[tuple[str, str, str]]:
    """Return axial chirality test molecules: [(name, smiles, note)]."""
    return list(AXIAL_MOLECULES)
