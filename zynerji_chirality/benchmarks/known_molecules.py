"""Curated SMILES database for negative controls and known molecules.

Includes achiral molecules, meso compounds, and constitutional isomers.
"""

from __future__ import annotations

# Achiral molecules: should score ~0
ACHIRAL_MOLECULES: list[tuple[str, str]] = [
    ("Methane", "C"),
    ("Ethane", "CC"),
    ("Ethanol", "CCO"),
    ("Benzene", "c1ccccc1"),
    ("Cyclohexane", "C1CCCCC1"),
    ("Acetic acid", "CC(=O)O"),
    ("Acetone", "CC(=O)C"),
    ("Toluene", "Cc1ccccc1"),
    ("Naphthalene", "c1ccc2ccccc2c1"),
    ("Water", "O"),
    ("Methanol", "CO"),
    ("Urea", "NC(=O)N"),
    ("Formaldehyde", "C=O"),
    ("Glycine", "NCC(=O)O"),
    ("Propane", "CCC"),
    ("Butane", "CCCC"),
]

# Meso compounds: have chiral centers but are achiral overall
# due to internal mirror plane
MESO_COMPOUNDS: list[tuple[str, str]] = [
    ("meso-Tartaric acid", "O[C@H](C(=O)O)[C@@H](O)C(=O)O"),
    ("meso-2,3-Butanediol", "C[C@H](O)[C@@H](O)C"),
]

# Constitutional isomers: same formula, different connectivity (not enantiomers)
CONSTITUTIONAL_ISOMERS: list[tuple[str, str, str]] = [
    ("Ethanol vs Dimethyl ether", "CCO", "COC"),
    ("1-Propanol vs 2-Propanol", "CCCO", "CC(O)C"),
    ("Butane vs Isobutane", "CCCC", "CC(C)C"),
]


def get_achiral() -> list[tuple[str, str]]:
    """Return achiral molecules: [(name, SMILES)]."""
    return list(ACHIRAL_MOLECULES)


def get_meso() -> list[tuple[str, str]]:
    """Return meso compounds: [(name, SMILES)]."""
    return list(MESO_COMPOUNDS)


def get_constitutional_isomers() -> list[tuple[str, str, str]]:
    """Return constitutional isomers: [(name, smiles_a, smiles_b)]."""
    return list(CONSTITUTIONAL_ISOMERS)
