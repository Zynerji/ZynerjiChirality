"""Expanded benchmark molecules — sugars, natural products, steroids.

Tests the engine on harder multi-center molecules with known chirality.
"""

from __future__ import annotations


# Sugar pairs: D and L forms are enantiomers
# D-sugars have the highest-numbered chiral center in R config
SUGAR_PAIRS: list[tuple[str, str, str]] = [
    (
        "Glucose",
        "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",   # D-Glucose
        "OC[C@@H]1OC(O)[C@@H](O)[C@H](O)[C@H]1O",    # L-Glucose
    ),
    (
        "Galactose",
        "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@H]1O",     # D-Galactose
        "OC[C@@H]1OC(O)[C@@H](O)[C@H](O)[C@@H]1O",   # L-Galactose
    ),
    (
        "Mannose",
        "OC[C@H]1OC(O)[C@@H](O)[C@@H](O)[C@@H]1O",   # D-Mannose
        "OC[C@@H]1OC(O)[C@H](O)[C@H](O)[C@H]1O",     # L-Mannose
    ),
    (
        "Fructose",
        "OC[C@@H]1OC(O)(CO)[C@H](O)[C@@H]1O",         # D-Fructose
        "OC[C@H]1OC(O)(CO)[C@@H](O)[C@H]1O",          # L-Fructose
    ),
]

# Natural product enantiomer pairs
NATURAL_PRODUCT_PAIRS: list[tuple[str, str, str, str]] = [
    (
        "Menthol",
        "C[C@@H]1CC[C@H](C(C)C)[C@H](O)C1",  # (-)-Menthol (1R,2S,5R)
        "C[C@H]1CC[C@@H](C(C)C)[C@@H](O)C1",  # (+)-Menthol (1S,2R,5S)
        "Cooling agent, most abundant in peppermint",
    ),
    (
        "Camphor",
        "C[C@]1(C)C2CC[C@]1(C)C(=O)C2",       # (+)-Camphor (1R,4R)
        "C[C@@]1(C)C2CC[C@@]1(C)C(=O)C2",     # (-)-Camphor (1S,4S)
        "Natural terpenoid from camphor tree",
    ),
    (
        "Carvone",
        "C[C@@H]1CC(=CC1=O)C",                 # (R)-Carvone (spearmint)
        "C[C@H]1CC(=CC1=O)C",                  # (S)-Carvone (caraway)
        "(R)=spearmint, (S)=caraway — classic enantiomer smell difference",
    ),
]

# Steroids: single-molecule chirality tests (no enantiomer pair available)
STEROID_MOLECULES: list[tuple[str, str, str]] = [
    (
        "Cholesterol",
        "[C@@H]1(CC[C@@H]2[C@]1(CC[C@H]1[C@H]2CC=C2C[C@@H](O)CC[C@@]21C)C)[C@H](C)CCCC(C)C",
        "Multi-center steroid, should be detected as chiral",
    ),
    (
        "Testosterone",
        "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@H]34)[C@@H]1CC[C@@H]2O",
        "Male sex hormone, 6 chiral centers",
    ),
]

# Tougher achiral controls
TOUGH_ACHIRAL: list[tuple[str, str]] = [
    ("Adamantane", "C1C2CC3CC1CC(C2)C3"),
    ("Norbornane", "C1CC2CC1CC2"),
    ("Biphenyl", "c1ccc(-c2ccccc2)cc1"),
    ("Ferrocene", "[Fe+2].[cH-]1cccc1.[cH-]1cccc1"),
]


def get_sugar_pairs() -> list[tuple[str, str, str]]:
    """Return sugar D/L pairs: [(name, d_smiles, l_smiles)]."""
    return list(SUGAR_PAIRS)


def get_natural_product_pairs() -> list[tuple[str, str, str, str]]:
    """Return natural product pairs: [(name, r_smiles, s_smiles, note)]."""
    return list(NATURAL_PRODUCT_PAIRS)


def get_steroid_molecules() -> list[tuple[str, str, str]]:
    """Return steroid molecules: [(name, smiles, note)]."""
    return list(STEROID_MOLECULES)


def get_tough_achiral() -> list[tuple[str, str]]:
    """Return tough achiral controls: [(name, smiles)]."""
    return list(TOUGH_ACHIRAL)
