"""Planar chirality benchmark molecules — paracyclophanes.

Focus on organic [2.2]paracyclophane derivatives which are well-supported
by RDKit 3D embedding. Metallocenes are marked experimental due to
limited RDKit support for metal-cyclopentadienyl bonding.
"""

from __future__ import annotations

# [2.2]Paracyclophane: the prototypical planar chiral molecule
# Note: SMILES encoding of planar chirality is non-standard;
# detection relies on 3D geometry analysis
PLANAR_MOLECULES: list[tuple[str, str, str]] = [
    (
        "[2.2]Paracyclophane",
        "C1CC2=CC=C(CC3=CC=C1C=C3)C=C2",
        "Prototype planar chirality — bridged aromatic rings",
    ),
    (
        "4-Bromo-[2.2]paracyclophane",
        "BrC1=CC2=CC=C(CC3=CC=C(C=C3)CC2)C=C1",
        "Substituted paracyclophane — monosubstituted variant",
    ),
]

# Experimental: metallocenes (may not work with RDKit)
EXPERIMENTAL_METALLOCENES: list[tuple[str, str, str]] = [
    (
        "Ferrocene",
        "[Fe+2].[cH-]1cccc1.[cH-]1cccc1",
        "EXPERIMENTAL: RDKit metallocene bonding limited",
    ),
]


def get_planar_molecules() -> list[tuple[str, str, str]]:
    """Return planar chirality test molecules: [(name, smiles, note)]."""
    return list(PLANAR_MOLECULES)


def get_experimental_metallocenes() -> list[tuple[str, str, str]]:
    """Return experimental metallocene molecules: [(name, smiles, note)]."""
    return list(EXPERIMENTAL_METALLOCENES)
