"""L/D amino acid enantiomer pairs for chirality benchmarking.

All 20 standard amino acids have L and D forms (except glycine, which is achiral).
L-amino acids dominate biology; D-forms are mirror images.

SMILES use @/@ notation:
- L-amino acids: S configuration at alpha carbon (natural form)
- D-amino acids: R configuration at alpha carbon (mirror image)
"""

from __future__ import annotations

# (name, L-SMILES, D-SMILES)
# L = S config, D = R config at the alpha carbon
AMINO_ACID_PAIRS: list[tuple[str, str, str]] = [
    ("Alanine",       "N[C@@H](C)C(=O)O",                             "N[C@H](C)C(=O)O"),
    ("Valine",        "N[C@@H](C(C)C)C(=O)O",                         "N[C@H](C(C)C)C(=O)O"),
    ("Leucine",       "N[C@@H](CC(C)C)C(=O)O",                        "N[C@H](CC(C)C)C(=O)O"),
    ("Isoleucine",    "N[C@@H]([C@@H](C)CC)C(=O)O",                   "N[C@H]([C@H](C)CC)C(=O)O"),
    ("Proline",       "O=C(O)[C@@H]1CCCN1",                           "O=C(O)[C@H]1CCCN1"),
    ("Phenylalanine", "N[C@@H](Cc1ccccc1)C(=O)O",                     "N[C@H](Cc1ccccc1)C(=O)O"),
    ("Tryptophan",    "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",             "N[C@H](Cc1c[nH]c2ccccc12)C(=O)O"),
    ("Methionine",    "N[C@@H](CCSC)C(=O)O",                          "N[C@H](CCSC)C(=O)O"),
    ("Serine",        "N[C@@H](CO)C(=O)O",                            "N[C@H](CO)C(=O)O"),
    ("Threonine",     "N[C@@H]([C@@H](O)C)C(=O)O",                    "N[C@H]([C@H](O)C)C(=O)O"),
    ("Cysteine",      "N[C@@H](CS)C(=O)O",                            "N[C@H](CS)C(=O)O"),
    ("Tyrosine",      "N[C@@H](Cc1ccc(O)cc1)C(=O)O",                  "N[C@H](Cc1ccc(O)cc1)C(=O)O"),
    ("Asparagine",    "N[C@@H](CC(=O)N)C(=O)O",                       "N[C@H](CC(=O)N)C(=O)O"),
    ("Glutamine",     "N[C@@H](CCC(=O)N)C(=O)O",                      "N[C@H](CCC(=O)N)C(=O)O"),
    ("Aspartate",     "N[C@@H](CC(=O)O)C(=O)O",                       "N[C@H](CC(=O)O)C(=O)O"),
    ("Glutamate",     "N[C@@H](CCC(=O)O)C(=O)O",                      "N[C@H](CCC(=O)O)C(=O)O"),
    ("Lysine",        "N[C@@H](CCCCN)C(=O)O",                         "N[C@H](CCCCN)C(=O)O"),
    ("Arginine",      "N[C@@H](CCCNC(=N)N)C(=O)O",                    "N[C@H](CCCNC(=N)N)C(=O)O"),
    ("Histidine",     "N[C@@H](Cc1c[nH]cn1)C(=O)O",                   "N[C@H](Cc1c[nH]cn1)C(=O)O"),
]

# Glycine is achiral (no chiral center)
GLYCINE_SMILES = "NCC(=O)O"


def get_all_pairs() -> list[tuple[str, str, str]]:
    """Return all L/D amino acid pairs."""
    return list(AMINO_ACID_PAIRS)


def get_achiral() -> list[tuple[str, str]]:
    """Return achiral amino acids: [(name, SMILES)]."""
    return [("Glycine", GLYCINE_SMILES)]
