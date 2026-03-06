#!/usr/bin/env python3
"""Atlas v5 benchmark: Compare full FP vs differential FP cosine similarity.

Runs on 12 drug enantiomer pairs and amino acids, printing cosine similarity
for both fingerprint types side-by-side.

Usage:
    python scripts/atlas_v5_diff_fp.py
"""

import numpy as np
from zynerji_chirality.chirality.fingerprint import (
    chirality_fingerprint,
    chirality_differential_fingerprint,
)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# Enantiomer pairs: (name, R-SMILES, S-SMILES)
PAIRS = [
    ("Alanine", "N[C@H](C)C(=O)O", "N[C@@H](C)C(=O)O"),
    ("Phenylalanine", "N[C@H](Cc1ccccc1)C(=O)O", "N[C@@H](Cc1ccccc1)C(=O)O"),
    ("Leucine", "N[C@H](CC(C)C)C(=O)O", "N[C@@H](CC(C)C)C(=O)O"),
    ("Valine", "N[C@H](C(C)C)C(=O)O", "N[C@@H](C(C)C)C(=O)O"),
    ("Serine", "N[C@H](CO)C(=O)O", "N[C@@H](CO)C(=O)O"),
    ("Ibuprofen", "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1", "CC(C)Cc1ccc([C@H](C)C(=O)O)cc1"),
    ("Naproxen", "COc1ccc2cc([C@@H](C)C(=O)O)ccc2c1", "COc1ccc2cc([C@H](C)C(=O)O)ccc2c1"),
    ("Propranolol", "CC(C)NCC(O)COc1cccc2ccccc12", "CC(C)NCC(O)COc1cccc2ccccc12"),  # achiral control
    ("Cysteine", "N[C@H](CS)C(=O)O", "N[C@@H](CS)C(=O)O"),
    ("Threonine", "N[C@H]([C@@H](O)C)C(=O)O", "N[C@@H]([C@H](O)C)C(=O)O"),
    ("Tryptophan", "N[C@H](Cc1c[nH]c2ccccc12)C(=O)O", "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O"),
    ("Methionine", "N[C@H](CCSC)C(=O)O", "N[C@@H](CCSC)C(=O)O"),
]


def main():
    print(f"{'Pair':<20} {'Full FP cos':>12} {'Diff FP cos':>12} {'Diff norm-R':>12} {'Diff norm-S':>12}")
    print("-" * 72)

    for name, smi_r, smi_s in PAIRS:
        try:
            fp_r = chirality_fingerprint(smi_r)
            fp_s = chirality_fingerprint(smi_s)
            full_cos = cosine(fp_r, fp_s)

            dfp_r = chirality_differential_fingerprint(smi_r, mode="pure_diff")
            dfp_s = chirality_differential_fingerprint(smi_s, mode="pure_diff")
            diff_cos = cosine(dfp_r, dfp_s)

            print(f"{name:<20} {full_cos:>12.4f} {diff_cos:>12.4f} {np.linalg.norm(dfp_r):>12.4f} {np.linalg.norm(dfp_s):>12.4f}")
        except Exception as e:
            print(f"{name:<20} ERROR: {e}")

    # Achiral controls
    print("\n--- Achiral Controls ---")
    for name, smi in [("Glycine", "NCC(=O)O"), ("Ethanol", "CCO"), ("Benzene", "c1ccccc1")]:
        dfp = chirality_differential_fingerprint(smi, mode="pure_diff")
        print(f"{name:<20} diff_fp_norm = {np.linalg.norm(dfp):.6f}")


if __name__ == "__main__":
    main()
