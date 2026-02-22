#!/usr/bin/env python3
"""Compare enantiomer retrieval precision for different fingerprint types.

Compares ZynerjiChirality spectral fingerprints vs Morgan ECFP4 vs MACCS keys
for enantiomer pair retrieval across the benchmark sets.
"""

from __future__ import annotations

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

from zynerji_chirality.chirality.fingerprint import (
    chirality_fingerprint,
    fingerprint_similarity,
)
from zynerji_chirality.benchmarks.amino_acids import get_all_pairs
from zynerji_chirality.benchmarks.rs_pairs import get_all_pairs as get_drug_pairs


def morgan_fingerprint(smiles: str, nbits: int = 2048, radius: int = 2) -> np.ndarray:
    """Compute Morgan ECFP4 fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.float64)


def maccs_fingerprint(smiles: str) -> np.ndarray:
    """Compute MACCS keys fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.float64)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity mapped to [0, 1]."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return (float(np.dot(a, b) / (na * nb)) + 1.0) / 2.0


def evaluate_retrieval(pairs, fp_func, sim_func, name: str) -> dict:
    """Evaluate enantiomer retrieval: for each molecule, is its enantiomer the top match?"""
    # Build fingerprint database
    all_smiles = []
    all_fps = []
    pair_map = {}  # smiles -> enantiomer_smiles

    for pair in pairs:
        # pairs are (name, smi_a, smi_b) or (name, smi_a, smi_b, note)
        smi_a = pair[1]
        smi_b = pair[2]
        try:
            fp_a = fp_func(smi_a)
            fp_b = fp_func(smi_b)
            if fp_a is None or fp_b is None:
                continue
        except Exception:
            continue

        idx_a = len(all_smiles)
        all_smiles.append(smi_a)
        all_fps.append(fp_a)

        idx_b = len(all_smiles)
        all_smiles.append(smi_b)
        all_fps.append(fp_b)

        pair_map[idx_a] = idx_b
        pair_map[idx_b] = idx_a

    if not all_fps:
        return {"precision_at_1": 0.0, "precision_at_3": 0.0, "name": name, "n_pairs": 0}

    # For each query, rank all others by similarity
    n = len(all_fps)
    precision_at_1 = 0
    precision_at_3 = 0
    n_queries = 0

    for query_idx in pair_map:
        expected_idx = pair_map[query_idx]
        query_fp = all_fps[query_idx]

        # Compute similarities to all others
        sims = []
        for j in range(n):
            if j == query_idx:
                continue
            sim = sim_func(query_fp, all_fps[j])
            sims.append((j, sim))

        sims.sort(key=lambda x: x[1], reverse=True)

        # Check precision@k
        if sims[0][0] == expected_idx:
            precision_at_1 += 1
        if expected_idx in [s[0] for s in sims[:3]]:
            precision_at_3 += 1
        n_queries += 1

    return {
        "precision_at_1": precision_at_1 / max(n_queries, 1),
        "precision_at_3": precision_at_3 / max(n_queries, 1),
        "name": name,
        "n_pairs": len(pair_map) // 2,
    }


def main():
    print("Fingerprint Comparison: Enantiomer Retrieval Precision")
    print("=" * 70)

    # Collect all pairs
    amino_pairs = get_all_pairs()
    drug_pairs = get_drug_pairs()
    all_pairs = [(p[0], p[1], p[2]) for p in amino_pairs] + [(p[0], p[1], p[2]) for p in drug_pairs]

    print(f"\nDataset: {len(all_pairs)} enantiomer pairs "
          f"({len(amino_pairs)} amino acids + {len(drug_pairs)} drugs)")

    # Define fingerprint methods
    methods = [
        ("ZynerjiChirality (128d)", lambda s: chirality_fingerprint(s, nbits=128), cosine_similarity),
        ("Morgan ECFP4 (2048d)", morgan_fingerprint, cosine_similarity),
        ("MACCS Keys (167d)", maccs_fingerprint, cosine_similarity),
    ]

    print(f"\n{'Method':<30s} {'P@1':>8s} {'P@3':>8s} {'Pairs':>6s}")
    print("-" * 56)

    for method_name, fp_func, sim_func in methods:
        result = evaluate_retrieval(all_pairs, fp_func, sim_func, method_name)
        print(f"{result['name']:<30s} {result['precision_at_1']:>7.1%} {result['precision_at_3']:>7.1%} {result['n_pairs']:>6d}")

    print("\n" + "=" * 70)
    print("P@1 = precision at rank 1 (enantiomer is top match)")
    print("P@3 = precision at rank 3 (enantiomer is in top 3)")
    print("Higher is better. ZynerjiChirality should excel at enantiomer retrieval")
    print("because it encodes handedness directly in the spectral fingerprint.")


if __name__ == "__main__":
    main()
