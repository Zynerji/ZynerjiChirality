#!/usr/bin/env python3
"""Compare enantiomer retrieval precision for different fingerprint types.

Compares ZynerjiChirality spectral fingerprints vs Morgan ECFP4 vs MACCS keys
for enantiomer pair retrieval across the benchmark sets.

Two evaluation modes:
1. Raw cosine k-NN (naive similarity search)
2. Chirality-aware search (FingerprintStore enantiomer mode — filters by
   opposite chirality sign, then ranks by structural similarity)
"""

from __future__ import annotations

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

from zynerji_chirality.chirality.detector import HelixChiralityDetector
from zynerji_chirality.chirality.fingerprint import (
    chirality_fingerprint,
    fingerprint_similarity,
)
from zynerji_chirality.db.store import FingerprintStore
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
    """Evaluate enantiomer retrieval via raw cosine k-NN."""
    all_smiles = []
    all_fps = []
    pair_map = {}

    for pair in pairs:
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

    n = len(all_fps)
    precision_at_1 = 0
    precision_at_3 = 0
    n_queries = 0

    for query_idx in pair_map:
        expected_idx = pair_map[query_idx]
        query_fp = all_fps[query_idx]

        sims = []
        for j in range(n):
            if j == query_idx:
                continue
            sim = sim_func(query_fp, all_fps[j])
            sims.append((j, sim))

        sims.sort(key=lambda x: x[1], reverse=True)

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


def evaluate_store_retrieval(pairs) -> dict:
    """Evaluate enantiomer retrieval using FingerprintStore enantiomer search mode.

    This uses chirality-aware search: filter by opposite chirality sign,
    then rank by structural similarity. This is how ZynerjiChirality
    fingerprints are intended to be used for enantiomer retrieval.
    """
    detector = HelixChiralityDetector()
    store = FingerprintStore(":memory:")

    # Ingest all molecules
    smiles_to_id = {}
    pair_map = {}  # mol_id -> enantiomer_mol_id

    mol_ids_a = []
    mol_ids_b = []

    for pair in pairs:
        smi_a = pair[1]
        smi_b = pair[2]
        name = pair[0]
        try:
            fp_a = chirality_fingerprint(smi_a, nbits=128)
            res_a = detector.detect(smi_a)
            fp_b = chirality_fingerprint(smi_b, nbits=128)
            res_b = detector.detect(smi_b)
        except Exception:
            continue

        id_a = store.add(smi_a, fp_a, res_a.chirality_score, res_a.chirality_sign,
                         name=f"{name}-A")
        id_b = store.add(smi_b, fp_b, res_b.chirality_score, res_b.chirality_sign,
                         name=f"{name}-B")

        smiles_to_id[smi_a] = id_a
        smiles_to_id[smi_b] = id_b
        pair_map[id_a] = id_b
        pair_map[id_b] = id_a
        mol_ids_a.append((id_a, fp_a))
        mol_ids_b.append((id_b, fp_b))

    if not pair_map:
        return {"precision_at_1": 0.0, "precision_at_3": 0.0, "n_pairs": 0}

    precision_at_1 = 0
    precision_at_3 = 0
    n_queries = 0

    for query_id, expected_id in pair_map.items():
        entry = store.get(query_id)
        if entry is None:
            continue

        query_fp = entry["fingerprint"]
        results = store.search_similar(query_fp, k=3, mode="enantiomer")

        if not results:
            n_queries += 1
            continue

        result_ids = [r.mol_id for r in results]
        if result_ids[0] == expected_id:
            precision_at_1 += 1
        if expected_id in result_ids[:3]:
            precision_at_3 += 1
        n_queries += 1

    store.close()

    return {
        "precision_at_1": precision_at_1 / max(n_queries, 1),
        "precision_at_3": precision_at_3 / max(n_queries, 1),
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

    # --Mode 1: Raw cosine k-NN ──
    print("\n--Mode 1: Raw Cosine k-NN (naive similarity search) --")
    print(f"\n{'Method':<30s} {'P@1':>8s} {'P@3':>8s} {'Pairs':>6s}")
    print("-" * 56)

    methods = [
        ("ZynerjiChirality (128d)", lambda s: chirality_fingerprint(s, nbits=128), cosine_similarity),
        ("Morgan ECFP4 (2048d)", morgan_fingerprint, cosine_similarity),
        ("MACCS Keys (167d)", maccs_fingerprint, cosine_similarity),
    ]

    for method_name, fp_func, sim_func in methods:
        result = evaluate_retrieval(all_pairs, fp_func, sim_func, method_name)
        print(f"{result['name']:<30s} {result['precision_at_1']:>7.1%} {result['precision_at_3']:>7.1%} {result['n_pairs']:>6d}")

    print("\n  Note: Morgan/MACCS score 100% because enantiomers have identical")
    print("  connectivity -> identical fingerprints. ZynerjiChirality scores lower")
    print("  because it deliberately encodes handedness (R != S in feature space).")

    # --Mode 2: Chirality-aware enantiomer search ──
    print("\n--Mode 2: Chirality-Aware Enantiomer Search --")
    print("  (FingerprintStore: filter by opposite sign, then rank by similarity)")
    print()

    store_result = evaluate_store_retrieval(all_pairs)
    print(f"{'ZynerjiChirality + store':<30s} {store_result['precision_at_1']:>7.1%} {store_result['precision_at_3']:>7.1%} {store_result['n_pairs']:>6d}")

    print("\n  This is the intended usage: chirality sign separates R from S,")
    print("  then cosine similarity finds the structurally closest enantiomer.")

    # --Mode 3: Chirality discrimination ──
    print("\n--Mode 3: Chirality Discrimination --")
    print("  (Can the fingerprint tell enantiomers apart from identical molecules?)")
    print()

    n_discriminated = 0
    n_total = 0
    for pair in all_pairs:
        smi_a, smi_b = pair[1], pair[2]
        try:
            fp_a = chirality_fingerprint(smi_a, nbits=128)
            fp_b = chirality_fingerprint(smi_b, nbits=128)
            self_sim = cosine_similarity(fp_a, fp_a)
            cross_sim = cosine_similarity(fp_a, fp_b)
            if self_sim > cross_sim:
                n_discriminated += 1
            n_total += 1
        except Exception:
            continue

    morgan_disc = 0
    maccs_disc = 0
    for pair in all_pairs:
        smi_a, smi_b = pair[1], pair[2]
        try:
            m_a = morgan_fingerprint(smi_a)
            m_b = morgan_fingerprint(smi_b)
            if cosine_similarity(m_a, m_a) > cosine_similarity(m_a, m_b):
                morgan_disc += 1
            k_a = maccs_fingerprint(smi_a)
            k_b = maccs_fingerprint(smi_b)
            if cosine_similarity(k_a, k_a) > cosine_similarity(k_a, k_b):
                maccs_disc += 1
        except Exception:
            pass

    print(f"{'Method':<30s} {'Discrimination':>14s}")
    print("-" * 46)
    print(f"{'ZynerjiChirality (128d)':<30s} {n_discriminated:>6d}/{n_total:<6d} ({n_discriminated/max(n_total,1):.1%})")
    print(f"{'Morgan ECFP4 (2048d)':<30s} {morgan_disc:>6d}/{n_total:<6d} ({morgan_disc/max(n_total,1):.1%})")
    print(f"{'MACCS Keys (167d)':<30s} {maccs_disc:>6d}/{n_total:<6d} ({maccs_disc/max(n_total,1):.1%})")

    print("\n  Discrimination = fingerprint(A) is more similar to itself than to")
    print("  fingerprint(enantiomer(A)). Higher means the method can tell apart R vs S.")
    print("  This is the core capability that ZynerjiChirality adds.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
