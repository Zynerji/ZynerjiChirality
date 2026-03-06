#!/usr/bin/env python3
"""Atlas v8 — Large-N ChEMBL differential FP bioactivity correlation.

Uses 500 real ChEMBL differential activity pairs (>3x fold-change)
to test whether differential FP divergence correlates with bioactivity
fold-change at scale.

Usage:
    python scripts/atlas_v8_chembl.py [--limit 500] [--min-fold 3.0]
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

from zynerji_chirality.chirality.fingerprint import (
    chirality_differential_fingerprint,
    per_center_differential_fingerprint,
    fingerprint_similarity,
)


@dataclass
class DifferentialPair:
    smiles1: str
    smiles2: str
    fold_change: float
    chembl_id_a: str
    chembl_id_b: str


def load_differential_pairs(
    hits_path: str = "chembl_work/screening_hits.json",
    limit: int = 500,
    min_fold: float = 3.0,
    max_fold: float = 1000.0,
    max_atoms: int = 60,
) -> list[DifferentialPair]:
    """Load differential activity pairs from screening hits JSON."""
    with open(hits_path) as f:
        hits = json.load(f)

    pairs = []
    for h in hits:
        if h["hit_type"] != "differential":
            continue
        fc = h["priority_score"]
        if fc < min_fold or fc > max_fold:
            continue

        smi_a = h["mol_a_smiles"]
        smi_b = h["mol_b_smiles"]

        # Skip very large molecules (slow to fingerprint)
        from rdkit import Chem
        mol_a = Chem.MolFromSmiles(smi_a)
        mol_b = Chem.MolFromSmiles(smi_b)
        if mol_a is None or mol_b is None:
            continue
        if mol_a.GetNumHeavyAtoms() > max_atoms or mol_b.GetNumHeavyAtoms() > max_atoms:
            continue

        pairs.append(DifferentialPair(
            smiles1=smi_a,
            smiles2=smi_b,
            fold_change=fc,
            chembl_id_a=h["mol_a_chembl_id"],
            chembl_id_b=h["mol_b_chembl_id"],
        ))

        if len(pairs) >= limit:
            break

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--min-fold", type=float, default=3.0)
    parser.add_argument("--max-fold", type=float, default=1000.0)
    parser.add_argument("--max-atoms", type=int, default=60)
    parser.add_argument("--hits", default="chembl_work/screening_hits.json")
    args = parser.parse_args()

    print(f"Loading pairs (limit={args.limit}, fold={args.min_fold}-{args.max_fold}x)...")
    pairs = load_differential_pairs(
        args.hits, args.limit, args.min_fold, args.max_fold, args.max_atoms,
    )
    print(f"Loaded {len(pairs)} pairs")

    div_global = []
    log_fc = []
    n_fail = 0
    t0 = time.time()

    for i, p in enumerate(pairs):
        try:
            fp1 = chirality_differential_fingerprint(p.smiles1, mode="pure_diff")
            fp2 = chirality_differential_fingerprint(p.smiles2, mode="pure_diff")

            n1, n2 = np.linalg.norm(fp1), np.linalg.norm(fp2)
            if n1 < 1e-8 or n2 < 1e-8:
                # One is achiral — skip
                n_fail += 1
                continue

            sim = fingerprint_similarity(fp1, fp2)
            div_global.append(1 - sim)
            log_fc.append(np.log10(p.fold_change))

        except Exception:
            n_fail += 1
            continue

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(pairs)} ({elapsed:.1f}s, {n_fail} failures)")

    elapsed = time.time() - t0
    print(f"\nProcessed {len(div_global)} pairs in {elapsed:.1f}s ({n_fail} failures)")

    if len(div_global) < 10:
        print("Too few valid pairs for correlation analysis")
        return

    div_global = np.array(div_global)
    log_fc = np.array(log_fc)

    rho, p_rho = spearmanr(div_global, log_fc)
    r, p_r = pearsonr(div_global, log_fc)

    print(f"\n{'='*60}")
    print(f"Atlas v8.0 — ChEMBL Large-N Results (n={len(div_global)})")
    print(f"{'='*60}")
    print(f"Spearman rho = {rho:.4f} (p = {p_rho:.2e})")
    print(f"Pearson r    = {r:.4f} (p = {p_r:.2e})")
    print(f"\nDivergence: mean={div_global.mean():.4f}, std={div_global.std():.4f}")
    print(f"            min={div_global.min():.4f}, max={div_global.max():.4f}")
    print(f"log10(FC):  mean={log_fc.mean():.4f}, std={log_fc.std():.4f}")

    # Quartile analysis
    q25, q50, q75 = np.percentile(log_fc, [25, 50, 75])
    low = div_global[log_fc <= q25]
    high = div_global[log_fc >= q75]
    print(f"\nQuartile analysis:")
    print(f"  Low FC (log10 <= {q25:.2f}): mean div = {low.mean():.4f} (n={len(low)})")
    print(f"  High FC (log10 >= {q75:.2f}): mean div = {high.mean():.4f} (n={len(high)})")

    # Write results
    results = {
        "n_pairs": len(div_global),
        "n_failures": n_fail,
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "pearson_r": float(r),
        "pearson_p": float(p_r),
        "div_mean": float(div_global.mean()),
        "div_std": float(div_global.std()),
        "elapsed_s": elapsed,
    }
    out_path = Path(args.hits).parent / "atlas_v8_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
