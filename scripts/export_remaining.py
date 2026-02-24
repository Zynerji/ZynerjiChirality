#!/usr/bin/env python3
"""Export un-fingerprinted molecules to JSONL for GPU processing.

Reads enriched_pairs.json, queries DB for completed fingerprints,
writes remaining SMILES to JSONL (compatible with cuda_retry.py).

Usage:
    python3 scripts/export_remaining.py \
        --pairs chembl_work/enriched_pairs.json \
        --db chembl_work/chembl_screen.db \
        --output chembl_work/remaining_molecules.jsonl
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export remaining molecules for GPU fingerprinting")
    parser.add_argument("--pairs", required=True, help="Path to enriched_pairs.json")
    parser.add_argument("--db", required=True, help="Path to fingerprint SQLite DB")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    # Load all unique SMILES from enriched pairs
    print(f"Loading pairs from {args.pairs}...")
    t0 = time.time()
    with open(args.pairs) as f:
        pairs = json.load(f)

    all_smiles = {}  # smiles -> chembl_id
    for entry in pairs:
        # Nested structure: entry["pair"]["mol_a"/"mol_b"]
        pair_data = entry.get("pair", entry)
        for mol_key in ("mol_a", "mol_b"):
            mol = pair_data.get(mol_key, {})
            smi = mol.get("smiles") or mol.get("canonical_smiles")
            chembl_id = mol.get("chembl_id", "")
            if smi and smi not in all_smiles:
                all_smiles[smi] = chembl_id

    print(f"  Found {len(all_smiles)} unique SMILES from {len(pairs)} pairs ({time.time()-t0:.1f}s)")

    # Query DB for already-fingerprinted molecules
    print(f"Querying DB {args.db}...")
    conn = sqlite3.connect(args.db)

    # Get all canonical SMILES that have fingerprints
    done = set()
    cursor = conn.execute("""
        SELECT m.canonical_smiles FROM molecules m
        INNER JOIN fingerprints f ON f.mol_id = m.id
    """)
    for row in cursor:
        done.add(row[0])

    # Also check by raw SMILES (in case canonical differs)
    cursor = conn.execute("""
        SELECT m.smiles FROM molecules m
        INNER JOIN fingerprints f ON f.mol_id = m.id
    """)
    for row in cursor:
        done.add(row[0])

    conn.close()
    print(f"  {len(done)} molecules already fingerprinted in DB")

    # Filter remaining
    remaining = []
    for smi, cid in all_smiles.items():
        if smi not in done:
            remaining.append({
                "smiles": smi,
                "chembl_id": cid,
                "smiles_len": len(smi),
            })

    # Sort by size (smallest first — fastest on GPU)
    remaining.sort(key=lambda m: m["smiles_len"])
    print(f"  {len(remaining)} molecules remaining (need fingerprinting)")

    if remaining:
        print(f"  Size range: {remaining[0]['smiles_len']}-{remaining[-1]['smiles_len']} chars")

    # Write JSONL
    with open(args.output, "w") as f:
        for entry in remaining:
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(remaining)} molecules to {args.output}")
    print(f"Run: PYTHONPATH=. python3 scripts/cuda_retry.py --failed {args.output} --db {args.db}")


if __name__ == "__main__":
    main()
