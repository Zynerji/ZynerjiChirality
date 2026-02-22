#!/usr/bin/env python3
"""CLI for ingesting SMILES files into the FingerprintStore.

Usage:
    python scripts/ingest.py input.smi --db fingerprints.db
    python scripts/ingest.py input.smi --db fingerprints.db --workers 4
"""

from __future__ import annotations

import argparse
import sys
import time

from zynerji_chirality.chirality.detector import HelixChiralityDetector
from zynerji_chirality.chirality.fingerprint import chirality_fingerprint
from zynerji_chirality.db.store import FingerprintStore


def main():
    parser = argparse.ArgumentParser(description="Ingest SMILES into FingerprintStore")
    parser.add_argument("input", help="Input file (one SMILES per line, optional tab-separated name)")
    parser.add_argument("--db", default="fingerprints.db", help="Output SQLite database path")
    parser.add_argument("--nbits", type=int, default=128, help="Fingerprint bit length")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    args = parser.parse_args()

    # Read input
    molecules = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            smiles = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else None
            molecules.append((smiles, name))

    print(f"Loaded {len(molecules)} molecules from {args.input}")

    detector = HelixChiralityDetector()
    store = FingerprintStore(args.db)

    t0 = time.time()
    n_success = 0
    n_fail = 0

    for i, (smiles, name) in enumerate(molecules):
        try:
            fp = chirality_fingerprint(smiles, nbits=args.nbits)
            result = detector.detect(smiles)
            store.add(
                smiles=smiles,
                fingerprint=fp,
                chirality_score=result.chirality_score,
                chirality_sign=result.chirality_sign,
                name=name,
                nbits=args.nbits,
            )
            n_success += 1
        except Exception as e:
            n_fail += 1
            print(f"  FAIL {smiles}: {e}", file=sys.stderr)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(molecules)}...")

    elapsed = time.time() - t0
    print(f"\nDone: {n_success} ingested, {n_fail} failed, {elapsed:.1f}s")
    print(f"Database: {args.db} ({store.count()} molecules)")
    store.close()


if __name__ == "__main__":
    main()
