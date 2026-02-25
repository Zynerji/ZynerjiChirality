#!/usr/bin/env python3
"""CUDA retry: process all deferred molecules using GPU distance geometry.

Unlike gpu_retry.py (which spawns processes for CPU timeout protection),
this runs directly with the CUDA pipeline (which doesn't hang).

Usage:
    PYTHONPATH=. python3 scripts/cuda_retry.py \
        --failed chembl_work/failed_molecules.jsonl \
        --db chembl_work/chembl_screen.db
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cuda_retry")


def main():
    parser = argparse.ArgumentParser(description="CUDA retry for deferred molecules")
    parser.add_argument("--failed", required=True, help="Path to failed_molecules.jsonl")
    parser.add_argument("--db", help="Path to fingerprint SQLite DB (optional)")
    parser.add_argument("--nbits", type=int, default=128, help="Fingerprint bits")
    parser.add_argument("--restarts", type=int, default=3, help="DG random restarts")
    parser.add_argument("--test", type=int, help="Process only first N molecules")
    args = parser.parse_args()

    # Load molecules
    molecules = []
    with open(args.failed) as f:
        for line in f:
            line = line.strip()
            if line:
                molecules.append(json.loads(line))
    molecules.sort(key=lambda m: m.get("smiles_len", len(m.get("smiles", ""))))

    if args.test:
        molecules = molecules[:args.test]

    logger.info("Loaded %d molecules (sizes: %d-%d chars)",
                len(molecules),
                molecules[0].get("smiles_len", 0) if molecules else 0,
                molecules[-1].get("smiles_len", 0) if molecules else 0)

    # Initialize CUDA pipeline
    from zynerji_chirality.cuda.pipeline import CUDAConformerPipeline
    from zynerji_chirality.chirality.fingerprint import chirality_fingerprint
    from zynerji_chirality.chirality.detector import HelixChiralityDetector

    pipeline = CUDAConformerPipeline(n_restarts=args.restarts)
    detector = HelixChiralityDetector()

    # Optional DB store
    store = None
    if args.db:
        from zynerji_chirality.db.store import FingerprintStore
        store = FingerprintStore(args.db)
        logger.info("DB: %s (%d existing)", args.db, store.count())

    t0 = time.time()
    n_success = 0
    n_fail = 0
    batch_entries = []
    still_failed = []

    for i, mol_info in enumerate(molecules):
        smiles = mol_info["smiles"]
        chembl_id = mol_info.get("chembl_id", "")
        t_mol = time.time()

        try:
            # CUDA 3D embedding
            mol = pipeline.embed_molecule(smiles)
            embed_time = time.time() - t_mol

            # Chirality fingerprint
            fp = chirality_fingerprint(mol, nbits=args.nbits)

            # Chirality detection
            result = detector.detect(mol)

            entry = {
                "smiles": smiles,
                "fingerprint": fp,
                "chirality_score": result.chirality_score,
                "chirality_sign": result.chirality_sign,
                "name": chembl_id,
                "metadata": {"chembl_id": chembl_id or "", "source": "cuda_pipeline"},
            }
            batch_entries.append(entry)
            n_success += 1
            mol_time = time.time() - t_mol

            if (i + 1) % 10 == 0 or i < 5:
                logger.info(
                    "[%d/%d] %s: %d atoms, %.2fs (embed=%.2fs), score=%.4f",
                    i + 1, len(molecules), chembl_id,
                    mol.GetNumAtoms(), mol_time, embed_time,
                    result.chirality_score,
                )

        except Exception as e:
            n_fail += 1
            mol_time = time.time() - t_mol
            still_failed.append({
                "_failed": True,
                "smiles": smiles,
                "chembl_id": chembl_id,
                "reason": f"cuda: {str(e)[:200]}",
                "smiles_len": len(smiles),
            })
            logger.warning(
                "[%d/%d] FAILED %s (len=%d, %.2fs): %s",
                i + 1, len(molecules), chembl_id,
                len(smiles), mol_time, str(e)[:80],
            )

        # Batch insert every 50 molecules
        if store and len(batch_entries) >= 50:
            store.batch_add_fast(batch_entries, nbits=args.nbits)
            logger.info("DB commit: %d entries", len(batch_entries))
            batch_entries = []

        # Progress every 50 molecules
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.001)
            logger.info(
                "=== Progress: %d/%d (%.1f mol/s), success=%d, fail=%d, %.0fs ===",
                i + 1, len(molecules), rate, n_success, n_fail, elapsed,
            )

    # Final batch
    if store and batch_entries:
        store.batch_add_fast(batch_entries, nbits=args.nbits)
        logger.info("Final DB commit: %d entries", len(batch_entries))

    # Write still-failed
    if still_failed:
        fail_path = Path(args.failed).parent / "cuda_still_failed.jsonl"
        with open(fail_path, "w") as f:
            for entry in still_failed:
                f.write(json.dumps(entry) + "\n")
        logger.info("Wrote %d still-failed to %s", len(still_failed), fail_path)

    elapsed = time.time() - t0
    logger.info(
        "\n=== COMPLETE ===\n"
        "  Total: %d molecules\n"
        "  Success: %d (%.1f%%)\n"
        "  Failed: %d\n"
        "  Time: %.1fs (%.2f mol/s)\n"
        "  DB count: %s",
        len(molecules), n_success, n_success / max(len(molecules), 1) * 100,
        n_fail, elapsed, len(molecules) / max(elapsed, 0.001),
        store.count() if store else "N/A",
    )


if __name__ == "__main__":
    main()
