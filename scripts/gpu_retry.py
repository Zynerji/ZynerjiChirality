#!/usr/bin/env python3
"""GPU/HPC retry: process failed/deferred molecules on high-performance VM.

Reads failed_molecules.jsonl (produced by PairFingerprinter), processes each
molecule with per-process hard timeout (handles RDKit C++ ETKDG hangs), and
stores results in the fingerprint database.

Usage:
    # Quick test (first 10 molecules, 60s timeout)
    python gpu_retry.py --failed failed_molecules.jsonl --test 10

    # Full run on 64-core VM
    python gpu_retry.py --failed failed_molecules.jsonl --db chembl_screen.db --workers 32 --timeout 120

    # With CUDA distance geometry (when available)
    python gpu_retry.py --failed failed_molecules.jsonl --db chembl_screen.db --cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gpu_retry")


def _worker_fn(smiles: str, chembl_id: str, nbits: int, result_queue):
    """Worker process: fingerprint + detect one molecule."""
    try:
        from zynerji_chirality.chirality.fingerprint import chirality_fingerprint
        from zynerji_chirality.chirality.detector import HelixChiralityDetector

        fp = chirality_fingerprint(smiles, nbits=nbits)
        detector = HelixChiralityDetector()
        result = detector.detect(smiles)

        result_queue.put({
            "smiles": smiles,
            "fingerprint": fp.tolist() if hasattr(fp, "tolist") else list(fp),
            "chirality_score": result.chirality_score,
            "chirality_sign": result.chirality_sign,
            "name": chembl_id,
            "metadata": {"chembl_id": chembl_id or "", "source": "gpu_retry"},
        })
    except Exception as e:
        result_queue.put({
            "_failed": True,
            "smiles": smiles,
            "chembl_id": chembl_id,
            "reason": str(e)[:300],
            "smiles_len": len(smiles),
        })


def _worker_fn_cuda(smiles: str, chembl_id: str, nbits: int, result_queue):
    """Worker process using CUDA distance geometry for 3D embedding."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from zynerji_chirality.cuda.pipeline import CUDAConformerPipeline
        from zynerji_chirality.chirality.fingerprint import chirality_fingerprint
        from zynerji_chirality.chirality.detector import HelixChiralityDetector
        from zynerji_chirality.core.mol_graph import smiles_to_mol3d

        # Try CUDA pipeline for 3D embedding
        pipeline = CUDAConformerPipeline()
        mol = pipeline.embed_molecule(smiles)

        # Use the CUDA-embedded mol for fingerprinting
        fp = chirality_fingerprint(mol, nbits=nbits)
        detector = HelixChiralityDetector()
        result = detector.detect(smiles)

        result_queue.put({
            "smiles": smiles,
            "fingerprint": fp.tolist() if hasattr(fp, "tolist") else list(fp),
            "chirality_score": result.chirality_score,
            "chirality_sign": result.chirality_sign,
            "name": chembl_id,
            "metadata": {"chembl_id": chembl_id or "", "source": "gpu_retry_cuda"},
        })
    except Exception as e:
        result_queue.put({
            "_failed": True,
            "smiles": smiles,
            "chembl_id": chembl_id,
            "reason": f"cuda: {str(e)[:280]}",
            "smiles_len": len(smiles),
        })


def process_molecule_with_timeout(
    smiles: str,
    chembl_id: str,
    nbits: int = 128,
    timeout_sec: int = 120,
    use_cuda: bool = False,
) -> dict:
    """Process one molecule with hard timeout via process termination.

    Spawns a fresh process for each molecule. If the process doesn't complete
    within timeout_sec, it's terminated (SIGTERM) then killed (SIGKILL).
    This handles RDKit C++ ETKDG hangs that block Python signal handlers.
    """
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    worker = _worker_fn_cuda if use_cuda else _worker_fn
    p = ctx.Process(target=worker, args=(smiles, chembl_id, nbits, q))

    t0 = time.time()
    p.start()
    p.join(timeout=timeout_sec)
    elapsed = time.time() - t0

    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join()
        return {
            "_failed": True,
            "smiles": smiles,
            "chembl_id": chembl_id,
            "reason": f"timeout_{timeout_sec}s",
            "smiles_len": len(smiles),
            "elapsed": elapsed,
        }

    if not q.empty():
        result = q.get_nowait()
        result["elapsed"] = elapsed
        return result

    return {
        "_failed": True,
        "smiles": smiles,
        "chembl_id": chembl_id,
        "reason": "process_died_no_result",
        "smiles_len": len(smiles),
        "elapsed": elapsed,
    }


def parallel_retry(
    molecules: list[dict],
    nbits: int = 128,
    n_workers: int = 32,
    timeout_sec: int = 120,
    use_cuda: bool = False,
    store_path: str | None = None,
) -> dict:
    """Process molecules in parallel batches with per-molecule hard timeout."""
    n_total = len(molecules)
    logger.info(
        "Processing %d molecules (workers=%d, timeout=%ds, cuda=%s)",
        n_total, n_workers, timeout_sec, use_cuda,
    )

    results = []
    n_success = 0
    n_fail = 0
    n_timeout = 0
    t0 = time.time()

    # Process in batches of n_workers
    for batch_start in range(0, n_total, n_workers):
        batch = molecules[batch_start : batch_start + n_workers]
        batch_results = []

        # Launch all processes in this batch
        ctx = multiprocessing.get_context("spawn")
        procs = []
        queues = []

        for mol_info in batch:
            q = ctx.Queue()
            worker = _worker_fn_cuda if use_cuda else _worker_fn
            p = ctx.Process(
                target=worker,
                args=(mol_info["smiles"], mol_info.get("chembl_id", ""), nbits, q),
            )
            p.start()
            procs.append(p)
            queues.append(q)

        # Wait for all with timeout
        deadline = time.time() + timeout_sec + 10
        for p in procs:
            remaining = max(1, deadline - time.time())
            p.join(timeout=remaining)

        # Collect results
        for i, (p, q) in enumerate(zip(procs, queues)):
            mol_info = batch[i]
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join()
                r = {
                    "_failed": True,
                    "smiles": mol_info["smiles"],
                    "chembl_id": mol_info.get("chembl_id", ""),
                    "reason": f"timeout_{timeout_sec}s",
                    "smiles_len": len(mol_info["smiles"]),
                }
                n_timeout += 1
                n_fail += 1
            elif not q.empty():
                r = q.get_nowait()
                if r.get("_failed"):
                    n_fail += 1
                else:
                    n_success += 1
            else:
                r = {
                    "_failed": True,
                    "smiles": mol_info["smiles"],
                    "chembl_id": mol_info.get("chembl_id", ""),
                    "reason": "process_died",
                    "smiles_len": len(mol_info["smiles"]),
                }
                n_fail += 1

            batch_results.append(r)

        results.extend(batch_results)

        # Batch insert successful results into DB
        if store_path:
            entries = [r for r in batch_results if not r.get("_failed")]
            if entries:
                _batch_insert(store_path, entries, nbits)

        # Progress
        done = batch_start + len(batch)
        elapsed = time.time() - t0
        logger.info(
            "Progress: %d/%d (%.1fs), success=%d, fail=%d, timeout=%d",
            done, n_total, elapsed, n_success, n_fail, n_timeout,
        )

    elapsed = time.time() - t0
    stats = {
        "n_total": n_total,
        "n_success": n_success,
        "n_fail": n_fail,
        "n_timeout": n_timeout,
        "elapsed_seconds": elapsed,
        "success_rate": n_success / max(n_total, 1),
    }
    logger.info(
        "Complete: %d/%d success (%.1f%%), %d timeout, %.1fs",
        n_success, n_total, stats["success_rate"] * 100, n_timeout, elapsed,
    )

    # Write still-failed molecules
    if store_path:
        still_failed = [r for r in results if r.get("_failed")]
        if still_failed:
            fail_path = Path(store_path).parent / "still_failed_molecules.jsonl"
            with open(fail_path, "w") as f:
                for r in still_failed:
                    f.write(json.dumps(r) + "\n")
            logger.info("Wrote %d still-failed molecules to %s", len(still_failed), fail_path)

    return stats


def _batch_insert(db_path: str, entries: list[dict], nbits: int):
    """Insert successful results into FingerprintStore."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from zynerji_chirality.db.store import FingerprintStore
        store = FingerprintStore(db_path)
        store.batch_add_fast(entries, nbits=nbits)
    except Exception as e:
        logger.error("DB insert failed: %s", e)


def load_failed_molecules(jsonl_path: str) -> list[dict]:
    """Load molecules from failed_molecules.jsonl."""
    molecules = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mol = json.loads(line)
            molecules.append(mol)
    logger.info(
        "Loaded %d molecules from %s (reasons: %s)",
        len(molecules),
        jsonl_path,
        _count_reasons(molecules),
    )
    return molecules


def _count_reasons(molecules: list[dict]) -> str:
    reasons = {}
    for m in molecules:
        r = m.get("reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))


def main():
    parser = argparse.ArgumentParser(description="Retry failed/deferred molecules on HPC/GPU VM")
    parser.add_argument("--failed", required=True, help="Path to failed_molecules.jsonl")
    parser.add_argument("--db", help="Path to fingerprint SQLite database")
    parser.add_argument("--workers", type=int, default=32, help="Parallel workers (default: 32)")
    parser.add_argument("--timeout", type=int, default=120, help="Per-molecule timeout in seconds")
    parser.add_argument("--nbits", type=int, default=128, help="Fingerprint bits")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA distance geometry")
    parser.add_argument("--test", type=int, help="Only process first N molecules (quick test)")
    parser.add_argument("--sort-by-length", action="store_true", help="Process shortest first")
    args = parser.parse_args()

    molecules = load_failed_molecules(args.failed)

    if args.sort_by_length:
        molecules.sort(key=lambda m: m.get("smiles_len", len(m.get("smiles", ""))))

    if args.test:
        molecules = molecules[: args.test]
        logger.info("Test mode: processing first %d molecules", args.test)

    if not molecules:
        logger.info("No molecules to process")
        return

    stats = parallel_retry(
        molecules,
        nbits=args.nbits,
        n_workers=args.workers,
        timeout_sec=args.timeout,
        use_cuda=args.cuda,
        store_path=args.db,
    )

    print("\n=== Results ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
