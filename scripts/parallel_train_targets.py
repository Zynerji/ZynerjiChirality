#!/usr/bin/env python3
"""Launch parallel target model training across N workers.

Two-phase strategy to eliminate GPU contention:

Phase 1 — Pre-compute (single CUDA context):
    A single process fingerprints all unique SMILES from all remaining target
    datasets and saves results to fp_precomputed.pkl. This takes ~2–3 hours
    depending on how many molecules need CUDA 3D embedding. Progress is logged
    and checkpointed every 2,000 molecules.

Phase 2 — Parallel train (N CPU-only workers):
    N workers split the remaining targets and train sklearn models. Each worker
    loads fp_precomputed.pkl into its in-process fingerprint cache at startup,
    so all fingerprint calls are cache hits — no GPU, no redundant computation.

Also reconciles done_targets.json with actual PKL files before starting, so
interrupted parallel runs don't re-train already-saved models.

Usage:
    python scripts/parallel_train_targets.py \\
        --enriched chembl_work/enriched_pairs.json \\
        --work-dir targets_work \\
        --workers 8

    # Skip pre-compute if fp_precomputed.pkl already exists
    python scripts/parallel_train_targets.py \\
        --enriched chembl_work/enriched_pairs.json \\
        --work-dir targets_work \\
        --workers 8 \\
        --skip-precompute
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _reconcile_done_targets(work_dir: Path) -> set[str]:
    """Sync done_targets.json with actual PKL files (ground truth).

    Workers may save PKL files without flushing done_targets.json if interrupted.
    This ensures we don't re-train targets that already have a model file.
    """
    done_path = work_dir / "done_targets.json"
    done_ids: set[str] = set()

    if done_path.exists():
        with open(done_path) as f:
            raw = json.load(f)
        done_ids = set(raw if isinstance(raw, list) else raw.keys())

    # PKL files are the authoritative source
    pkl_ids: set[str] = set()
    for pkl in work_dir.glob("target_*.pkl"):
        # target_CHEMBL12345.pkl -> CHEMBL12345
        tid = pkl.stem[len("target_"):]
        pkl_ids.add(tid)

    if pkl_ids - done_ids:
        missing_from_json = pkl_ids - done_ids
        logger.info(
            "Reconciling done_targets: %d in JSON + %d PKL-only = %d total",
            len(done_ids), len(missing_from_json), len(done_ids | pkl_ids),
        )
        done_ids = done_ids | pkl_ids
        with open(done_path, "w") as f:
            json.dump(sorted(done_ids), f)

    logger.info("Done targets after reconciliation: %d", len(done_ids))
    return done_ids


def _run_precompute(work_dir: Path, enriched: str, nbits: int, min_records: int) -> Path:
    """Run precompute_fingerprints.py as a subprocess and wait for completion."""
    out_file = work_dir / "fp_precomputed.pkl"
    cmd = [
        sys.executable,
        "scripts/precompute_fingerprints.py",
        "--work-dir", str(work_dir),
        "--nbits", str(nbits),
        "--min-records", str(min_records),
        "--resume",
    ]

    log_file = work_dir / "precompute.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    logger.info("Starting pre-compute fingerprints → %s", log_file)
    logger.info("Command: %s", " ".join(cmd))

    t_start = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, env=env)

    logger.info("Pre-compute PID %d started. Monitoring every 60s...", proc.pid)

    last_size = 0
    while proc.poll() is None:
        time.sleep(60)
        elapsed = time.time() - t_start
        # Show tail of log for progress
        try:
            with open(log_file) as lf:
                lines = lf.readlines()
            # Print new lines since last check
            new_lines = lines[last_size:]
            last_size = len(lines)
            for line in new_lines[-3:]:
                logger.info("[precompute] %s", line.rstrip())
        except Exception:
            pass

        size_mb = out_file.stat().st_size / 1e6 if out_file.exists() else 0
        logger.info("[%.0fs] Pre-compute running... pkl=%.1fMB", elapsed, size_mb)

    rc = proc.returncode
    elapsed = time.time() - t_start
    if rc != 0:
        logger.error("Pre-compute failed (rc=%d) after %.0fs — check %s", rc, elapsed, log_file)
        sys.exit(1)

    logger.info("Pre-compute done in %.0fs (%.1fmin)", elapsed, elapsed / 60)
    if out_file.exists():
        logger.info("fp_precomputed.pkl: %.1f MB", out_file.stat().st_size / 1e6)
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Parallel target model training (2-phase)")
    parser.add_argument("--enriched", required=True, help="Path to enriched_pairs.json")
    parser.add_argument("--work-dir", default="targets_work", help="Output directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--nbits", type=int, default=128, help="Fingerprint bits")
    parser.add_argument("--min-records", type=int, default=20, help="Min records per target")
    parser.add_argument("--skip-precompute", action="store_true",
                        help="Skip Phase 1 if fp_precomputed.pkl already exists")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Load target datasets
    datasets_path = work_dir / "target_datasets.json"
    if not datasets_path.exists():
        logger.error("target_datasets.json not found in %s — run full training first", work_dir)
        sys.exit(1)

    with open(datasets_path) as f:
        datasets = json.load(f)

    # --- Reconcile done_targets with PKL files ---
    done_ids = _reconcile_done_targets(work_dir)

    # --- Phase 1: Pre-compute fingerprints ---
    fp_precomputed = work_dir / "fp_precomputed.pkl"
    if args.skip_precompute and fp_precomputed.exists():
        logger.info("Skipping pre-compute (--skip-precompute, fp_precomputed.pkl exists: %.1f MB)",
                    fp_precomputed.stat().st_size / 1e6)
    else:
        fp_precomputed = _run_precompute(work_dir, args.enriched, args.nbits, args.min_records)

    # --- Phase 2: Parallel training ---
    eligible = [
        d for d in datasets
        if len(d["records"]) >= args.min_records
        and d.get("target_chembl_id", "") not in done_ids
    ]
    eligible_sorted = sorted(eligible, key=lambda x: -len(x["records"]))
    remaining_ids = [d["target_chembl_id"] for d in eligible_sorted]

    logger.info("Remaining targets: %d (splitting into %d workers)", len(remaining_ids), args.workers)

    if not remaining_ids:
        logger.info("All targets already trained!")
        return

    # Interleaved split for load balance
    n = args.workers
    chunks = [remaining_ids[i::n] for i in range(n)]
    chunks = [c for c in chunks if c]

    logger.info("Chunks: %s", [len(c) for c in chunks])

    # Write chunk files
    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_file = work_dir / f"worker_{i}_targets.json"
        with open(chunk_file, "w") as f:
            json.dump(chunk, f)
        chunk_files.append(chunk_file)
        logger.info("Worker %d: %d targets (e.g. %s)", i, len(chunk), chunk[:3])

    # Launch workers (CPU-only — GPU was used in pre-compute, no contention now)
    done_path = work_dir / "done_targets.json"
    procs = []
    for i, chunk_file in enumerate(chunk_files):
        log_file = work_dir / f"worker_{i}.log"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # All cores divided evenly — workers are CPU-only sklearn training
        blas_threads = max(1, 128 // n - 2)
        env["OMP_NUM_THREADS"] = str(blas_threads)
        env["OPENBLAS_NUM_THREADS"] = str(blas_threads)
        env["MKL_NUM_THREADS"] = str(blas_threads)

        cmd = [
            sys.executable,
            "scripts/train_target_models.py",
            "--enriched", args.enriched,
            "--work-dir", args.work_dir,
            "--nbits", str(args.nbits),
            "--min-records", str(args.min_records),
            "--resume",
            "--no-gpu",                              # No GPU: all FPs are pre-computed
            "--target-ids-file", str(chunk_file),
            "--precomputed-fps", str(fp_precomputed),
        ]

        with open(log_file, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, env=env)

        logger.info("Worker %d started (PID %d, CPU-only) → %s", i, proc.pid, log_file)
        procs.append((i, proc))

    logger.info("All %d workers launched (CPU-only, FPs pre-loaded). Monitoring...", len(procs))

    # Monitor progress
    t_start = time.time()
    while True:
        time.sleep(30)
        alive = [(i, p) for i, p in procs if p.poll() is None]

        n_done = 0
        if done_path.exists():
            try:
                with open(done_path) as f:
                    raw = json.load(f)
                n_done = len(raw if isinstance(raw, list) else raw.keys())
            except Exception:
                pass

        pkl_count = len(list(work_dir.glob("target_*.pkl")))
        elapsed = time.time() - t_start
        logger.info(
            "[%.0fs] Workers alive: %d/%d | PKL files: %d | done_targets: %d",
            elapsed, len(alive), len(procs), pkl_count, n_done,
        )

        if not alive:
            logger.info("All workers finished!")
            break

    # Final reconcile
    final_done = _reconcile_done_targets(work_dir)
    pkl_files = list(work_dir.glob("target_*.pkl"))
    logger.info("Final PKL models: %d | Done targets: %d", len(pkl_files), len(final_done))

    # Clean up chunk files
    for f in chunk_files:
        f.unlink(missing_ok=True)

    logger.info("Parallel training complete!")
    logger.info("Next: run global model training or start dashboard")


if __name__ == "__main__":
    main()
