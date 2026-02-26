#!/usr/bin/env python3
"""Pre-compute chirality fingerprints for all unique SMILES in target_datasets.json.

Runs a single CUDA context to fingerprint every unique SMILES across all target
datasets, saving results to fp_precomputed.pkl. Parallel training workers can
then load this file to pre-populate their _FP_CACHE — eliminating GPU contention
and redundant computation across workers.

Usage:
    python scripts/precompute_fingerprints.py --work-dir targets_work --resume

Output:
    targets_work/fp_precomputed.pkl  — dict {smiles: np.ndarray (nbits,)}
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _enable_gpu() -> bool:
    """Enable CUDA conformer pipeline (threshold=30 atoms)."""
    try:
        from zynerji_chirality.cuda.pipeline import CUDAConformerPipeline
        from zynerji_chirality.core.mol_graph import set_cuda_pipeline
        pipeline = CUDAConformerPipeline(device=0)
        set_cuda_pipeline(pipeline, gpu_atom_threshold=30)
        logger.info("GPU acceleration ENABLED (CUDA conformer pipeline, threshold=30 atoms)")
        return True
    except Exception as e:
        logger.info("GPU not available (%s), using CPU ETKDG", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute chirality fingerprints")
    parser.add_argument("--work-dir", default="targets_work", help="Directory with target_datasets.json")
    parser.add_argument("--nbits", type=int, default=128, help="Fingerprint bits")
    parser.add_argument("--min-records", type=int, default=20, help="Min records per target")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--resume", action="store_true", help="Load existing pkl and skip already computed")
    parser.add_argument("--checkpoint-every", type=int, default=2000, help="Save checkpoint every N molecules")
    args = parser.parse_args()

    if not args.no_gpu:
        _enable_gpu()

    from zynerji_chirality.chirality.fingerprint import chirality_fingerprint

    work_dir = Path(args.work_dir)
    out_file = work_dir / "fp_precomputed.pkl"
    datasets_path = work_dir / "target_datasets.json"

    if not datasets_path.exists():
        logger.error("target_datasets.json not found in %s — run training first to generate it", work_dir)
        sys.exit(1)

    with open(datasets_path) as f:
        datasets = json.load(f)

    # Collect all unique SMILES across eligible targets
    all_smiles: set[str] = set()
    n_eligible = 0
    for d in datasets:
        if len(d["records"]) < args.min_records:
            continue
        n_eligible += 1
        for r in d["records"]:
            s_a = r.get("smiles_a", "")
            s_b = r.get("smiles_b", "")
            if s_a:
                all_smiles.add(s_a)
            if s_b:
                all_smiles.add(s_b)

    all_smiles_sorted = sorted(all_smiles)
    logger.info("Eligible targets: %d | Total unique SMILES: %d", n_eligible, len(all_smiles_sorted))

    # Load existing cache if resuming
    cache: dict = {}
    if args.resume and out_file.exists():
        with open(out_file, "rb") as f:
            cache = pickle.load(f)
        logger.info("Loaded %d existing fingerprints from %s", len(cache), out_file)

    remaining = [s for s in all_smiles_sorted if s not in cache]
    logger.info("To compute: %d | Already cached: %d", len(remaining), len(cache))

    if not remaining:
        logger.info("All fingerprints already precomputed!")
        return

    t_start = time.time()
    n_done = 0
    n_fail = 0
    log_every = 500

    for i, smiles in enumerate(remaining):
        try:
            fp = chirality_fingerprint(smiles, nbits=args.nbits)
            cache[smiles] = fp
            n_done += 1
        except Exception as e:
            logger.debug("Failed SMILES %s: %s", smiles[:40], e)
            n_fail += 1

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0.0
            eta_s = (len(remaining) - i - 1) / rate if rate > 0 else float("inf")
            logger.info(
                "[%d/%d] done=%d fail=%d rate=%.1f/s ETA=%.0fmin",
                i + 1, len(remaining), n_done, n_fail, rate, eta_s / 60,
            )

        if (i + 1) % args.checkpoint_every == 0:
            with open(out_file, "wb") as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Checkpoint: %d fingerprints saved to %s", len(cache), out_file)

    # Final save
    with open(out_file, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - t_start
    rate = n_done / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Done! %d fingerprints computed in %.0fs (%.1f/s) | %d failures",
        n_done, elapsed, rate, n_fail,
    )
    logger.info("Saved to %s (%.1f MB)", out_file, out_file.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
