#!/usr/bin/env python3
"""Train target-specific chirality sensitivity models.

Usage:
    python scripts/train_target_models.py --enriched chembl_work/enriched_pairs.json
    python scripts/train_target_models.py --enriched chembl_work/enriched_pairs.json --resume
    python scripts/train_target_models.py --profile "N[C@@H](C)C(=O)O" "N[C@H](C)C(=O)O" --work-dir targets_work
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _enable_gpu():
    """Try to enable CUDA pipeline for GPU-accelerated 3D embedding.

    Threshold set to 30 atoms: molecules with >= 30 heavy atoms AND
    unassigned stereocenters use GPU DG instead of CPU ETKDG.
    GPU DG is deterministic (0.5-1.8s), CPU ETKDG can hang for minutes on
    complex medium-sized molecules (ring systems, macrocycles, etc.).
    Tiny molecules (< 30 atoms) still use fast CPU ETKDG (~1ms).
    """
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


def main():
    parser = argparse.ArgumentParser(description="Train target chirality models")
    parser.add_argument("--enriched", help="Path to enriched_pairs.json")
    parser.add_argument("--work-dir", default="targets_work", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--nbits", type=int, default=128, help="Fingerprint bits")
    parser.add_argument("--min-records", type=int, default=20, help="Min records per target")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument(
        "--target-ids-file", help="JSON file with list of target IDs to train (for parallel workers)"
    )
    parser.add_argument(
        "--precomputed-fps", help="Path to fp_precomputed.pkl for pre-populating fingerprint cache"
    )
    parser.add_argument(
        "--profile", nargs=2, metavar=("SMILES_A", "SMILES_B"),
        help="Profile a molecule pair for target sensitivity",
    )
    args = parser.parse_args()

    if not args.no_gpu:
        _enable_gpu()

    if args.profile:
        # Profile mode — load models and predict
        from zynerji_chirality.targets.profiler import TargetProfiler

        profiler = TargetProfiler.load_from_directory(args.work_dir)
        profile = profiler.profile(args.profile[0], args.profile[1])

        print(f"\nTarget Sensitivity Profile")
        print(f"  Mol A: {profile.smiles_a}")
        print(f"  Mol B: {profile.smiles_b}")
        print(f"  Sensitive targets: {profile.n_sensitive}")
        print(f"  Top target: {profile.top_target} ({profile.top_fold_change:.1f}x)")
        print()

        for entry in profile.entries[:20]:
            flag = "*" if entry.predicted_fold_change >= 3.0 else " "
            print(
                f"  {flag} {entry.target_chembl_id:20s} "
                f"fold={entry.predicted_fold_change:6.1f}x "
                f"conf={entry.confidence:.3f} "
                f"[{entry.model_source}]"
            )
        return

    if not args.enriched:
        parser.error("--enriched is required for training")

    from zynerji_chirality.targets.trainer import TargetModelTrainer

    only_targets = None
    if args.target_ids_file:
        import json as _json
        with open(args.target_ids_file) as f:
            only_targets = _json.load(f)
        logger.info("Worker mode: %d target IDs from %s", len(only_targets), args.target_ids_file)

    trainer = TargetModelTrainer(
        work_dir=args.work_dir,
        nbits=args.nbits,
        min_records=args.min_records,
        cv_folds=args.cv_folds,
    )
    result = trainer.train_all(
        args.enriched,
        resume=args.resume,
        only_targets=only_targets,
        precomputed_fps_path=args.precomputed_fps,
    )

    print(f"\nTraining Complete")
    print(f"  Targets trained: {result.n_targets_trained}")
    print(f"  Targets skipped: {result.n_targets_skipped}")
    print(f"  Elapsed: {result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()
