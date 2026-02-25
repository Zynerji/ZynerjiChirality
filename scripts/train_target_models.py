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


def main():
    parser = argparse.ArgumentParser(description="Train target chirality models")
    parser.add_argument("--enriched", help="Path to enriched_pairs.json")
    parser.add_argument("--work-dir", default="targets_work", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--nbits", type=int, default=128, help="Fingerprint bits")
    parser.add_argument("--min-records", type=int, default=20, help="Min records per target")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds")
    parser.add_argument(
        "--profile", nargs=2, metavar=("SMILES_A", "SMILES_B"),
        help="Profile a molecule pair for target sensitivity",
    )
    args = parser.parse_args()

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

    trainer = TargetModelTrainer(
        work_dir=args.work_dir,
        nbits=args.nbits,
        min_records=args.min_records,
        cv_folds=args.cv_folds,
    )
    result = trainer.train_all(args.enriched, resume=args.resume)

    print(f"\nTraining Complete")
    print(f"  Targets trained: {result.n_targets_trained}")
    print(f"  Targets skipped: {result.n_targets_skipped}")
    print(f"  Elapsed: {result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()
