#!/usr/bin/env python3
"""Train ADMET chirality-aware property models.

Usage:
    python scripts/train_admet_models.py --enriched chembl_work/enriched_pairs.json
    python scripts/train_admet_models.py --profile "N[C@@H](C)C(=O)O"
    python scripts/train_admet_models.py --compare "N[C@@H](C)C(=O)O" "N[C@H](C)C(=O)O"
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train ADMET chirality models")
    parser.add_argument("--enriched", help="Path to enriched_pairs.json")
    parser.add_argument("--work-dir", default="admet_work", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--nbits", type=int, default=128, help="Fingerprint bits")
    parser.add_argument("--profile", help="Profile a molecule SMILES for ADMET")
    parser.add_argument(
        "--compare", nargs=2, metavar=("SMILES_A", "SMILES_B"),
        help="Differential ADMET between two enantiomers",
    )
    args = parser.parse_args()

    if args.profile:
        from zynerji_chirality.admet.profiler import ADMETProfiler

        profiler = ADMETProfiler.load_from_directory(args.work_dir)
        profile = profiler.profile(args.profile)

        print(f"\nADMET Profile for: {profile.smiles}")
        print(f"  Overall risk: {profile.overall_risk}")
        print(f"  High risk: {profile.n_high_risk}, Moderate: {profile.n_moderate_risk}")
        print()
        for entry in profile.entries:
            risk_flag = "!!" if entry.risk_level == "high_risk" else "! " if entry.risk_level == "moderate_risk" else "  "
            print(
                f"  {risk_flag} {entry.property_name:25s} "
                f"value={entry.predicted_value:8.2f} {entry.units:10s} "
                f"risk={entry.risk_level}"
            )
        return

    if args.compare:
        from zynerji_chirality.admet.profiler import ADMETProfiler
        from zynerji_chirality.admet.differential import DifferentialADMETAnalyzer

        profiler = ADMETProfiler.load_from_directory(args.work_dir)
        analyzer = DifferentialADMETAnalyzer(profiler)
        result = analyzer.analyze(args.compare[0], args.compare[1])

        print(f"\nDifferential ADMET Analysis")
        print(f"  Mol A: {result.smiles_a}")
        print(f"  Mol B: {result.smiles_b}")
        print(f"  Divergent: {result.n_divergent}, Worst: {result.worst_divergence:.1f}x")
        print(f"  Recommendation: {result.recommendation}")
        print()
        for entry in result.entries:
            div_flag = "*" if entry.risk_divergent else " "
            print(
                f"  {div_flag} {entry.property_name:25s} "
                f"A={entry.value_a:8.2f} B={entry.value_b:8.2f} "
                f"fold={entry.fold_difference:5.1f}x "
                f"risk_A={entry.risk_a} risk_B={entry.risk_b}"
            )
        return

    if not args.enriched:
        parser.error("--enriched is required for training")

    from zynerji_chirality.admet.data_loader import ADMETDataLoader, ADMET_PROPERTIES
    from zynerji_chirality.admet.property_model import ADMETPropertyPredictor

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # Load and classify data
    loader = ADMETDataLoader()
    datasets_path = work_dir / "admet_datasets.json"

    if args.resume and datasets_path.exists():
        datasets = loader.load_datasets(datasets_path)
    else:
        datasets = loader.load_from_enriched(args.enriched)
        loader.save_datasets(datasets, datasets_path)

    # Train per-property models
    done_path = work_dir / "done_properties.json"
    done_props: set[str] = set()
    if args.resume and done_path.exists():
        with open(done_path) as f:
            done_props = set(json.load(f))

    n_trained = 0
    for prop_name, dataset in datasets.items():
        if prop_name in done_props:
            logger.info("Skipping %s (already done)", prop_name)
            continue

        if len(dataset.records) < 10:
            logger.info("Skipping %s: only %d records", prop_name, len(dataset.records))
            continue

        try:
            units = ADMET_PROPERTIES.get(prop_name, {}).get("units", "")
            model = ADMETPropertyPredictor(
                property_name=prop_name,
                units=units,
                nbits=args.nbits,
                n_estimators=150,
            )
            model.fit_from_dataset(dataset)
            model.save(work_dir / f"admet_{prop_name}.pkl")

            n_trained += 1
            done_props.add(prop_name)
            with open(done_path, "w") as f:
                json.dump(sorted(done_props), f)

            logger.info("Trained ADMET model: %s (%d records)", prop_name, len(dataset.records))
        except Exception as e:
            logger.warning("Failed to train %s: %s", prop_name, e)

    elapsed = time.time() - start

    report = {
        "n_properties_trained": n_trained,
        "total_properties": len(datasets),
        "elapsed_seconds": round(elapsed, 1),
        "properties": {k: len(v.records) for k, v in datasets.items()},
    }
    with open(work_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nTraining Complete")
    print(f"  Properties trained: {n_trained}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
