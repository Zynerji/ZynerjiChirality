#!/usr/bin/env python3
"""ChEMBL Enantiomer Screening Pipeline.

Full pipeline: download ChEMBL -> discover enantiomer pairs ->
enrich with bioactivity -> fingerprint -> screen for hits.

Usage:
    # Full pipeline
    python scripts/chembl_pipeline.py --db chembl_screen.db --workers 4

    # Resume from checkpoint
    python scripts/chembl_pipeline.py --db chembl_screen.db --resume

    # Use a pre-downloaded SMILES file instead of ChEMBL download
    python scripts/chembl_pipeline.py --db chembl_screen.db --smiles-file compounds.smi

    # Screen only (after ingestion)
    python scripts/chembl_pipeline.py --db chembl_screen.db --screen-only

    # Query mode
    python scripts/chembl_pipeline.py --db chembl_screen.db --query "N[C@@H](C)C(=O)O"
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
    stream=sys.stdout,
)
logger = logging.getLogger("chembl_pipeline")


def stage_discover(args) -> list:
    """Stage 1: Download ChEMBL and discover enantiomer pairs."""
    from zynerji_chirality.chembl.download import ChEMBLDownloader
    from zynerji_chirality.chembl.pairs import EnantiomerPairFinder

    pairs_path = Path(args.work_dir) / "pairs.json"

    # Check for existing pairs
    if args.resume and pairs_path.exists():
        finder = EnantiomerPairFinder()
        pairs = finder.load_pairs(str(pairs_path))
        logger.info("Loaded %d pairs from checkpoint", len(pairs))
        return pairs

    # Download and iterate
    downloader = ChEMBLDownloader(version=args.chembl_version)

    if args.smiles_file:
        logger.info("Reading from SMILES file: %s", args.smiles_file)
        molecules = list(downloader.iterate_from_smiles_file(args.smiles_file))
    else:
        logger.info("Downloading ChEMBL (version=%s)...", args.chembl_version)
        molecules = list(downloader.iterate_chiral_molecules())

    logger.info("Found %d chiral molecules", len(molecules))

    # Discover pairs
    finder = EnantiomerPairFinder()
    pairs = finder.find_pairs(molecules, include_diastereomers=True)

    # Save checkpoint
    finder.save_pairs(pairs, str(pairs_path))

    return pairs


def stage_enrich(args, pairs: list) -> list:
    """Stage 2: Fetch bioactivity data for pairs."""
    from zynerji_chirality.chembl.activity import ActivityEnricher

    enriched_path = Path(args.work_dir) / "enriched_pairs.json"

    # Check for existing enrichment
    if args.resume and enriched_path.exists():
        enricher = ActivityEnricher()
        pair_activities = enricher.load_enriched(str(enriched_path))
        logger.info("Loaded %d enriched pairs from checkpoint", len(pair_activities))
        return pair_activities

    enricher = ActivityEnricher()
    pair_activities = enricher.enrich_pairs(pairs, batch_size=args.batch_size)

    # Save checkpoint
    enricher.save_enriched(pair_activities, str(enriched_path))

    return pair_activities


def stage_fingerprint(args, pairs: list) -> None:
    """Stage 3: Fingerprint all molecules in pairs."""
    from zynerji_chirality.chirality.detector import HelixChiralityDetector
    from zynerji_chirality.chembl.fingerprinter import PairFingerprinter
    from zynerji_chirality.db.store import FingerprintStore

    store = FingerprintStore(args.db)
    detector = HelixChiralityDetector()
    fingerprinter = PairFingerprinter(
        store=store,
        detector=detector,
        nbits=args.nbits,
        n_workers=args.workers,
    )

    checkpoint_path = str(Path(args.work_dir) / "fp_checkpoint.json")
    stats = fingerprinter.fingerprint_pairs(
        pairs,
        checkpoint_interval=100,
        checkpoint_path=checkpoint_path if args.resume else None,
    )

    logger.info("Fingerprinting stats: %s", json.dumps(stats, indent=2))
    store.close()


def stage_screen(args, pair_activities: list) -> None:
    """Stage 4: Run all screens and generate report."""
    from zynerji_chirality.chembl.screen import EnantiomerScreen
    from zynerji_chirality.db.store import FingerprintStore

    store = FingerprintStore(args.db)
    screen = EnantiomerScreen(store=store, pair_activities=pair_activities)

    # Run screens
    hits = screen.screen_all(min_fold_change=args.min_fold_change)

    # Generate report
    report_path = str(Path(args.work_dir) / "screening_report.txt")
    report = screen.report(hits, output=report_path)
    print(report)

    # Also save hits as JSON
    hits_path = str(Path(args.work_dir) / "screening_hits.json")
    hits_data = []
    for hit in hits:
        hits_data.append({
            "hit_type": hit.hit_type,
            "priority_score": hit.priority_score,
            "description": hit.description,
            "mol_a_chembl_id": hit.pair.mol_a.chembl_id,
            "mol_b_chembl_id": hit.pair.mol_b.chembl_id,
            "mol_a_smiles": hit.pair.mol_a.canonical_smiles,
            "mol_b_smiles": hit.pair.mol_b.canonical_smiles,
            "relationship": hit.pair.relationship,
        })
    with open(hits_path, "w") as f:
        json.dump(hits_data, f, indent=2)
    logger.info("Saved %d hits to %s", len(hits), hits_path)

    store.close()


def stage_query(args) -> None:
    """Query mode: find enantiomers similar to a query molecule."""
    from zynerji_chirality.chembl.activity import ActivityEnricher
    from zynerji_chirality.chembl.screen import EnantiomerScreen
    from zynerji_chirality.db.store import FingerprintStore

    enriched_path = Path(args.work_dir) / "enriched_pairs.json"
    if not enriched_path.exists():
        logger.error("No enriched pairs found. Run full pipeline first.")
        sys.exit(1)

    enricher = ActivityEnricher()
    pair_activities = enricher.load_enriched(str(enriched_path))

    store = FingerprintStore(args.db)
    screen = EnantiomerScreen(store=store, pair_activities=pair_activities)

    hits = screen.screen_by_query(args.query, k=args.k)
    report = screen.report(hits)
    print(report)

    store.close()


def main():
    parser = argparse.ArgumentParser(
        description="ChEMBL Enantiomer Screening Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--db", default="chembl_screen.db",
                        help="SQLite database path for fingerprints")
    parser.add_argument("--work-dir", default="chembl_work",
                        help="Working directory for checkpoints and outputs")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--nbits", type=int, default=128,
                        help="Fingerprint bit length")
    parser.add_argument("--chembl-version", default="latest",
                        help="ChEMBL version to download")
    parser.add_argument("--smiles-file", default=None,
                        help="Use SMILES file instead of ChEMBL download")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="API batch size for activity queries")
    parser.add_argument("--min-fold-change", type=float, default=3.0,
                        help="Minimum fold change for differential screening")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of results for query mode")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints")
    parser.add_argument("--screen-only", action="store_true",
                        help="Skip download/discover/fingerprint, just screen")
    parser.add_argument("--query", default=None,
                        help="Query SMILES for similarity search")
    parser.add_argument("--port", type=int, default=None,
                        help="Launch dashboard on this port after pipeline")

    args = parser.parse_args()

    # Create work directory
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if args.query:
        stage_query(args)
        return

    if args.screen_only:
        from zynerji_chirality.chembl.activity import ActivityEnricher
        enriched_path = work_dir / "enriched_pairs.json"
        if not enriched_path.exists():
            logger.error("No enriched pairs found at %s", enriched_path)
            sys.exit(1)
        enricher = ActivityEnricher()
        pair_activities = enricher.load_enriched(str(enriched_path))
        stage_screen(args, pair_activities)
        elapsed = time.time() - t0
        logger.info("Screening complete in %.1fs", elapsed)
        return

    # Full pipeline
    logger.info("=== Stage 1: Discover enantiomer pairs ===")
    pairs = stage_discover(args)
    logger.info("Discovered %d pairs", len(pairs))

    logger.info("=== Stage 2: Enrich with bioactivity data ===")
    pair_activities = stage_enrich(args, pairs)
    logger.info("Enriched %d pairs", len(pair_activities))

    logger.info("=== Stage 3: Fingerprint pairs ===")
    stage_fingerprint(args, pairs)

    logger.info("=== Stage 4: Screen for hits ===")
    stage_screen(args, pair_activities)

    elapsed = time.time() - t0
    logger.info("Pipeline complete in %.1fs", elapsed)

    # Optionally launch dashboard
    if args.port:
        logger.info("Launching dashboard on port %d...", args.port)
        from zynerji_chirality.dashboard.app import create_app
        app = create_app(
            db_path=args.db,
            work_dir=args.work_dir,
        )
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
