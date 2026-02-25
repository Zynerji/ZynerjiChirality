#!/usr/bin/env python3
"""ORD Enantioselectivity Screening Pipeline.

Full pipeline: download ORD data -> extract ee reactions ->
enrich with chirality analysis -> fingerprint -> screen for hits.

Usage:
    # Full pipeline (requires ord-schema + local .pb.gz files)
    python scripts/ord_pipeline.py --data-dir /path/to/ord --db ord_screen.db

    # From pre-extracted JSON
    python scripts/ord_pipeline.py --json-file reactions.json --db ord_screen.db

    # Resume from checkpoint
    python scripts/ord_pipeline.py --db ord_screen.db --resume

    # Screen only
    python scripts/ord_pipeline.py --db ord_screen.db --screen-only

    # Train ee model after screening
    python scripts/ord_pipeline.py --db ord_screen.db --train-model
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
logger = logging.getLogger("ord_pipeline")


def stage_discover(args) -> list:
    """Stage 1: Download/parse ORD and extract ee reactions."""
    from zynerji_chirality.ord.download import ORDDownloader
    from zynerji_chirality.ord.extractor import ORDExtractor

    reactions_path = Path(args.work_dir) / "reactions.json"

    if args.resume and reactions_path.exists():
        reactions = ORDDownloader.load_reactions(str(reactions_path))
        logger.info("Loaded %d reactions from checkpoint", len(reactions))
    elif args.json_file:
        downloader = ORDDownloader()
        raw = list(downloader.iterate_from_json(args.json_file))
        logger.info("Loaded %d raw reactions from JSON", len(raw))
        ORDDownloader.save_reactions(raw, str(reactions_path))
        reactions = raw
    elif args.data_dir:
        downloader = ORDDownloader()
        raw = list(downloader.iterate_from_pb_dir(args.data_dir))
        logger.info("Extracted %d ee reactions from protobuf", len(raw))
        ORDDownloader.save_reactions(raw, str(reactions_path))
        reactions = raw
    else:
        logger.error(
            "Must provide --data-dir (protobuf), --json-file, or --resume"
        )
        sys.exit(1)

    return reactions


def stage_enrich(args, reactions: list) -> list:
    """Stage 2: Enrich reactions with chirality analysis + fingerprints."""
    from zynerji_chirality.ord.download import ORDReaction
    from zynerji_chirality.ord.extractor import ORDExtractor
    from zynerji_chirality.ord.enricher import ORDEnricher

    enriched_path = Path(args.work_dir) / "enriched_reactions.json"
    progress_path = Path(args.work_dir) / "ord_progress.json"

    # First extract ee reactions if raw ORDReaction
    if reactions and hasattr(reactions[0], "ee_percent") and not hasattr(reactions[0], "catalyst_chirality_score"):
        extractor = ORDExtractor()
        ee_reactions = extractor.extract_ee_reactions(reactions)
        logger.info("Extracted %d ee reactions", len(ee_reactions))
    else:
        ee_reactions = reactions

    enricher = ORDEnricher(nbits=args.nbits)
    enrichments = enricher.enrich_batch(
        ee_reactions,
        chunk_size=args.enrich_chunk_size,
        checkpoint_path=str(enriched_path) if args.resume else None,
        progress_path=str(progress_path),
    )

    enricher.save_enriched(enrichments, str(enriched_path))
    logger.info("Enriched %d reactions", len(enrichments))
    return enrichments


def stage_fingerprint(args, enrichments: list) -> None:
    """Stage 3: Fingerprint reactions and store in DB."""
    from zynerji_chirality.ord.fingerprinter import ReactionFingerprinter
    from zynerji_chirality.db.store import FingerprintStore

    store = FingerprintStore(args.db)
    fingerprinter = ReactionFingerprinter(
        store=store,
        nbits=args.nbits,
        n_workers=args.workers,
    )

    checkpoint_path = str(Path(args.work_dir) / "fp_checkpoint.json")
    progress_path = str(Path(args.work_dir) / "ord_progress.json")
    stats = fingerprinter.fingerprint_reactions(
        enrichments,
        checkpoint_path=checkpoint_path if args.resume else None,
        progress_path=progress_path,
    )

    logger.info("Fingerprinting stats: %s", json.dumps(stats, indent=2))
    store.close()


def stage_screen(args, enrichments: list) -> None:
    """Stage 4: Screen reactions and generate report."""
    from zynerji_chirality.ord.screen import CatalystScreen, train_ee_model
    from zynerji_chirality.db.store import FingerprintStore

    store = None
    predictor = None

    if Path(args.db).exists():
        store = FingerprintStore(args.db)

    # Optionally train model
    if args.train_model:
        try:
            predictor = train_ee_model(enrichments)
            logger.info("Trained ee prediction model")
        except Exception as e:
            logger.warning("Could not train model: %s", e)

    screen = CatalystScreen(enrichments)
    hits = screen.screen_all(store=store, predictor=predictor)

    # Report
    report_path = str(Path(args.work_dir) / "screening_report.txt")
    report = screen.report(hits, output=report_path)
    print(report)

    # Save hits as JSON
    hits_path = str(Path(args.work_dir) / "screening_hits.json")
    hits_data = []
    for h in hits:
        hits_data.append({
            "hit_type": h.hit_type,
            "priority_score": h.priority_score,
            "description": h.description,
            "reaction_id": h.reaction.reaction_id,
            "catalyst": h.reaction.catalyst_smiles,
            "substrate": h.reaction.substrate_smiles,
            "ee_percent": h.reaction.ee_percent,
        })
    with open(hits_path, "w") as f:
        json.dump(hits_data, f, indent=2)
    logger.info("Saved %d hits to %s", len(hits), hits_path)

    if store:
        store.close()


def main():
    parser = argparse.ArgumentParser(
        description="ORD Enantioselectivity Screening Pipeline",
    )
    parser.add_argument("--db", default="ord_screen.db", help="SQLite database path")
    parser.add_argument(
        "--work-dir", default="ord_work", help="Working directory for checkpoints"
    )
    parser.add_argument(
        "--data-dir", default=None, help="ORD protobuf data directory"
    )
    parser.add_argument(
        "--json-file", default=None, help="Pre-extracted reactions JSON"
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--screen-only", action="store_true", help="Run screening only")
    parser.add_argument(
        "--train-model", action="store_true", help="Train ee prediction model"
    )
    parser.add_argument("--nbits", type=int, default=128, help="Fingerprint bits")
    parser.add_argument(
        "--enrich-chunk-size",
        type=int,
        default=200,
        help="Checkpoint interval for enrichment",
    )

    args = parser.parse_args()
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if args.screen_only:
        from zynerji_chirality.ord.enricher import ORDEnricher
        enriched_path = Path(args.work_dir) / "enriched_reactions.json"
        if not enriched_path.exists():
            logger.error("No enriched data found. Run full pipeline first.")
            sys.exit(1)
        enrichments = ORDEnricher.load_enriched(str(enriched_path))
        stage_screen(args, enrichments)
    else:
        logger.info("=== Stage 1: Discover ===")
        reactions = stage_discover(args)

        logger.info("=== Stage 2: Enrich ===")
        enrichments = stage_enrich(args, reactions)

        logger.info("=== Stage 3: Fingerprint ===")
        stage_fingerprint(args, enrichments)

        logger.info("=== Stage 4: Screen ===")
        stage_screen(args, enrichments)

    elapsed = time.time() - t0
    logger.info("Pipeline complete in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
