#!/usr/bin/env python3
"""COD Crystal Structure Chirality Pipeline.

Full pipeline: query COD for chiral space groups -> download CIFs ->
enrich with chirality analysis -> fingerprint -> screen for hits.

Usage:
    # Full pipeline
    python scripts/cod_pipeline.py --db cod_screen.db --workers 4

    # Resume from checkpoint
    python scripts/cod_pipeline.py --db cod_screen.db --resume

    # Screen only (after enrichment)
    python scripts/cod_pipeline.py --db cod_screen.db --screen-only

    # Limit structures per space group (for testing)
    python scripts/cod_pipeline.py --db cod_screen.db --limit-per-sg 10
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
logger = logging.getLogger("cod_pipeline")


def stage_discover(args) -> list:
    """Stage 1: Query COD for chiral crystal structures."""
    from zynerji_chirality.cod.download import CODDownloader

    structures_path = Path(args.work_dir) / "structures.json"

    if args.resume and structures_path.exists():
        structures = CODDownloader.load_structures(str(structures_path))
        logger.info("Loaded %d structures from checkpoint", len(structures))
        return structures

    downloader = CODDownloader(delay=args.delay)
    structures = list(downloader.search_chiral_structures(
        limit_per_sg=args.limit_per_sg,
    ))
    logger.info("Discovered %d chiral structures", len(structures))

    CODDownloader.save_structures(structures, str(structures_path))
    return structures


def stage_enrich(args, structures: list) -> list:
    """Stage 2: Download CIFs and enrich with chirality analysis."""
    from zynerji_chirality.cod.enricher import CODEnricher

    enriched_path = Path(args.work_dir) / "enriched_structures.json"
    progress_path = Path(args.work_dir) / "cod_progress.json"
    cif_dir = str(Path(args.work_dir) / "cifs")

    enricher = CODEnricher(cutoff=args.cutoff, skip_cif=args.skip_cif)
    enrichments = enricher.enrich_batch(
        structures,
        cif_dir=cif_dir,
        chunk_size=args.enrich_chunk_size,
        checkpoint_path=str(enriched_path) if args.resume else None,
        progress_path=str(progress_path),
    )

    # Final save
    enricher.save_enriched(enrichments, str(enriched_path))
    logger.info("Enriched %d structures", len(enrichments))
    return enrichments


def stage_fingerprint(args, enrichments: list) -> None:
    """Stage 3: Fingerprint structures and store in DB."""
    from zynerji_chirality.cod.fingerprinter import CrystalFingerprinter
    from zynerji_chirality.db.store import FingerprintStore

    store = FingerprintStore(args.db)
    fingerprinter = CrystalFingerprinter(
        store=store,
        nbits=args.nbits,
        n_workers=args.workers,
    )

    checkpoint_path = str(Path(args.work_dir) / "fp_checkpoint.json")
    progress_path = str(Path(args.work_dir) / "cod_progress.json")
    stats = fingerprinter.fingerprint_structures(
        enrichments,
        checkpoint_path=checkpoint_path if args.resume else None,
        progress_path=progress_path,
    )

    logger.info("Fingerprinting stats: %s", json.dumps(stats, indent=2))
    store.close()


def stage_screen(args, enrichments: list) -> None:
    """Stage 4: Screen structures and generate report."""
    from zynerji_chirality.cod.screen import CrystalScreen

    screen = CrystalScreen(enrichments)
    hits = screen.screen_all()

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
            "cod_id": h.structure.cod_id,
            "formula": h.structure.formula,
            "space_group": h.structure.space_group,
            "sg_number": h.structure.sg_number,
        })
    with open(hits_path, "w") as f:
        json.dump(hits_data, f, indent=2)
    logger.info("Saved %d hits to %s", len(hits), hits_path)


def main():
    parser = argparse.ArgumentParser(
        description="COD Crystal Structure Chirality Pipeline",
    )
    parser.add_argument("--db", default="cod_screen.db", help="SQLite database path")
    parser.add_argument(
        "--work-dir", default="cod_work", help="Working directory for checkpoints"
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--screen-only", action="store_true", help="Run screening only")
    parser.add_argument("--nbits", type=int, default=128, help="Fingerprint bits")
    parser.add_argument(
        "--cutoff", type=float, default=3.0, help="Distance cutoff (Angstroms)"
    )
    parser.add_argument(
        "--limit-per-sg",
        type=int,
        default=500,
        help="Max structures per space group",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5, help="Seconds between API requests"
    )
    parser.add_argument(
        "--enrich-chunk-size",
        type=int,
        default=100,
        help="Checkpoint interval for enrichment",
    )
    parser.add_argument(
        "--skip-cif",
        action="store_true",
        help="Skip CIF download, use lattice parameters only (faster, no network needed)",
    )

    args = parser.parse_args()
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if args.screen_only:
        from zynerji_chirality.cod.enricher import CODEnricher
        enriched_path = Path(args.work_dir) / "enriched_structures.json"
        if not enriched_path.exists():
            logger.error("No enriched data found. Run full pipeline first.")
            sys.exit(1)
        enrichments = CODEnricher.load_enriched(str(enriched_path))
        stage_screen(args, enrichments)
    else:
        logger.info("=== Stage 1: Discover ===")
        structures = stage_discover(args)

        logger.info("=== Stage 2: Enrich ===")
        enrichments = stage_enrich(args, structures)

        logger.info("=== Stage 3: Fingerprint ===")
        stage_fingerprint(args, enrichments)

        logger.info("=== Stage 4: Screen ===")
        stage_screen(args, enrichments)

    elapsed = time.time() - t0
    logger.info("Pipeline complete in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
