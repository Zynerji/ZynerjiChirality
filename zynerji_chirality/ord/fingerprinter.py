"""Fingerprint enantioselective reactions and store in FingerprintStore.

Module-level picklable workers, parallel processing, and batch DB inserts.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from zynerji_chirality.ord.enricher import ReactionEnrichment
from zynerji_chirality.db.store import FingerprintStore

logger = logging.getLogger(__name__)


def _fingerprint_reaction_one(args: tuple) -> dict | None:
    """Module-level picklable worker: fingerprint one reaction.

    Parameters (as tuple):
        reaction_id, catalyst_smiles, substrate_smiles, nbits
    """
    reaction_id, catalyst_smiles, substrate_smiles, nbits = args
    try:
        from zynerji_chirality.chirality.fingerprint import chirality_fingerprint

        cat_fp = None
        sub_fp = None

        if catalyst_smiles:
            cat_fp = chirality_fingerprint(catalyst_smiles, nbits=nbits)
        if substrate_smiles:
            sub_fp = chirality_fingerprint(substrate_smiles, nbits=nbits)

        if cat_fp is not None and sub_fp is not None:
            combined = np.concatenate([
                cat_fp, sub_fp, cat_fp * sub_fp, np.abs(cat_fp - sub_fp)
            ])
        elif cat_fp is not None:
            combined = np.concatenate([cat_fp, np.zeros(nbits * 3)])
        elif sub_fp is not None:
            combined = np.concatenate([np.zeros(nbits), sub_fp, np.zeros(nbits * 2)])
        else:
            return {"_failed": True, "reaction_id": reaction_id, "reason": "no SMILES"}

        return {
            "smiles": reaction_id,  # Use reaction_id as key
            "fingerprint": combined,
            "chirality_score": float(np.sum(combined > 0) / len(combined)),
            "chirality_sign": 0.0,
            "name": reaction_id,
            "metadata": {
                "reaction_id": reaction_id,
                "catalyst": catalyst_smiles or "",
                "substrate": substrate_smiles or "",
                "source": "ord_pipeline",
            },
        }
    except Exception as e:
        return {"_failed": True, "reaction_id": reaction_id, "reason": str(e)}


class ReactionFingerprinter:
    """Fingerprint reactions and store in FingerprintStore.

    Parameters
    ----------
    store : FingerprintStore
        Database for fingerprint storage.
    nbits : int
        Fingerprint dimensionality per component.
    n_workers : int
        Number of parallel workers.
    """

    def __init__(
        self,
        store: FingerprintStore,
        nbits: int = 128,
        n_workers: int = 4,
    ):
        self.store = store
        self.nbits = nbits
        self.n_workers = n_workers

    def fingerprint_reactions(
        self,
        enrichments: list[ReactionEnrichment],
        checkpoint_path: str | None = None,
        checkpoint_interval: int = 100,
        progress_path: str | None = None,
    ) -> dict:
        """Fingerprint enriched reactions and store in DB.

        Parameters
        ----------
        enrichments : list
            Enriched reactions with catalyst/substrate data.
        checkpoint_path : str, optional
            Path for checkpoint JSON.
        checkpoint_interval : int
            Save after this many reactions.
        progress_path : str, optional
            Path for lightweight progress JSON.

        Returns
        -------
        dict
            Stats: n_total, n_success, n_failed, elapsed_seconds.
        """
        # Load checkpoint
        done_ids: set[str] = set()
        if checkpoint_path and Path(checkpoint_path).exists():
            with open(checkpoint_path) as f:
                cp = json.load(f)
            done_ids = set(cp.get("done_ids", []))
            logger.info("Resuming: %d already fingerprinted", len(done_ids))

        valid = [
            e for e in enrichments
            if e.combined_fp is not None
            and e.reaction.reaction_id not in done_ids
        ]
        logger.info(
            "Fingerprinting %d reactions (%d skipped)", len(valid), len(done_ids)
        )

        t0 = time.monotonic()
        n_success = 0
        n_failed = 0
        batch_results = []

        for i, enrichment in enumerate(valid):
            try:
                rxn = enrichment.reaction
                fp = enrichment.combined_fp

                batch_results.append({
                    "smiles": rxn.reaction_id,
                    "fingerprint": fp,
                    "chirality_score": rxn.catalyst_chirality_score,
                    "chirality_sign": rxn.catalyst_chirality_sign,
                    "name": rxn.reaction_id,
                    "metadata": {
                        "reaction_id": rxn.reaction_id,
                        "catalyst": rxn.catalyst_smiles or "",
                        "substrate": rxn.substrate_smiles or "",
                        "ee_percent": rxn.ee_percent,
                        "source": "ord_pipeline",
                    },
                })
                done_ids.add(rxn.reaction_id)
                n_success += 1
            except Exception:
                n_failed += 1

            if len(batch_results) >= checkpoint_interval or i == len(valid) - 1:
                if batch_results:
                    self.store.batch_add_fast(batch_results)
                    batch_results = []

                if checkpoint_path:
                    with open(checkpoint_path, "w") as f:
                        json.dump({"done_ids": list(done_ids)}, f)

                if progress_path:
                    progress = {
                        "stage": "fingerprint",
                        "reactions_done": len(done_ids),
                        "reactions_total": len(enrichments),
                        "n_success": n_success,
                        "n_failed": n_failed,
                        "percent_complete": round(
                            len(done_ids) / max(len(enrichments), 1) * 100, 1
                        ),
                    }
                    with open(progress_path, "w") as f:
                        json.dump(progress, f)

        elapsed = time.monotonic() - t0
        stats = {
            "n_total": len(valid),
            "n_success": n_success,
            "n_failed": n_failed,
            "elapsed_seconds": round(elapsed, 1),
        }
        logger.info("Fingerprinting complete: %s", stats)
        return stats
