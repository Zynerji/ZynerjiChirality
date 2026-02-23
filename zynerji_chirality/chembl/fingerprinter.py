"""Fingerprint enantiomer pairs with ZynerjiChirality.

Takes discovered EnantiomerPairs, computes chirality-aware fingerprints
for each molecule, and stores them in the FingerprintStore for similarity search.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from zynerji_chirality.chembl.pairs import EnantiomerPair
from zynerji_chirality.chirality.detector import HelixChiralityDetector
from zynerji_chirality.chirality.fingerprint import chirality_fingerprint, batch_fingerprint
from zynerji_chirality.db.store import FingerprintStore

logger = logging.getLogger(__name__)


class PairFingerprinter:
    """Fingerprint enantiomer pairs and store in FingerprintStore.

    Parameters
    ----------
    store : FingerprintStore
        Database to store fingerprints.
    detector : HelixChiralityDetector
        Chirality detector for scoring.
    nbits : int
        Fingerprint bit length.
    n_workers : int
        Number of parallel workers for fingerprint computation.
    """

    def __init__(
        self,
        store: FingerprintStore,
        detector: HelixChiralityDetector | None = None,
        nbits: int = 128,
        n_workers: int = 4,
    ):
        self.store = store
        self.detector = detector or HelixChiralityDetector()
        self.nbits = nbits
        self.n_workers = n_workers

    def fingerprint_pairs(
        self,
        pairs: list[EnantiomerPair],
        checkpoint_interval: int = 100,
        checkpoint_path: str | None = None,
    ) -> dict:
        """Fingerprint all molecules in pairs, store in FingerprintStore.

        Parameters
        ----------
        pairs : list[EnantiomerPair]
            Pairs to fingerprint.
        checkpoint_interval : int
            Save checkpoint every N pairs.
        checkpoint_path : str, optional
            Path to checkpoint file for resume on failure.

        Returns
        -------
        dict
            Stats with success/fail counts and timing.
        """
        # Determine resume point
        start_idx = 0
        if checkpoint_path and Path(checkpoint_path).exists():
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            start_idx = ckpt.get("last_completed", 0)
            logger.info("Resuming from pair %d", start_idx)

        # Collect all unique SMILES
        all_smiles = []
        smiles_to_info: dict[str, dict] = {}
        for pair in pairs[start_idx:]:
            for mol in [pair.mol_a, pair.mol_b]:
                if mol.canonical_smiles not in smiles_to_info:
                    all_smiles.append(mol.canonical_smiles)
                    smiles_to_info[mol.canonical_smiles] = {
                        "chembl_id": mol.chembl_id,
                        "smiles": mol.canonical_smiles,
                    }

        logger.info(
            "Fingerprinting %d unique molecules from %d pairs (workers=%d)",
            len(all_smiles), len(pairs) - start_idx, self.n_workers,
        )

        t0 = time.time()
        n_success = 0
        n_fail = 0

        # Process in batches for checkpointing
        batch_size = checkpoint_interval * 2  # 2 molecules per pair
        for batch_start in range(0, len(all_smiles), batch_size):
            batch_smiles = all_smiles[batch_start:batch_start + batch_size]

            # Compute fingerprints in parallel
            fps = batch_fingerprint(
                batch_smiles,
                nbits=self.nbits,
                n_workers=self.n_workers,
            )

            # Detect chirality and store
            entries = []
            for smiles, fp in zip(batch_smiles, fps):
                if fp is None:
                    n_fail += 1
                    continue

                try:
                    result = self.detector.detect(smiles)
                    entries.append({
                        "smiles": smiles,
                        "fingerprint": fp,
                        "chirality_score": result.chirality_score,
                        "chirality_sign": result.chirality_sign,
                        "name": smiles_to_info[smiles].get("chembl_id"),
                        "metadata": {
                            "chembl_id": smiles_to_info[smiles].get("chembl_id", ""),
                            "source": "chembl_pipeline",
                        },
                    })
                    n_success += 1
                except Exception as e:
                    logger.debug("Failed to detect %s: %s", smiles, e)
                    n_fail += 1

            # Batch insert
            if entries:
                self.store.batch_add_fast(entries, nbits=self.nbits)

            # Progress report
            total_done = batch_start + len(batch_smiles)
            elapsed = time.time() - t0
            rate = total_done / max(elapsed, 0.001)
            logger.info(
                "Progress: %d/%d (%.0f mol/s), success=%d, fail=%d",
                total_done, len(all_smiles), rate, n_success, n_fail,
            )

            # Save checkpoint
            if checkpoint_path:
                pairs_done = start_idx + (total_done // 2)
                with open(checkpoint_path, "w") as f:
                    json.dump({
                        "last_completed": pairs_done,
                        "n_success": n_success,
                        "n_fail": n_fail,
                        "elapsed": elapsed,
                    }, f)

        elapsed = time.time() - t0
        stats = {
            "n_pairs": len(pairs),
            "n_molecules": len(all_smiles),
            "n_success": n_success,
            "n_fail": n_fail,
            "elapsed_seconds": elapsed,
            "rate_per_second": len(all_smiles) / max(elapsed, 0.001),
            "store_count": self.store.count(),
        }

        logger.info(
            "Fingerprinting complete: %d success, %d fail, %.1fs (%.0f mol/s)",
            n_success, n_fail, elapsed, stats["rate_per_second"],
        )
        return stats
