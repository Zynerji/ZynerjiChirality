"""GPU-accelerated batch fingerprinter for failed/deferred molecules.

Reads failed_molecules.jsonl, processes each molecule through the CUDA
conformer pipeline, computes chirality fingerprints, and stores results
in FingerprintStore.

Drop-in replacement for PairFingerprinter's output format.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class GPUFingerprinter:
    """Batch fingerprinter using CUDA distance geometry for large molecules.

    Processes molecules that were deferred by the CPU pipeline (SMILES > 300
    chars that hang ETKDG). Uses CUDAConformerPipeline for 3D embedding,
    then standard chirality fingerprint computation.
    """

    def __init__(
        self,
        store=None,
        nbits: int = 128,
        n_restarts: int = 4,
        device: int = 0,
    ):
        """Initialize GPU fingerprinter.

        Parameters
        ----------
        store : FingerprintStore, optional
            SQLite fingerprint database. If None, results are returned but not stored.
        nbits : int
            Fingerprint length.
        n_restarts : int
            Number of DG random restarts per molecule.
        device : int
            CUDA device index.
        """
        self.store = store
        self.nbits = nbits

        from zynerji_chirality.cuda.pipeline import CUDAConformerPipeline
        self.pipeline = CUDAConformerPipeline(
            device=device,
            n_restarts=n_restarts,
        )

    def fingerprint_one(self, smiles: str, chembl_id: str = "") -> dict:
        """Fingerprint a single molecule using CUDA pipeline.

        Returns dict compatible with FingerprintStore.batch_add_fast().
        """
        from zynerji_chirality.chirality.fingerprint import chirality_fingerprint
        from zynerji_chirality.chirality.detector import HelixChiralityDetector

        # Use CUDA pipeline for 3D embedding
        mol = self.pipeline.embed_molecule(smiles)

        # Compute chirality fingerprint from the embedded molecule
        fp = chirality_fingerprint(mol, nbits=self.nbits)

        # Detect chirality
        detector = HelixChiralityDetector()
        # Pass the already-embedded mol to avoid re-embedding
        result = detector.detect(mol)

        return {
            "smiles": smiles,
            "fingerprint": fp.tolist() if hasattr(fp, "tolist") else list(fp),
            "chirality_score": result.chirality_score,
            "chirality_sign": result.chirality_sign,
            "name": chembl_id,
            "metadata": {"chembl_id": chembl_id or "", "source": "cuda_pipeline"},
        }

    def process_failed_file(
        self,
        jsonl_path: str,
        batch_size: int = 10,
    ) -> dict:
        """Process all molecules from a failed_molecules.jsonl file.

        Parameters
        ----------
        jsonl_path : str
            Path to failed_molecules.jsonl.
        batch_size : int
            Number of molecules to process before DB commit.

        Returns
        -------
        dict
            Statistics: n_total, n_success, n_fail, elapsed_seconds.
        """
        molecules = self._load_molecules(jsonl_path)
        n_total = len(molecules)
        logger.info("Processing %d failed/deferred molecules with CUDA", n_total)

        t0 = time.time()
        n_success = 0
        n_fail = 0
        still_failed = []
        batch_entries = []

        for i, mol_info in enumerate(molecules):
            smiles = mol_info["smiles"]
            chembl_id = mol_info.get("chembl_id", "")

            try:
                entry = self.fingerprint_one(smiles, chembl_id)
                batch_entries.append(entry)
                n_success += 1
            except Exception as e:
                n_fail += 1
                still_failed.append({
                    "_failed": True,
                    "smiles": smiles,
                    "chembl_id": chembl_id,
                    "reason": f"cuda_failed: {str(e)[:200]}",
                    "smiles_len": len(smiles),
                })
                logger.warning(
                    "CUDA failed for %s (len=%d): %s",
                    chembl_id, len(smiles), str(e)[:100],
                )

            # Batch commit
            if len(batch_entries) >= batch_size:
                self._commit_batch(batch_entries)
                batch_entries = []

            # Progress
            if (i + 1) % 10 == 0 or i == n_total - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 0.001)
                logger.info(
                    "CUDA progress: %d/%d (%.1f mol/s), success=%d, fail=%d",
                    i + 1, n_total, rate, n_success, n_fail,
                )

        # Final batch commit
        if batch_entries:
            self._commit_batch(batch_entries)

        # Write still-failed
        if still_failed:
            fail_path = Path(jsonl_path).parent / "cuda_still_failed.jsonl"
            with open(fail_path, "w") as f:
                for entry in still_failed:
                    f.write(json.dumps(entry) + "\n")
            logger.info("Wrote %d still-failed to %s", len(still_failed), fail_path)

        elapsed = time.time() - t0
        stats = {
            "n_total": n_total,
            "n_success": n_success,
            "n_fail": n_fail,
            "elapsed_seconds": elapsed,
            "rate_per_second": n_total / max(elapsed, 0.001),
            "success_rate": n_success / max(n_total, 1),
        }

        logger.info(
            "CUDA fingerprinting complete: %d/%d success (%.1f%%), %.1fs",
            n_success, n_total, stats["success_rate"] * 100, elapsed,
        )

        return stats

    def _commit_batch(self, entries: list[dict]):
        """Commit a batch of entries to the database."""
        if self.store and entries:
            try:
                self.store.batch_add_fast(entries, nbits=self.nbits)
                logger.debug("Committed %d entries to DB", len(entries))
            except Exception as e:
                logger.error("DB commit failed: %s", e)

    @staticmethod
    def _load_molecules(jsonl_path: str) -> list[dict]:
        """Load molecules from JSONL file."""
        molecules = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                mol = json.loads(line)
                molecules.append(mol)
        return molecules
