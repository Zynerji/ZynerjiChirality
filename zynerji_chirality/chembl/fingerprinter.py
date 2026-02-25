"""Fingerprint enantiomer pairs with ZynerjiChirality.

Takes discovered EnantiomerPairs, computes chirality-aware fingerprints
for each molecule, and stores them in the FingerprintStore for similarity search.

Large peptides (SMILES > _MAX_SMILES_LEN chars) are deferred to failed_molecules.jsonl
for retry on GPU VM with CUDA distance geometry kernel.
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

# SMILES longer than this are deferred to GPU retry (ETKDG hangs on large peptides)
_MAX_SMILES_LEN = 300


def _detect_and_fingerprint_one(args: tuple) -> dict | None:
    """Module-level worker: fingerprint + detect in one shot (picklable)."""
    smiles, chembl_id, nbits = args
    try:
        from zynerji_chirality.chirality.fingerprint import chirality_fingerprint
        from zynerji_chirality.chirality.detector import HelixChiralityDetector
        fp = chirality_fingerprint(smiles, nbits=nbits)
        detector = HelixChiralityDetector()
        result = detector.detect(smiles)
        return {
            "smiles": smiles,
            "fingerprint": fp,
            "chirality_score": result.chirality_score,
            "chirality_sign": result.chirality_sign,
            "name": chembl_id,
            "metadata": {"chembl_id": chembl_id or "", "source": "chembl_pipeline"},
        }
    except Exception:
        return {"_failed": True, "smiles": smiles, "chembl_id": chembl_id,
                "reason": "exception", "smiles_len": len(smiles)}


class PairFingerprinter:
    """Fingerprint enantiomer pairs and store in FingerprintStore."""

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
        """Fingerprint all molecules in pairs, store in FingerprintStore."""
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

        # Skip molecules already in DB (use store's connection, not a new one,
        # so :memory: DBs work correctly in tests)
        existing_smiles = set(
            r[0] for r in self.store._conn.execute(
                "SELECT smiles FROM molecules"
            ).fetchall()
        )
        before_skip = len(all_smiles)
        all_smiles = [s for s in all_smiles if s not in existing_smiles]

        # Separate large peptides (defer to GPU retry)
        work_dir = Path(checkpoint_path).parent if checkpoint_path else Path(".")
        failed_log_path = work_dir / "failed_molecules.jsonl"

        cpu_smiles = []
        n_deferred = 0
        for smi in all_smiles:
            if len(smi) > _MAX_SMILES_LEN:
                n_deferred += 1
                info = smiles_to_info[smi]
                with open(failed_log_path, "a") as f:
                    f.write(json.dumps({
                        "_failed": True,
                        "smiles": smi,
                        "chembl_id": info.get("chembl_id"),
                        "reason": "deferred_gpu",
                        "smiles_len": len(smi),
                    }) + "\n")
            else:
                cpu_smiles.append(smi)

        logger.info(
            "Fingerprinting %d molecules (workers=%d), skipped %d in DB, deferred %d large to GPU (%s)",
            len(cpu_smiles), self.n_workers,
            before_skip - len(all_smiles), n_deferred, failed_log_path,
        )

        t0 = time.time()
        n_success = 0
        n_fail = 0

        batch_size = 1000
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")

        for batch_start in range(0, len(cpu_smiles), batch_size):
            batch_smiles = cpu_smiles[batch_start:batch_start + batch_size]

            work_items = [
                (smi, smiles_to_info[smi].get("chembl_id"), self.nbits)
                for smi in batch_smiles
            ]

            with ctx.Pool(processes=self.n_workers) as pool:
                results = pool.map(_detect_and_fingerprint_one, work_items)

            # Collect results and log failures
            entries = []
            for r in results:
                if r is None:
                    n_fail += 1
                elif r.get("_failed"):
                    n_fail += 1
                    with open(failed_log_path, "a") as f:
                        f.write(json.dumps(r) + "\n")
                else:
                    entries.append(r)
                    n_success += 1

            # Batch insert
            if entries:
                self.store.batch_add_fast(entries, nbits=self.nbits)

            # Progress report
            total_done = batch_start + len(batch_smiles)
            elapsed = time.time() - t0
            rate = total_done / max(elapsed, 0.001)
            logger.info(
                "Progress: %d/%d (%.0f mol/s), success=%d, fail=%d, deferred=%d",
                total_done, len(cpu_smiles), rate, n_success, n_fail, n_deferred,
            )

            # Save checkpoint
            if checkpoint_path:
                pairs_done = start_idx + (total_done // 2)
                with open(checkpoint_path, "w") as f:
                    json.dump({
                        "last_completed": pairs_done,
                        "n_success": n_success,
                        "n_fail": n_fail,
                        "n_deferred": n_deferred,
                        "elapsed": elapsed,
                    }, f)

        elapsed = time.time() - t0
        stats = {
            "n_pairs": len(pairs),
            "n_molecules": len(cpu_smiles),
            "n_success": n_success,
            "n_fail": n_fail,
            "n_deferred": n_deferred,
            "elapsed_seconds": elapsed,
            "rate_per_second": len(cpu_smiles) / max(elapsed, 0.001),
            "store_count": self.store.count(),
        }

        logger.info(
            "Fingerprinting complete: %d success, %d fail, %d deferred to GPU, %.1fs (%.0f mol/s)",
            n_success, n_fail, n_deferred, elapsed, stats["rate_per_second"],
        )
        return stats
