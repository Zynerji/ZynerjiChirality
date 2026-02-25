#!/usr/bin/env python3
"""Retry fingerprinting for molecules in DB that have no fingerprint.

These are molecules already in the molecules table but missing from fingerprints.
Uses the CUDA pipeline (with RDKit fallback) same as cuda_retry.py.
"""
import sqlite3
import sys
import time
import traceback
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, ".")

from zynerji_chirality.chirality.fingerprint import chirality_fingerprint
from zynerji_chirality.chirality.detector import HelixChiralityDetector
from zynerji_chirality.db.store import FingerprintStore
from rdkit import Chem
from rdkit.Chem import AllChem

# Try CUDA pipeline
try:
    from zynerji_chirality.cuda.pipeline import CUDAPipeline
    cuda_pipeline = CUDAPipeline()
    logger.info("CUDA pipeline available")
except ImportError:
    cuda_pipeline = None
    logger.info("CUDA pipeline not available, using RDKit only")

DB = sys.argv[1] if len(sys.argv) > 1 else "chembl_work/chembl_screen.db"
NBITS = 256

conn = sqlite3.connect(DB)
rows = conn.execute(
    "SELECT m.id, m.name, m.smiles FROM molecules m "
    "WHERE m.id NOT IN (SELECT mol_id FROM fingerprints) "
    "ORDER BY m.id"
).fetchall()
conn.close()

print(f"Retrying {len(rows)} molecules without fingerprints...")

detector = HelixChiralityDetector()
store = FingerprintStore(DB)
success = 0
failed = []
batch_entries = []

for i, (mid, name, smi) in enumerate(rows):
    t0 = time.time()
    try:
        mol = None

        # Try CUDA embed first
        if cuda_pipeline is not None:
            try:
                mol = cuda_pipeline.embed_molecule(smi)
            except Exception as e:
                logger.debug("CUDA failed for %s: %s", name, e)

        # Fallback: RDKit ETKDG
        if mol is None:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"RDKit cannot parse SMILES")
            mol = Chem.AddHs(mol)
            res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            if res != 0:
                # Try random coords
                params = AllChem.ETKDGv3()
                params.useRandomCoords = True
                res = AllChem.EmbedMolecule(mol, params)
                if res != 0:
                    # Last resort: no 3D coords, compute anyway
                    logger.warning("No 3D embedding for %s, using 2D", name)

        # Chirality fingerprint
        fp = chirality_fingerprint(mol, nbits=NBITS)

        # Chirality detection
        result = detector.detect(mol)

        entry = {
            "smiles": smi,
            "fingerprint": fp,
            "chirality_score": result.chirality_score,
            "chirality_sign": result.chirality_sign,
            "name": name,
            "metadata": {"chembl_id": name or "", "source": "retry_failed"},
        }
        batch_entries.append(entry)
        success += 1
        elapsed = time.time() - t0
        logger.info(
            "[%d/%d] id=%d %s: OK (score=%.4f, sign=%+.1f, %.2fs)",
            i + 1, len(rows), mid, name, result.chirality_score, result.chirality_sign, elapsed
        )

    except Exception as e:
        elapsed = time.time() - t0
        failed.append((mid, name, smi, str(e)))
        logger.error("[%d/%d] id=%d %s: FAILED (%.2fs): %s", i + 1, len(rows), mid, name, elapsed, e)
        traceback.print_exc()

# Flush remaining batch
if batch_entries:
    store.batch_add_fast(batch_entries, nbits=NBITS)
    logger.info("Inserted %d fingerprints into DB", len(batch_entries))

# Verify
conn = sqlite3.connect(DB)
remaining = conn.execute(
    "SELECT COUNT(*) FROM molecules m WHERE m.id NOT IN (SELECT mol_id FROM fingerprints)"
).fetchone()[0]
total_fp = conn.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0]
conn.close()

print(f"\nDone: {success}/{len(rows)} succeeded, {len(failed)} failed")
print(f"Total fingerprints in DB: {total_fp}")
print(f"Still missing: {remaining}")

if failed:
    print("\nStill failing:")
    for mid, name, smi, err in failed:
        print(f"  id={mid} {name} (len={len(smi)}): {err}")
