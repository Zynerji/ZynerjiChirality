"""SQLite-backed persistent fingerprint store with similarity search.

Stores molecular fingerprints, chirality scores, and metadata for
fast retrieval and chirality-aware similarity search.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rdkit import Chem


_SCHEMA = """
CREATE TABLE IF NOT EXISTS molecules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    smiles TEXT NOT NULL,
    canonical_smiles TEXT NOT NULL,
    name TEXT
);

CREATE TABLE IF NOT EXISTS fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mol_id INTEGER NOT NULL REFERENCES molecules(id),
    nbits INTEGER NOT NULL,
    params_hash TEXT NOT NULL,
    fingerprint_blob BLOB NOT NULL,
    chirality_score REAL NOT NULL,
    chirality_sign REAL NOT NULL,
    UNIQUE(mol_id, nbits, params_hash)
);

CREATE TABLE IF NOT EXISTS metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mol_id INTEGER NOT NULL REFERENCES molecules(id),
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    UNIQUE(mol_id, key)
);

CREATE INDEX IF NOT EXISTS idx_molecules_canonical ON molecules(canonical_smiles);
CREATE INDEX IF NOT EXISTS idx_fingerprints_mol ON fingerprints(mol_id);
CREATE INDEX IF NOT EXISTS idx_metadata_mol ON metadata(mol_id);
CREATE INDEX IF NOT EXISTS idx_molecules_smiles ON molecules(smiles);
"""


@dataclass
class SearchResult:
    """Result from a similarity search."""
    mol_id: int
    smiles: str
    canonical_smiles: str
    name: str | None
    similarity: float
    chirality_score: float
    chirality_sign: float


class FingerprintStore:
    """SQLite-backed store for molecular fingerprints.

    Parameters
    ----------
    db_path : str or Path
        Path to SQLite database file. Use ":memory:" for in-memory.
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        self._fp_matrix_cache: dict[tuple[int, str], tuple[np.ndarray, list]] | None = None

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def add(
        self,
        smiles: str,
        fingerprint: np.ndarray,
        chirality_score: float,
        chirality_sign: float,
        name: str | None = None,
        nbits: int = 128,
        params_hash: str = "default",
        metadata: dict[str, str] | None = None,
    ) -> int:
        """Add a molecule and its fingerprint to the store.

        Returns the molecule ID.
        """
        mol = Chem.MolFromSmiles(smiles)
        canonical = Chem.MolToSmiles(mol) if mol else smiles

        cur = self._conn.execute(
            "INSERT INTO molecules (smiles, canonical_smiles, name) VALUES (?, ?, ?)",
            (smiles, canonical, name),
        )
        mol_id = cur.lastrowid

        fp_blob = fingerprint.astype(np.float64).tobytes()
        self._conn.execute(
            "INSERT INTO fingerprints (mol_id, nbits, params_hash, fingerprint_blob, "
            "chirality_score, chirality_sign) VALUES (?, ?, ?, ?, ?, ?)",
            (mol_id, nbits, params_hash, fp_blob, chirality_score, chirality_sign),
        )

        if metadata:
            for key, value in metadata.items():
                self._conn.execute(
                    "INSERT OR REPLACE INTO metadata (mol_id, key, value) VALUES (?, ?, ?)",
                    (mol_id, key, value),
                )

        self._conn.commit()
        return mol_id

    def get(self, mol_id: int, nbits: int = 128, params_hash: str = "default") -> dict | None:
        """Get a molecule and its fingerprint by ID."""
        row = self._conn.execute(
            "SELECT m.smiles, m.canonical_smiles, m.name, "
            "f.fingerprint_blob, f.chirality_score, f.chirality_sign "
            "FROM molecules m JOIN fingerprints f ON m.id = f.mol_id "
            "WHERE m.id = ? AND f.nbits = ? AND f.params_hash = ?",
            (mol_id, nbits, params_hash),
        ).fetchone()

        if row is None:
            return None

        fp = np.frombuffer(row[3], dtype=np.float64)
        return {
            "mol_id": mol_id,
            "smiles": row[0],
            "canonical_smiles": row[1],
            "name": row[2],
            "fingerprint": fp,
            "chirality_score": row[4],
            "chirality_sign": row[5],
        }

    def batch_add(
        self,
        entries: list[dict],
        nbits: int = 128,
        params_hash: str = "default",
    ) -> list[int]:
        """Add multiple molecules at once.

        Each entry dict should have: smiles, fingerprint, chirality_score,
        chirality_sign, and optionally name and metadata.
        """
        mol_ids = []
        for entry in entries:
            mol_id = self.add(
                smiles=entry["smiles"],
                fingerprint=entry["fingerprint"],
                chirality_score=entry["chirality_score"],
                chirality_sign=entry["chirality_sign"],
                name=entry.get("name"),
                nbits=nbits,
                params_hash=params_hash,
                metadata=entry.get("metadata"),
            )
            mol_ids.append(mol_id)
        return mol_ids

    def batch_add_fast(
        self,
        entries: list[dict],
        nbits: int = 128,
        params_hash: str = "default",
    ) -> list[int]:
        """Add multiple molecules in a single transaction (10-50x faster).

        Each entry dict should have: smiles, fingerprint, chirality_score,
        chirality_sign, and optionally name and metadata.
        """
        mol_ids = []
        try:
            self._conn.execute("BEGIN")
            for entry in entries:
                smiles = entry["smiles"]
                mol = Chem.MolFromSmiles(smiles)
                canonical = Chem.MolToSmiles(mol) if mol else smiles

                cur = self._conn.execute(
                    "INSERT INTO molecules (smiles, canonical_smiles, name) VALUES (?, ?, ?)",
                    (smiles, canonical, entry.get("name")),
                )
                mol_id = cur.lastrowid

                fp_blob = entry["fingerprint"].astype(np.float64).tobytes()
                self._conn.execute(
                    "INSERT INTO fingerprints (mol_id, nbits, params_hash, fingerprint_blob, "
                    "chirality_score, chirality_sign) VALUES (?, ?, ?, ?, ?, ?)",
                    (mol_id, nbits, params_hash, fp_blob,
                     entry["chirality_score"], entry["chirality_sign"]),
                )

                metadata = entry.get("metadata")
                if metadata:
                    for key, value in metadata.items():
                        self._conn.execute(
                            "INSERT OR REPLACE INTO metadata (mol_id, key, value) VALUES (?, ?, ?)",
                            (mol_id, key, str(value)),
                        )
                mol_ids.append(mol_id)

            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

        # Invalidate cache
        self._fp_matrix_cache = None
        return mol_ids

    def _load_fingerprint_matrix(
        self,
        nbits: int = 128,
        params_hash: str = "default",
    ) -> tuple[np.ndarray, list]:
        """Load all fingerprints as a numpy matrix for vectorized search.

        Returns (matrix, row_data) where matrix is (N, nbits) and row_data
        contains the molecule info for each row.
        """
        cache_key = (nbits, params_hash)
        if self._fp_matrix_cache is not None and cache_key in self._fp_matrix_cache:
            return self._fp_matrix_cache[cache_key]

        rows = self._conn.execute(
            "SELECT m.id, m.smiles, m.canonical_smiles, m.name, "
            "f.fingerprint_blob, f.chirality_score, f.chirality_sign "
            "FROM molecules m JOIN fingerprints f ON m.id = f.mol_id "
            "WHERE f.nbits = ? AND f.params_hash = ?",
            (nbits, params_hash),
        ).fetchall()

        if not rows:
            result = (np.empty((0, nbits)), [])
            self._fp_matrix_cache = {cache_key: result}
            return result

        fps = []
        row_data = []
        for row in rows:
            fp = np.frombuffer(row[4], dtype=np.float64)
            fps.append(fp)
            row_data.append({
                "mol_id": row[0],
                "smiles": row[1],
                "canonical_smiles": row[2],
                "name": row[3],
                "chirality_score": row[5],
                "chirality_sign": row[6],
            })

        matrix = np.vstack(fps)
        result = (matrix, row_data)
        self._fp_matrix_cache = {cache_key: result}
        return result

    def count(self) -> int:
        """Return total number of molecules in store."""
        row = self._conn.execute("SELECT COUNT(*) FROM molecules").fetchone()
        return row[0]

    def search_similar(
        self,
        query_fp: np.ndarray,
        k: int = 10,
        mode: str = "similar",
        nbits: int = 128,
        params_hash: str = "default",
    ) -> list[SearchResult]:
        """Search for similar molecules by fingerprint (vectorized).

        Parameters
        ----------
        query_fp : np.ndarray
            Query fingerprint vector.
        k : int
            Number of results to return.
        mode : str
            "similar" — cosine k-NN on fingerprints.
            "enantiomer" — filter to opposite chirality_sign, then rank by cosine.
        nbits : int
            Fingerprint bit length to search.
        params_hash : str
            Parameter hash to filter by.

        Returns
        -------
        list[SearchResult]
            Top-k most similar molecules.
        """
        query_norm = np.linalg.norm(query_fp)
        if query_norm < 1e-12:
            return []

        matrix, row_data = self._load_fingerprint_matrix(nbits, params_hash)
        if len(row_data) == 0:
            return []

        # Vectorized cosine similarity
        norms = np.linalg.norm(matrix, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            return []

        cos_sims = np.zeros(len(row_data))
        cos_sims[valid] = (matrix[valid] @ query_fp) / (norms[valid] * query_norm)
        similarities = (cos_sims + 1.0) / 2.0  # map to [0, 1]

        # Build results
        results = []
        for i, rd in enumerate(row_data):
            if not valid[i]:
                continue
            results.append(SearchResult(
                mol_id=rd["mol_id"],
                smiles=rd["smiles"],
                canonical_smiles=rd["canonical_smiles"],
                name=rd["name"],
                similarity=float(similarities[i]),
                chirality_score=rd["chirality_score"],
                chirality_sign=rd["chirality_sign"],
            ))

        # For enantiomer mode, filter to opposite sign of the most similar match
        if mode == "enantiomer" and results:
            results.sort(key=lambda r: r.similarity, reverse=True)
            query_sign = results[0].chirality_sign
            results = [r for r in results if r.chirality_sign * query_sign < 0]

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:k]
