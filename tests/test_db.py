"""Tests for fingerprint database store."""

import numpy as np
import pytest

from zynerji_chirality.db.store import FingerprintStore, SearchResult


@pytest.fixture
def store():
    """Create an in-memory store for testing."""
    with FingerprintStore(":memory:") as s:
        yield s


class TestFingerprintStore:
    def test_add_and_get(self, store):
        """Add a molecule and retrieve it."""
        fp = np.random.randn(128)
        mol_id = store.add("CCO", fp, chirality_score=0.0, chirality_sign=0.0, name="Ethanol")
        assert mol_id > 0

        result = store.get(mol_id)
        assert result is not None
        assert result["smiles"] == "CCO"
        assert result["name"] == "Ethanol"
        assert np.allclose(result["fingerprint"], fp)
        assert result["chirality_score"] == 0.0

    def test_get_nonexistent(self, store):
        """Getting a nonexistent ID returns None."""
        assert store.get(999) is None

    def test_batch_add(self, store):
        """Batch add multiple molecules."""
        entries = [
            {"smiles": "CCO", "fingerprint": np.random.randn(128),
             "chirality_score": 0.0, "chirality_sign": 0.0, "name": "Ethanol"},
            {"smiles": "N[C@@H](C)C(=O)O", "fingerprint": np.random.randn(128),
             "chirality_score": 0.1, "chirality_sign": -1.0, "name": "L-Alanine"},
        ]
        mol_ids = store.batch_add(entries)
        assert len(mol_ids) == 2
        assert store.count() == 2

    def test_count(self, store):
        """Count tracks additions."""
        assert store.count() == 0
        store.add("CCO", np.zeros(128), 0.0, 0.0)
        assert store.count() == 1

    def test_metadata(self, store):
        """Metadata is stored and retrievable."""
        fp = np.zeros(128)
        mol_id = store.add("CCO", fp, 0.0, 0.0, metadata={"source": "test"})
        # Metadata is stored (verify through raw query)
        row = store._conn.execute(
            "SELECT value FROM metadata WHERE mol_id = ? AND key = ?",
            (mol_id, "source"),
        ).fetchone()
        assert row[0] == "test"


class TestSimilaritySearch:
    def _populate(self, store):
        """Add test molecules to store."""
        np.random.seed(42)
        # Add some molecules with known fingerprints
        store.add("N[C@@H](C)C(=O)O", np.array([1.0] * 64 + [0.0] * 64),
                   chirality_score=0.1, chirality_sign=-1.0, name="L-Ala")
        store.add("N[C@H](C)C(=O)O", np.array([-1.0] * 64 + [0.0] * 64),
                   chirality_score=0.1, chirality_sign=1.0, name="D-Ala")
        store.add("CCO", np.array([0.0] * 64 + [1.0] * 64),
                   chirality_score=0.0, chirality_sign=0.0, name="Ethanol")

    def test_similar_search(self, store):
        """Similar search returns closest matches."""
        self._populate(store)
        query = np.array([0.9] * 64 + [0.1] * 64)
        results = store.search_similar(query, k=3, mode="similar")
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        # L-Ala should be most similar
        assert results[0].name == "L-Ala"

    def test_enantiomer_search(self, store):
        """Enantiomer search finds opposite-sign molecules."""
        self._populate(store)
        query = np.array([1.0] * 64 + [0.0] * 64)  # similar to L-Ala
        results = store.search_similar(query, k=3, mode="enantiomer")
        # Should find D-Ala (opposite sign)
        assert len(results) > 0
        assert results[0].chirality_sign > 0  # opposite of L-Ala's -1.0

    def test_empty_store_search(self, store):
        """Search on empty store returns empty list."""
        query = np.random.randn(128)
        results = store.search_similar(query, k=5)
        assert results == []

    def test_zero_query(self, store):
        """Zero vector query returns empty."""
        self._populate(store)
        results = store.search_similar(np.zeros(128), k=5)
        assert results == []
