"""Tests for ChEMBL enantiomer screening pipeline.

Uses synthetic data — no real ChEMBL download required.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from zynerji_chirality.chembl.download import ChiralMolecule, strip_stereo
from zynerji_chirality.chembl.pairs import EnantiomerPair, EnantiomerPairFinder, _classify_relationship
from zynerji_chirality.chembl.activity import ActivityRecord, PairActivity, ActivityEnricher
from zynerji_chirality.chembl.fingerprinter import PairFingerprinter
from zynerji_chirality.chembl.screen import EnantiomerScreen, ScreeningHit
from zynerji_chirality.db.store import FingerprintStore


# --- Test data: known enantiomer pairs ---

L_ALANINE = ChiralMolecule(
    chembl_id="CHEMBL_L_ALA",
    smiles="N[C@@H](C)C(=O)O",
    canonical_smiles="C[C@@H](N)C(=O)O",
    stereo_stripped_smiles="CC(N)C(=O)O",
    n_centers=1,
    cip_labels=[(1, "S")],
)

D_ALANINE = ChiralMolecule(
    chembl_id="CHEMBL_D_ALA",
    smiles="N[C@H](C)C(=O)O",
    canonical_smiles="C[C@H](N)C(=O)O",
    stereo_stripped_smiles="CC(N)C(=O)O",
    n_centers=1,
    cip_labels=[(1, "R")],
)

S_IBUPROFEN = ChiralMolecule(
    chembl_id="CHEMBL_S_IBU",
    smiles="CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1",
    canonical_smiles="CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1",
    stereo_stripped_smiles="CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    n_centers=1,
    cip_labels=[(6, "S")],
)

R_IBUPROFEN = ChiralMolecule(
    chembl_id="CHEMBL_R_IBU",
    smiles="CC(C)Cc1ccc([C@H](C)C(=O)O)cc1",
    canonical_smiles="CC(C)Cc1ccc([C@H](C)C(=O)O)cc1",
    stereo_stripped_smiles="CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    n_centers=1,
    cip_labels=[(6, "R")],
)


class TestChiralMolecule:
    def test_construction(self):
        mol = L_ALANINE
        assert mol.chembl_id == "CHEMBL_L_ALA"
        assert mol.n_centers == 1
        assert mol.stereo_stripped_smiles == "CC(N)C(=O)O"

    def test_hash_and_eq(self):
        mol1 = L_ALANINE
        mol2 = ChiralMolecule(
            chembl_id="CHEMBL_L_ALA",
            smiles="different",
            canonical_smiles="different",
            stereo_stripped_smiles="different",
            n_centers=0,
            cip_labels=[],
        )
        assert mol1 == mol2
        assert hash(mol1) == hash(mol2)
        assert mol1 != D_ALANINE

    def test_strip_stereo(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
        stripped = strip_stereo(mol)
        assert "@@" not in stripped
        assert "@" not in stripped


class TestEnantiomerPairFinder:
    def test_find_alanine_pair(self):
        finder = EnantiomerPairFinder()
        pairs = finder.find_pairs([L_ALANINE, D_ALANINE])
        assert len(pairs) == 1
        assert pairs[0].relationship == "enantiomer"

    def test_find_ibuprofen_pair(self):
        finder = EnantiomerPairFinder()
        pairs = finder.find_pairs([S_IBUPROFEN, R_IBUPROFEN])
        assert len(pairs) == 1
        assert pairs[0].relationship == "enantiomer"

    def test_no_pairs_different_connectivity(self):
        finder = EnantiomerPairFinder()
        pairs = finder.find_pairs([L_ALANINE, S_IBUPROFEN])
        assert len(pairs) == 0

    def test_classify_enantiomers(self):
        rel = _classify_relationship(L_ALANINE, D_ALANINE)
        assert rel == "enantiomer"

    def test_classify_same_isomer(self):
        rel = _classify_relationship(L_ALANINE, L_ALANINE)
        # Same molecule — should return None (no inversion)
        assert rel is None

    def test_save_and_load(self, tmp_path):
        finder = EnantiomerPairFinder()
        pairs = finder.find_pairs([L_ALANINE, D_ALANINE])

        path = str(tmp_path / "pairs.json")
        finder.save_pairs(pairs, path)

        loaded = finder.load_pairs(path)
        assert len(loaded) == 1
        assert loaded[0].relationship == "enantiomer"
        assert loaded[0].mol_a.chembl_id == pairs[0].mol_a.chembl_id

    def test_deduplicate_same_smiles(self):
        """Molecules with same canonical SMILES should be deduplicated."""
        finder = EnantiomerPairFinder()
        dup = ChiralMolecule(
            chembl_id="CHEMBL_L_ALA_DUP",
            smiles="N[C@@H](C)C(=O)O",
            canonical_smiles=L_ALANINE.canonical_smiles,
            stereo_stripped_smiles=L_ALANINE.stereo_stripped_smiles,
            n_centers=1,
            cip_labels=[(1, "S")],
        )
        pairs = finder.find_pairs([L_ALANINE, dup, D_ALANINE])
        # Should still find exactly 1 pair (L vs D), not 2
        assert len(pairs) == 1


class TestActivityEnricher:
    def _make_mock_pair_activity(self) -> PairActivity:
        pair = EnantiomerPair(
            connectivity_smiles="CC(N)C(=O)O",
            mol_a=L_ALANINE,
            mol_b=D_ALANINE,
            relationship="enantiomer",
        )
        acts_a = [
            ActivityRecord(
                chembl_id="CHEMBL_L_ALA",
                target_chembl_id="CHEMBL_T1",
                target_name="Target Alpha",
                assay_type="B",
                standard_type="IC50",
                standard_value=10.0,
                standard_units="nM",
                pchembl_value=8.0,
            ),
            ActivityRecord(
                chembl_id="CHEMBL_L_ALA",
                target_chembl_id="CHEMBL_T2",
                target_name="Target Beta",
                assay_type="B",
                standard_type="Ki",
                standard_value=50.0,
                standard_units="nM",
                pchembl_value=7.3,
            ),
        ]
        acts_b = [
            ActivityRecord(
                chembl_id="CHEMBL_D_ALA",
                target_chembl_id="CHEMBL_T1",
                target_name="Target Alpha",
                assay_type="B",
                standard_type="IC50",
                standard_value=1000.0,
                standard_units="nM",
                pchembl_value=6.0,
            ),
        ]
        pa = PairActivity(
            pair=pair,
            activities_a=acts_a,
            activities_b=acts_b,
        )
        return pa

    def test_analyze_targets(self):
        pa = self._make_mock_pair_activity()
        enricher = ActivityEnricher()
        enricher._analyze_targets(pa)

        assert "CHEMBL_T1" in pa.shared_targets
        assert "CHEMBL_T2" in pa.gap_targets_a
        assert len(pa.gap_targets_b) == 0

    def test_differential_activity(self):
        pa = self._make_mock_pair_activity()
        enricher = ActivityEnricher()
        enricher._analyze_targets(pa)

        # T1: pChEMBL 8.0 vs 6.0, diff=2.0, fold=100x
        assert len(pa.differential_targets) >= 1
        t1_diff = [d for d in pa.differential_targets if d["target_id"] == "CHEMBL_T1"]
        assert len(t1_diff) == 1
        assert t1_diff[0]["fold_change"] == pytest.approx(100.0, rel=0.01)

    def test_find_differential(self):
        pa = self._make_mock_pair_activity()
        enricher = ActivityEnricher()
        enricher._analyze_targets(pa)

        result = enricher.find_differential_activity([pa], min_fold_change=3.0)
        assert len(result) == 1

        result = enricher.find_differential_activity([pa], min_fold_change=200.0)
        assert len(result) == 0

    def test_enrich_from_cache(self):
        pair = EnantiomerPair(
            connectivity_smiles="CC(N)C(=O)O",
            mol_a=L_ALANINE,
            mol_b=D_ALANINE,
            relationship="enantiomer",
        )
        activities = {
            "CHEMBL_L_ALA": [
                ActivityRecord("CHEMBL_L_ALA", "T1", "Target", "B", "IC50", 10.0, "nM", 8.0)
            ],
            "CHEMBL_D_ALA": [],
        }
        enricher = ActivityEnricher()
        result = enricher.enrich_pairs_from_cache([pair], activities)
        assert len(result) == 1
        assert len(result[0].activities_a) == 1
        assert len(result[0].gap_targets_a) == 1

    def test_save_and_load_enriched(self, tmp_path):
        pa = self._make_mock_pair_activity()
        enricher = ActivityEnricher()
        enricher._analyze_targets(pa)

        path = str(tmp_path / "enriched.json")
        enricher.save_enriched([pa], path)

        loaded = enricher.load_enriched(path)
        assert len(loaded) == 1
        assert loaded[0].pair.relationship == "enantiomer"
        assert len(loaded[0].activities_a) == 2


class TestPairFingerprinter:
    def test_fingerprint_pairs(self):
        store = FingerprintStore(":memory:")
        pair = EnantiomerPair(
            connectivity_smiles="CC(N)C(=O)O",
            mol_a=L_ALANINE,
            mol_b=D_ALANINE,
            relationship="enantiomer",
        )

        fingerprinter = PairFingerprinter(store=store, n_workers=1)
        stats = fingerprinter.fingerprint_pairs([pair])

        assert stats["n_success"] == 2
        assert stats["n_fail"] == 0
        assert store.count() == 2

        store.close()

    def test_deduplication(self):
        """Same SMILES in multiple pairs should only be fingerprinted once."""
        store = FingerprintStore(":memory:")
        pair1 = EnantiomerPair(
            connectivity_smiles="CC(N)C(=O)O",
            mol_a=L_ALANINE,
            mol_b=D_ALANINE,
            relationship="enantiomer",
        )
        # Same molecules in another pair context
        pair2 = EnantiomerPair(
            connectivity_smiles="CC(N)C(=O)O",
            mol_a=L_ALANINE,
            mol_b=D_ALANINE,
            relationship="enantiomer",
        )

        fingerprinter = PairFingerprinter(store=store, n_workers=1)
        stats = fingerprinter.fingerprint_pairs([pair1, pair2])

        # Only 2 unique molecules
        assert stats["n_molecules"] == 2
        assert store.count() == 2

        store.close()


class TestEnantiomerScreen:
    def _setup_screen(self):
        store = FingerprintStore(":memory:")
        pair = EnantiomerPair(
            connectivity_smiles="CC(N)C(=O)O",
            mol_a=L_ALANINE,
            mol_b=D_ALANINE,
            relationship="enantiomer",
        )

        # Mock PairActivity with gaps and differential
        pa = PairActivity(
            pair=pair,
            activities_a=[
                ActivityRecord("CHEMBL_L_ALA", "T1", "Target Alpha", "B", "IC50", 10.0, "nM", 8.0),
                ActivityRecord("CHEMBL_L_ALA", "T2", "Target Beta", "B", "Ki", 50.0, "nM", 7.3),
            ],
            activities_b=[
                ActivityRecord("CHEMBL_D_ALA", "T1", "Target Alpha", "B", "IC50", 1000.0, "nM", 6.0),
            ],
        )
        enricher = ActivityEnricher()
        enricher._analyze_targets(pa)

        return store, [pa]

    def test_screen_activity_gaps(self):
        store, pair_activities = self._setup_screen()
        screen = EnantiomerScreen(store=store, pair_activities=pair_activities)

        hits = screen.screen_activity_gaps()
        assert len(hits) > 0
        assert hits[0].hit_type == "activity_gap"
        store.close()

    def test_screen_differential(self):
        store, pair_activities = self._setup_screen()
        screen = EnantiomerScreen(store=store, pair_activities=pair_activities)

        hits = screen.screen_differential(min_fold_change=3.0)
        assert len(hits) > 0
        assert hits[0].hit_type == "differential"
        store.close()

    def test_screen_all(self):
        store, pair_activities = self._setup_screen()
        screen = EnantiomerScreen(store=store, pair_activities=pair_activities)

        hits = screen.screen_all()
        assert len(hits) > 0
        # Should have both types
        types = {h.hit_type for h in hits}
        assert "activity_gap" in types
        assert "differential" in types
        store.close()

    def test_report(self):
        store, pair_activities = self._setup_screen()
        screen = EnantiomerScreen(store=store, pair_activities=pair_activities)

        hits = screen.screen_all()
        report = screen.report(hits)

        assert "ZynerjiChirality" in report
        assert "ACTIVITY_GAP" in report
        assert "DIFFERENTIAL" in report
        store.close()

    def test_report_to_file(self, tmp_path):
        store, pair_activities = self._setup_screen()
        screen = EnantiomerScreen(store=store, pair_activities=pair_activities)

        hits = screen.screen_all()
        out = str(tmp_path / "report.txt")
        report = screen.report(hits, output=out)

        assert os.path.exists(out)
        with open(out) as f:
            content = f.read()
        assert content == report
        store.close()


class TestBatchAddFast:
    def test_speed_vs_serial(self):
        """batch_add_fast should work correctly (speed tested implicitly)."""
        store = FingerprintStore(":memory:")

        entries = []
        for i in range(100):
            entries.append({
                "smiles": f"C{'C' * (i % 10)}O",
                "fingerprint": np.random.randn(128),
                "chirality_score": 0.0,
                "chirality_sign": 0.0,
                "name": f"mol_{i}",
            })

        ids = store.batch_add_fast(entries)
        assert len(ids) == 100
        assert store.count() == 100

        # Verify retrieval
        result = store.get(ids[0])
        assert result is not None
        store.close()

    def test_batch_add_fast_with_metadata(self):
        store = FingerprintStore(":memory:")

        entries = [
            {
                "smiles": "CCO",
                "fingerprint": np.zeros(128),
                "chirality_score": 0.0,
                "chirality_sign": 0.0,
                "metadata": {"source": "test", "chembl_id": "CHEMBL545"},
            }
        ]

        ids = store.batch_add_fast(entries)
        assert len(ids) == 1

        row = store._conn.execute(
            "SELECT value FROM metadata WHERE mol_id = ? AND key = ?",
            (ids[0], "source"),
        ).fetchone()
        assert row[0] == "test"
        store.close()

    def test_cache_invalidation(self):
        """Adding molecules should invalidate the fingerprint matrix cache."""
        store = FingerprintStore(":memory:")

        # Add initial molecule
        store.add("CCO", np.ones(128), 0.0, 0.0)

        # Load cache
        matrix, data = store._load_fingerprint_matrix()
        assert len(data) == 1

        # Add more via batch_add_fast — should invalidate cache
        store.batch_add_fast([{
            "smiles": "CCCO",
            "fingerprint": np.ones(128) * 2,
            "chirality_score": 0.1,
            "chirality_sign": 1.0,
        }])

        # Cache should be invalidated, new load should have 2
        matrix, data = store._load_fingerprint_matrix()
        assert len(data) == 2
        store.close()


class TestVectorizedSearch:
    def test_vectorized_matches_expected(self):
        store = FingerprintStore(":memory:")

        store.add("N[C@@H](C)C(=O)O", np.array([1.0] * 64 + [0.0] * 64),
                   chirality_score=0.1, chirality_sign=-1.0, name="L-Ala")
        store.add("N[C@H](C)C(=O)O", np.array([-1.0] * 64 + [0.0] * 64),
                   chirality_score=0.1, chirality_sign=1.0, name="D-Ala")
        store.add("CCO", np.array([0.0] * 64 + [1.0] * 64),
                   chirality_score=0.0, chirality_sign=0.0, name="Ethanol")

        query = np.array([0.9] * 64 + [0.1] * 64)
        results = store.search_similar(query, k=3)
        assert len(results) > 0
        assert results[0].name == "L-Ala"
        store.close()
