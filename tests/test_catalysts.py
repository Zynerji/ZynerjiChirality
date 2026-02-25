"""Tests for the chiral catalysts module."""

import numpy as np
import pytest

from zynerji_chirality.catalysts.ee_predictor import EEPredictor, EEPrediction
from zynerji_chirality.catalysts.ligand_screen import (
    LigandScreener,
    LigandScore,
    rank_by_chirality_profile,
)
from zynerji_chirality.catalysts.reaction_sim import (
    CatalyticReactionAnalyzer,
    CatalyticReactionResult,
)
from zynerji_chirality.catalysts.benchmarks import (
    get_benchmark_data,
    run_binap_benchmark,
    BINAP_HYDROGENATION_DATA,
)


# ── Synthetic training data for EEPredictor ───────────────────────────────

def _make_training_data():
    """Create synthetic reaction data for testing."""
    return [
        {
            "catalyst": "N[C@@H](C)C(=O)O",     # L-alanine as catalyst proxy
            "substrate": "CC(=O)CC(=O)OCC",       # ethyl acetoacetate
            "ee": 95.0,
            "major": "R",
        },
        {
            "catalyst": "N[C@H](C)C(=O)O",       # D-alanine
            "substrate": "CC(=O)CC(=O)OCC",
            "ee": 93.0,
            "major": "S",
        },
        {
            "catalyst": "N[C@@H](C)C(=O)O",
            "substrate": "CC(=O)C1=CC=CC=C1",     # acetophenone
            "ee": 87.0,
            "major": "R",
        },
        {
            "catalyst": "N[C@H](C)C(=O)O",
            "substrate": "CC(=O)C1=CC=CC=C1",
            "ee": 85.0,
            "major": "S",
        },
        {
            "catalyst": "N[C@@H](CC1=CC=CC=C1)C(=O)O",  # L-phenylalanine
            "substrate": "CC(=O)CC(=O)OCC",
            "ee": 90.0,
            "major": "R",
        },
        {
            "catalyst": "N[C@@H](CC1=CC=CC=C1)C(=O)O",
            "substrate": "O=C1CCCCC1",              # cyclohexanone
            "ee": 40.0,
            "major": "S",
        },
    ]


# ── EEPredictor tests ────────────────────────────────────────────────────

class TestEEPredictor:

    def test_predict_unfitted_raises(self):
        """Should raise RuntimeError before fit()."""
        predictor = EEPredictor()
        with pytest.raises(RuntimeError, match="not fitted"):
            predictor.predict_ee("N[C@@H](C)C(=O)O", "CC(=O)CC(=O)OCC")

    def test_combined_fingerprint_shape(self):
        """Combined fingerprint should be 4 * nbits."""
        predictor = EEPredictor(nbits=64)
        fp = predictor._combined_fingerprint(
            "N[C@@H](C)C(=O)O", "CC(=O)CC(=O)OCC"
        )
        assert fp.shape == (4 * 64,)

    def test_fit_and_predict(self):
        """Fitted model should produce predictions in valid range."""
        predictor = EEPredictor(nbits=64)
        data = _make_training_data()
        predictor.fit(data)

        pred = predictor.predict_ee("N[C@@H](C)C(=O)O", "CC(=O)CC(=O)OCC")
        assert isinstance(pred, EEPrediction)
        assert 0.0 <= pred.ee_percent <= 100.0
        assert pred.major_enantiomer in ("R", "S")

    def test_ee_prediction_dataclass(self):
        """EEPrediction fields should all be populated."""
        predictor = EEPredictor(nbits=64)
        data = _make_training_data()
        predictor.fit(data)

        pred = predictor.predict_ee("N[C@@H](C)C(=O)O", "CC(=O)CC(=O)OCC")
        assert isinstance(pred.ee_percent, float)
        assert isinstance(pred.major_enantiomer, str)
        assert isinstance(pred.confidence, float)
        assert isinstance(pred.catalyst_fp, np.ndarray)
        assert isinstance(pred.substrate_fp, np.ndarray)
        assert pred.catalyst_fp.shape == (64,)
        assert pred.substrate_fp.shape == (64,)

    def test_cross_validate(self):
        """Cross-validation should return r2 scores."""
        predictor = EEPredictor(nbits=64)
        data = _make_training_data()
        result = predictor.cross_validate(data, cv=2)
        assert "mean_r2" in result
        assert "std_r2" in result
        assert "scores" in result
        assert isinstance(result["scores"], list)

    def test_fit_empty_raises(self):
        """Fitting with no valid data should raise."""
        predictor = EEPredictor(nbits=64)
        with pytest.raises(ValueError, match="No valid"):
            predictor.fit([{"catalyst": "INVALID", "substrate": "INVALID", "ee": 0}])


# ── LigandScreener tests ─────────────────────────────────────────────────

class TestLigandScreener:

    def test_screen_returns_ranked(self):
        """Results should be sorted by predicted ee (descending)."""
        screener = LigandScreener()
        ligands = [
            "N[C@@H](C)C(=O)O",           # L-alanine (chiral)
            "NCC(=O)O",                     # glycine (achiral)
            "N[C@@H](CC1=CC=CC=C1)C(=O)O", # L-phenylalanine (chiral)
        ]
        results = screener.screen("CC(=O)CC(=O)OCC", ligands)
        assert len(results) == 3
        assert all(isinstance(r, LigandScore) for r in results)
        # Should be ranked
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3
        # Descending ee
        for i in range(len(results) - 1):
            assert results[i].predicted_ee >= results[i + 1].predicted_ee

    def test_screen_with_predictor(self):
        """ML mode should use predictor."""
        screener = LigandScreener()
        predictor = EEPredictor(nbits=64)
        predictor.fit(_make_training_data())

        ligands = ["N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O"]
        results = screener.screen("CC(=O)CC(=O)OCC", ligands, ee_predictor=predictor)
        assert len(results) == 2
        assert all(r.predicted_ee >= 0 for r in results)

    def test_rank_by_chirality_profile(self):
        """Chiral molecules should rank higher than achiral."""
        ligands = [
            "NCC(=O)O",             # glycine (achiral)
            "N[C@@H](C)C(=O)O",    # L-alanine (chiral)
        ]
        ranked = rank_by_chirality_profile(ligands)
        assert len(ranked) == 2
        # Chiral should have higher score
        assert ranked[0][1] >= ranked[1][1]

    def test_empty_ligand_list(self):
        """Empty input should return empty output."""
        screener = LigandScreener()
        results = screener.screen("CC(=O)CC(=O)OCC", [])
        assert results == []


# ── CatalyticReactionAnalyzer tests ──────────────────────────────────────

class TestCatalyticReactionAnalyzer:

    def test_analyze_catalytic(self):
        """Should return CatalyticReactionResult."""
        analyzer = CatalyticReactionAnalyzer()
        # Simple SN2: R → S inversion with atom mapping
        rxn = "[C:1]([Br:2])>>[C:1]([O:3])"
        result = analyzer.analyze_catalytic(rxn, "N[C@@H](C)C(=O)O")
        assert isinstance(result, CatalyticReactionResult)
        assert isinstance(result.predicted_selectivity, str)
        assert result.predicted_selectivity in ("Re", "Si")

    def test_catalyst_chirality_included(self):
        """Catalyst chirality result should be populated."""
        analyzer = CatalyticReactionAnalyzer()
        rxn = "[C:1]([Br:2])>>[C:1]([O:3])"
        result = analyzer.analyze_catalytic(rxn, "N[C@@H](C)C(=O)O")
        assert result.catalyst_chirality is not None
        assert result.catalyst_chirality.is_chiral

    def test_face_selectivity_score(self):
        """Face selectivity score should be computed."""
        analyzer = CatalyticReactionAnalyzer()
        rxn = "[C:1]([Br:2])>>[C:1]([O:3])"
        result = analyzer.analyze_catalytic(rxn, "N[C@@H](C)C(=O)O")
        assert isinstance(result.face_selectivity_score, float)
        assert 0.0 <= result.face_selectivity_score <= 1.0

    def test_achiral_catalyst(self):
        """Achiral catalyst should still produce a result."""
        analyzer = CatalyticReactionAnalyzer()
        rxn = "[C:1]([Br:2])>>[C:1]([O:3])"
        result = analyzer.analyze_catalytic(rxn, "NCC(=O)O")  # glycine
        assert isinstance(result, CatalyticReactionResult)
        assert not result.catalyst_chirality.is_chiral


# ── Benchmark tests ──────────────────────────────────────────────────────

class TestBINAPBenchmark:

    def test_benchmark_data_loaded(self):
        """Benchmark data should have expected fields."""
        data = get_benchmark_data()
        assert len(data) >= 10
        for entry in data:
            assert "catalyst" in entry
            assert "substrate" in entry
            assert "ee" in entry
            assert "major_enantiomer" in entry
            assert "reaction_type" in entry
            assert 0 <= entry["ee"] <= 100

    def test_benchmark_data_immutable(self):
        """get_benchmark_data should return a copy."""
        data1 = get_benchmark_data()
        data2 = get_benchmark_data()
        data1.pop()
        assert len(data2) > len(data1)

    def test_run_benchmark(self):
        """Benchmark should run without error on fitted predictor."""
        predictor = EEPredictor(nbits=64)
        predictor.fit(_make_training_data())

        result = run_binap_benchmark(predictor)
        assert "n_reactions" in result
        assert "predictions" in result
        assert result["n_reactions"] >= 10
        assert isinstance(result["predictions"], list)

    def test_binap_data_constant(self):
        """BINAP_HYDROGENATION_DATA should be accessible."""
        assert isinstance(BINAP_HYDROGENATION_DATA, list)
        assert len(BINAP_HYDROGENATION_DATA) >= 10
