"""Tests for ADMET chirality prediction."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from zynerji_chirality.admet.data_loader import (
    ADMET_PROPERTIES,
    ADMETRecord,
    ADMETPropertyDataset,
    ADMETDataLoader,
)
from zynerji_chirality.admet.property_model import (
    ADMETPropertyPrediction,
    ADMETPropertyPredictor,
)
from zynerji_chirality.admet.profiler import (
    ADMETProfileEntry,
    ADMETProfile,
    ADMETProfiler,
)
from zynerji_chirality.admet.differential import (
    DifferentialADMETEntry,
    DifferentialADMETResult,
    DifferentialADMETAnalyzer,
)

SMILES_R = "N[C@@H](C)C(=O)O"
SMILES_S = "N[C@H](C)C(=O)O"
SMILES_ACHIRAL = "CC(=O)O"  # acetic acid


def _make_records(property_name="herg", n=30):
    """Generate synthetic ADMET records."""
    records = []
    for i in range(n):
        smiles = SMILES_R if i % 2 == 0 else SMILES_S
        records.append(ADMETRecord(
            smiles=smiles,
            chembl_id=f"CHEMBL{i}",
            property_name=property_name,
            standard_type="IC50",
            standard_value=5.0 + i * 0.5,
            standard_units="uM",
            target_name="hERG" if property_name == "herg" else "Test target",
        ))
    return records


def _make_dataset(property_name="herg", n=30):
    """Generate synthetic ADMET dataset."""
    return ADMETPropertyDataset(
        property_name=property_name,
        description=f"Test {property_name}",
        records=_make_records(property_name, n),
    )


def _make_enriched_json(tmp_dir):
    """Create mock enriched_pairs.json with ADMET-like activities."""
    data = []
    for i in range(5):
        data.append({
            "pair": {
                "mol_a": {"chembl_id": f"CHEMBL_A{i}", "canonical_smiles": SMILES_R,
                          "n_centers": 1, "inchi_key": ""},
                "mol_b": {"chembl_id": f"CHEMBL_B{i}", "canonical_smiles": SMILES_S,
                          "n_centers": 1, "inchi_key": ""},
                "connectivity_smiles": "NC(C)C(=O)O",
                "relationship": "enantiomer",
            },
            "activities_a": [
                {
                    "chembl_id": f"CHEMBL_A{i}",
                    "target_chembl_id": "CHEMBL_hERG",
                    "target_name": "hERG potassium channel",
                    "assay_type": "B",
                    "standard_type": "IC50",
                    "standard_value": 15.0 + i,
                    "standard_units": "uM",
                    "pchembl_value": None,
                },
            ],
            "activities_b": [
                {
                    "chembl_id": f"CHEMBL_B{i}",
                    "target_chembl_id": "CHEMBL_hERG",
                    "target_name": "hERG potassium channel",
                    "assay_type": "B",
                    "standard_type": "IC50",
                    "standard_value": 5.0 + i,
                    "standard_units": "uM",
                    "pchembl_value": None,
                },
            ],
            "shared_targets": [],
            "differential_targets": [],
            "gap_targets_a": [],
            "gap_targets_b": [],
        })
    path = Path(tmp_dir) / "enriched_pairs.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return str(path)


# ── TestADMETDataLoader ─────────────────────────────────────────────


class TestADMETDataLoader:
    def test_classify_herg(self):
        loader = ADMETDataLoader()
        result = loader._classify_record({
            "assay_type": "B",
            "standard_type": "IC50",
            "target_name": "hERG potassium channel",
        })
        assert result == "herg"

    def test_classify_cyp(self):
        loader = ADMETDataLoader()
        result = loader._classify_record({
            "assay_type": "B",
            "standard_type": "IC50",
            "target_name": "Cytochrome P450 3A4",
        })
        assert result == "cyp_inhibition"

    def test_classify_unknown(self):
        loader = ADMETDataLoader()
        result = loader._classify_record({
            "assay_type": "B",
            "standard_type": "IC50",
            "target_name": "Some random receptor",
        })
        assert result is None

    def test_extract_from_enriched(self, tmp_path):
        enriched_path = _make_enriched_json(tmp_path)
        loader = ADMETDataLoader()
        datasets = loader.load_from_enriched(enriched_path)
        assert "herg" in datasets
        assert len(datasets["herg"].records) >= 1

    def test_save_load(self, tmp_path):
        datasets = {"herg": _make_dataset("herg", 10)}
        loader = ADMETDataLoader()
        save_path = tmp_path / "admet_datasets.json"
        loader.save_datasets(datasets, save_path)
        loaded = loader.load_datasets(save_path)
        assert "herg" in loaded
        assert len(loaded["herg"].records) == 10


# ── TestADMETPropertyPredictor ──────────────────────────────────────


class TestADMETPropertyPredictor:
    def test_unfitted_raises(self):
        model = ADMETPropertyPredictor(property_name="herg")
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_property(SMILES_R)

    def test_fit_and_predict(self):
        dataset = _make_dataset("herg", n=30)
        model = ADMETPropertyPredictor(
            property_name="herg", units="uM", nbits=64, n_estimators=10,
        )
        model.fit_from_dataset(dataset)

        pred = model.predict_property(SMILES_R)
        assert isinstance(pred, ADMETPropertyPrediction)
        assert pred.property_name == "herg"
        assert pred.units == "uM"
        assert pred.interpretation in ("low_risk", "moderate_risk", "high_risk")

    def test_risk_interpretation_herg(self):
        model = ADMETPropertyPredictor(property_name="herg")
        assert model._interpret_risk(5.0) == "high_risk"    # IC50 < 10uM
        assert model._interpret_risk(20.0) == "moderate_risk"  # 10 < IC50 < 30
        assert model._interpret_risk(50.0) == "low_risk"    # IC50 > 30uM

    def test_save_load(self, tmp_path):
        dataset = _make_dataset("herg", n=30)
        model = ADMETPropertyPredictor(
            property_name="herg", units="uM", nbits=64, n_estimators=10,
        )
        model.fit_from_dataset(dataset)

        save_path = tmp_path / "admet_herg.pkl"
        model.save(save_path)

        loaded = ADMETPropertyPredictor()
        loaded.load(save_path)
        assert loaded.property_name == "herg"
        assert loaded._fitted


# ── TestADMETProfiler ───────────────────────────────────────────────


class TestADMETProfiler:
    def test_profile_synthetic(self):
        dataset = _make_dataset("herg", n=30)
        model = ADMETPropertyPredictor(
            property_name="herg", units="uM", nbits=64, n_estimators=10,
        )
        model.fit_from_dataset(dataset)

        profiler = ADMETProfiler(models={"herg": model})
        profile = profiler.profile(SMILES_R)

        assert isinstance(profile, ADMETProfile)
        assert len(profile.entries) == 1
        assert profile.overall_risk in ("low_risk", "moderate_risk", "high_risk")

    def test_overall_risk_calculation(self):
        # Create two models with different risk outcomes
        dataset_herg = _make_dataset("herg", n=30)
        model_herg = ADMETPropertyPredictor(
            property_name="herg", units="uM", nbits=64, n_estimators=10,
        )
        model_herg.fit_from_dataset(dataset_herg)

        profiler = ADMETProfiler(models={"herg": model_herg})
        profile = profiler.profile(SMILES_R)

        # Overall risk should be worst of all entries
        if any(e.risk_level == "high_risk" for e in profile.entries):
            assert profile.overall_risk == "high_risk"

    def test_load_from_directory(self, tmp_path):
        dataset = _make_dataset("herg", n=30)
        model = ADMETPropertyPredictor(
            property_name="herg", units="uM", nbits=64, n_estimators=10,
        )
        model.fit_from_dataset(dataset)
        model.save(tmp_path / "admet_herg.pkl")

        profiler = ADMETProfiler.load_from_directory(tmp_path)
        assert "herg" in profiler.models


# ── TestDifferentialADMETAnalyzer ───────────────────────────────────


class TestDifferentialADMETAnalyzer:
    def test_analyze_pair(self):
        dataset = _make_dataset("herg", n=30)
        model = ADMETPropertyPredictor(
            property_name="herg", units="uM", nbits=64, n_estimators=10,
        )
        model.fit_from_dataset(dataset)

        profiler = ADMETProfiler(models={"herg": model})
        analyzer = DifferentialADMETAnalyzer(profiler)

        result = analyzer.analyze(SMILES_R, SMILES_S)
        assert isinstance(result, DifferentialADMETResult)
        assert len(result.entries) == 1
        assert result.recommendation in ("chirality_matters", "chirality_neutral")

    def test_chirality_neutral_case(self):
        # Same molecule compared to itself should show no divergence
        dataset = _make_dataset("herg", n=30)
        model = ADMETPropertyPredictor(
            property_name="herg", units="uM", nbits=64, n_estimators=10,
        )
        model.fit_from_dataset(dataset)

        profiler = ADMETProfiler(models={"herg": model})
        analyzer = DifferentialADMETAnalyzer(profiler)

        result = analyzer.analyze(SMILES_R, SMILES_R)
        assert result.worst_divergence == pytest.approx(1.0, abs=0.01)
        assert result.recommendation == "chirality_neutral"
