"""Tests for target-specific chirality models."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from zynerji_chirality.targets.data_loader import (
    TargetTrainingRecord,
    TargetDataset,
    TargetDataExtractor,
)
from zynerji_chirality.targets.target_model import (
    TargetPrediction,
    TargetChiralityModel,
)
from zynerji_chirality.targets.global_model import (
    GlobalPrediction,
    GlobalChiralityPredictor,
)
from zynerji_chirality.targets.profiler import (
    TargetSensitivityEntry,
    TargetProfile,
    TargetProfiler,
)
from zynerji_chirality.targets.trainer import (
    TrainingResult,
    TargetModelTrainer,
)

# Test SMILES — L/D-alanine
SMILES_R = "N[C@@H](C)C(=O)O"
SMILES_S = "N[C@H](C)C(=O)O"


def _make_records(target_id="CHEMBL123", n=30):
    """Generate synthetic training records."""
    records = []
    for i in range(n):
        fc = 5.0 + i * 0.1 if i < n // 2 else 1.0 + i * 0.01
        records.append(TargetTrainingRecord(
            smiles_a=SMILES_R,
            smiles_b=SMILES_S,
            target_chembl_id=target_id,
            target_name=f"Target {target_id}",
            fold_change=fc,
            is_differential=fc >= 3.0,
            mean_activity_a=100.0,
            mean_activity_b=100.0 * fc,
        ))
    return records


def _make_dataset(target_id="CHEMBL123", n=30):
    """Generate a synthetic TargetDataset."""
    records = _make_records(target_id, n)
    n_diff = sum(1 for r in records if r.is_differential)
    return TargetDataset(
        target_chembl_id=target_id,
        target_name=f"Target {target_id}",
        records=records,
        n_differential=n_diff,
        n_non_differential=len(records) - n_diff,
    )


def _make_enriched_json(tmp_dir):
    """Create mock enriched_pairs.json."""
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
            "activities_a": [],
            "activities_b": [],
            "shared_targets": ["CHEMBL_T1", "CHEMBL_T2"],
            "differential_targets": [
                {
                    "target_id": "CHEMBL_T1",
                    "target_name": "Receptor 1",
                    "type": "IC50",
                    "mean_a": 10.0,
                    "mean_b": 100.0,
                    "fold_change": 10.0,
                    "units": "nM",
                },
            ],
            "gap_targets_a": [],
            "gap_targets_b": [],
        })
    path = Path(tmp_dir) / "enriched_pairs.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return str(path)


# ── TestTargetDataExtractor ─────────────────────────────────────────


class TestTargetDataExtractor:
    def test_extract_from_enriched(self, tmp_path):
        enriched_path = _make_enriched_json(tmp_path)
        extractor = TargetDataExtractor()
        datasets = extractor.extract_from_enriched(enriched_path)
        assert len(datasets) >= 1
        # CHEMBL_T1 should have 5 differential records
        t1_ds = [d for d in datasets if d.target_chembl_id == "CHEMBL_T1"]
        assert len(t1_ds) == 1
        assert t1_ds[0].n_differential == 5

    def test_filter_min_records(self, tmp_path):
        enriched_path = _make_enriched_json(tmp_path)
        extractor = TargetDataExtractor()
        datasets = extractor.extract_from_enriched(enriched_path)
        # With min_records=100, all should be filtered
        filtered = [d for d in datasets if len(d.records) >= 100]
        assert len(filtered) == 0

    def test_global_dataset_shape(self):
        datasets = [_make_dataset("T1", 20), _make_dataset("T2", 15)]
        extractor = TargetDataExtractor()
        records, target_ids = extractor.get_global_dataset(datasets)
        assert len(records) == 35
        assert len(target_ids) == 2

    def test_save_load(self, tmp_path):
        datasets = [_make_dataset("T1", 10)]
        extractor = TargetDataExtractor()
        save_path = tmp_path / "datasets.json"
        extractor.save_datasets(datasets, save_path)
        loaded = extractor.load_datasets(save_path)
        assert len(loaded) == 1
        assert loaded[0].target_chembl_id == "T1"
        assert len(loaded[0].records) == 10


# ── TestTargetChiralityModel ────────────────────────────────────────


class TestTargetChiralityModel:
    def test_unfitted_raises(self):
        model = TargetChiralityModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_sensitivity(SMILES_R, SMILES_S)

    def test_combined_fp_shape(self):
        model = TargetChiralityModel(nbits=64)
        fp = model._combined_fingerprint(SMILES_R, SMILES_S)
        assert fp.shape == (64 * 4,)

    def test_fit_and_predict(self):
        dataset = _make_dataset("CHEMBL123", n=30)
        model = TargetChiralityModel(nbits=64, n_estimators=10)
        model.fit_from_dataset(dataset)

        pred = model.predict_sensitivity(SMILES_R, SMILES_S)
        assert isinstance(pred, TargetPrediction)
        assert pred.target_chembl_id == "CHEMBL123"
        assert pred.predicted_fold_change >= 1.0
        assert 0.0 <= pred.confidence <= 1.0

    def test_save_load(self, tmp_path):
        dataset = _make_dataset("CHEMBL456", n=30)
        model = TargetChiralityModel(nbits=64, n_estimators=10)
        model.fit_from_dataset(dataset)

        save_path = tmp_path / "target_CHEMBL456.pkl"
        model.save(save_path)

        loaded = TargetChiralityModel()
        loaded.load(save_path)
        assert loaded.target_chembl_id == "CHEMBL456"
        assert loaded._fitted

        pred = loaded.predict_sensitivity(SMILES_R, SMILES_S)
        assert isinstance(pred, TargetPrediction)


# ── TestGlobalChiralityPredictor ────────────────────────────────────


class TestGlobalChiralityPredictor:
    def test_fit_and_predict(self):
        records = _make_records("T1", 20) + _make_records("T2", 20)
        model = GlobalChiralityPredictor(nbits=64, n_estimators=10)
        model.fit(records)

        pred = model.predict_for_target(SMILES_R, SMILES_S, "T1")
        assert isinstance(pred, GlobalPrediction)
        assert pred.predicted_fold_change >= 1.0
        assert pred.target_chembl_id == "T1"

    def test_predict_all_targets(self):
        records = _make_records("T1", 20) + _make_records("T2", 20)
        model = GlobalChiralityPredictor(nbits=64, n_estimators=10)
        model.fit(records)

        preds = model.predict_all_targets(SMILES_R, SMILES_S)
        assert len(preds) == 2
        # Sorted by fold change descending
        assert preds[0].predicted_fold_change >= preds[1].predicted_fold_change

    def test_unknown_target_fallback(self):
        records = _make_records("T1", 30)
        model = GlobalChiralityPredictor(nbits=64, n_estimators=10)
        model.fit(records)

        # Unknown target gets zero encoding — should still predict
        pred = model.predict_for_target(SMILES_R, SMILES_S, "UNKNOWN_TARGET")
        assert isinstance(pred, GlobalPrediction)
        assert pred.predicted_fold_change >= 1.0

    def test_save_load(self, tmp_path):
        records = _make_records("T1", 20) + _make_records("T2", 20)
        model = GlobalChiralityPredictor(nbits=64, n_estimators=10)
        model.fit(records)

        save_path = tmp_path / "global_model.pkl"
        model.save(save_path)

        loaded = GlobalChiralityPredictor()
        loaded.load(save_path)
        assert loaded._fitted
        assert loaded._n_targets == 2


# ── TestTargetProfiler ──────────────────────────────────────────────


class TestTargetProfiler:
    def test_profile_with_per_target(self):
        dataset = _make_dataset("CHEMBL123", n=30)
        model = TargetChiralityModel(nbits=64, n_estimators=10)
        model.fit_from_dataset(dataset)

        profiler = TargetProfiler(per_target_models={"CHEMBL123": model})
        profile = profiler.profile(SMILES_R, SMILES_S)

        assert isinstance(profile, TargetProfile)
        assert len(profile.entries) == 1
        assert profile.entries[0].model_source == "per_target"

    def test_profile_with_global_fallback(self):
        records = _make_records("T1", 20) + _make_records("T2", 20)
        global_model = GlobalChiralityPredictor(nbits=64, n_estimators=10)
        global_model.fit(records)

        profiler = TargetProfiler(global_model=global_model)
        profile = profiler.profile(SMILES_R, SMILES_S)

        assert isinstance(profile, TargetProfile)
        assert len(profile.entries) == 2
        assert all(e.model_source == "global" for e in profile.entries)

    def test_load_from_directory(self, tmp_path):
        dataset = _make_dataset("CHEMBL789", n=30)
        model = TargetChiralityModel(nbits=64, n_estimators=10)
        model.fit_from_dataset(dataset)
        model.save(tmp_path / "target_CHEMBL789.pkl")

        records = _make_records("T1", 20)
        gmodel = GlobalChiralityPredictor(nbits=64, n_estimators=10)
        gmodel.fit(records)
        gmodel.save(tmp_path / "global_model.pkl")

        profiler = TargetProfiler.load_from_directory(tmp_path)
        assert "CHEMBL789" in profiler.per_target_models
        assert profiler.global_model is not None


# ── TestTargetModelTrainer ──────────────────────────────────────────


class TestTargetModelTrainer:
    def test_train_all_synthetic(self, tmp_path):
        enriched_path = _make_enriched_json(tmp_path)
        work_dir = tmp_path / "targets_work"

        trainer = TargetModelTrainer(
            work_dir=work_dir,
            nbits=64,
            min_records=3,  # Low threshold for synthetic data
            cv_folds=2,
        )
        result = trainer.train_all(enriched_path)

        assert isinstance(result, TrainingResult)
        assert result.n_targets_trained >= 1
        assert result.elapsed_seconds >= 0
        assert (work_dir / "training_report.json").exists()
        assert (work_dir / "global_model.pkl").exists()

    def test_resume_checkpoint(self, tmp_path):
        enriched_path = _make_enriched_json(tmp_path)
        work_dir = tmp_path / "targets_work"

        trainer = TargetModelTrainer(
            work_dir=work_dir, nbits=64, min_records=3,
        )
        result1 = trainer.train_all(enriched_path)

        # Resume should skip already-done targets
        result2 = trainer.train_all(enriched_path, resume=True)
        assert result2.n_targets_trained == 0  # All already done
