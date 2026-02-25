"""Tests for chirality-aware generative chemistry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from zynerji_chirality.generative.enumerator import (
    StereoisomerVariant,
    StereoisomerEnumerator,
)
from zynerji_chirality.generative.scorer import (
    ChiralityScore,
    ChiralityScorer,
)
from zynerji_chirality.generative.mmp import (
    ChiralMMP,
    ChiralMMPAnalyzer,
)
from zynerji_chirality.generative.filter import (
    FilterResult,
    GenerativeFilter,
)
from zynerji_chirality.generative.optimizer import (
    OptimizationStep,
    OptimizationResult,
    ChiralOptimizer,
)

# Test SMILES
SMILES_1CENTER = "N[C@@H](C)C(=O)O"     # L-alanine (1 center)
SMILES_1CENTER_S = "N[C@H](C)C(=O)O"    # D-alanine
SMILES_2CENTER = "C[C@@H](O)[C@@H](O)C"  # 2 centers
SMILES_ACHIRAL = "CC(=O)O"                # acetic acid (no centers)


# ── TestStereoisomerEnumerator ──────────────────────────────────────


class TestStereoisomerEnumerator:
    def test_single_center(self):
        enum = StereoisomerEnumerator(compute_fingerprints=False)
        variants = enum.enumerate(SMILES_1CENTER)
        assert len(variants) == 2
        smiles_set = {v.canonical_smiles for v in variants}
        assert len(smiles_set) == 2  # Two distinct stereoisomers

    def test_two_centers(self):
        enum = StereoisomerEnumerator(compute_fingerprints=False)
        variants = enum.enumerate(SMILES_2CENTER)
        # 2 centers = up to 4 stereoisomers (some may be meso)
        assert len(variants) >= 2
        assert len(variants) <= 4

    def test_single_flips(self):
        enum = StereoisomerEnumerator(compute_fingerprints=False)
        flips = enum.enumerate_single_flips(SMILES_1CENTER)
        # 1 center → 1 flip
        assert len(flips) == 1
        assert flips[0].canonical_smiles != SMILES_1CENTER

    def test_max_centers_guard(self):
        enum = StereoisomerEnumerator(max_centers=2, compute_fingerprints=False)
        # Even if molecule has many centers, should not exceed 2^max_centers
        variants = enum.enumerate(SMILES_2CENTER)
        assert len(variants) <= 4  # 2^2

    def test_achiral_single_variant(self):
        enum = StereoisomerEnumerator(compute_fingerprints=False)
        variants = enum.enumerate(SMILES_ACHIRAL)
        assert len(variants) == 1
        assert variants[0].is_original
        assert variants[0].n_centers == 0

    def test_fingerprints_computed(self):
        enum = StereoisomerEnumerator(compute_fingerprints=True, nbits=64)
        variants = enum.enumerate(SMILES_1CENTER)
        for v in variants:
            assert v.fingerprint is not None
            assert v.fingerprint.shape == (64,)

    def test_achiral_single_flips_empty(self):
        enum = StereoisomerEnumerator(compute_fingerprints=False)
        flips = enum.enumerate_single_flips(SMILES_ACHIRAL)
        assert len(flips) == 0


# ── TestChiralityScorer ─────────────────────────────────────────────


class TestChiralityScorer:
    def test_score_without_models(self):
        """Score with no models should gracefully degrade."""
        scorer = ChiralityScorer()
        score = scorer.score(SMILES_1CENTER)

        assert isinstance(score, ChiralityScore)
        assert score.target_score == 0.0  # No target profiler
        assert score.admet_score == 1.0   # Default: no risk
        assert score.fingerprint_novelty == 0.5  # Default: neutral
        # Combined = 0.4*0 + 0.3*1 + 0.3*0.5 = 0.45
        assert abs(score.combined_score - 0.45) < 0.01

    def test_score_with_store(self):
        """Score with fingerprint store for novelty."""
        from zynerji_chirality.db.store import FingerprintStore
        from zynerji_chirality.chirality.fingerprint import chirality_fingerprint

        store = FingerprintStore(":memory:")
        # Add a molecule to the store
        fp = chirality_fingerprint(SMILES_1CENTER, nbits=128)
        store.add(
            smiles=SMILES_1CENTER,
            fingerprint=fp,
            chirality_score=0.5,
            chirality_sign=1.0,
        )

        scorer = ChiralityScorer(store=store)
        score = scorer.score(SMILES_1CENTER)

        # Should have low novelty (molecule is in the database)
        assert score.fingerprint_novelty < 0.5

    def test_rank_stereoisomers(self):
        enum = StereoisomerEnumerator(compute_fingerprints=False)
        variants = enum.enumerate(SMILES_1CENTER)

        scorer = ChiralityScorer()
        ranked = scorer.rank_stereoisomers(variants)

        assert len(ranked) == 2
        # Scores should be non-negative
        for variant, score in ranked:
            assert score.combined_score >= 0

    def test_score_batch(self):
        scorer = ChiralityScorer()
        scores = scorer.score_batch([SMILES_1CENTER, SMILES_1CENTER_S, SMILES_ACHIRAL])
        assert len(scores) == 3
        # Should be sorted by combined_score descending
        for i in range(len(scores) - 1):
            assert scores[i].combined_score >= scores[i + 1].combined_score


# ── TestChiralMMPAnalyzer ───────────────────────────────────────────


class TestChiralMMPAnalyzer:
    def test_find_beneficial_flips(self):
        scorer = ChiralityScorer()
        analyzer = ChiralMMPAnalyzer(scorer=scorer)

        flips = analyzer.find_beneficial_flips(SMILES_1CENTER)
        # With no models, both enantiomers score the same, so may be empty
        assert isinstance(flips, list)

    def test_no_beneficial_flips_achiral(self):
        scorer = ChiralityScorer()
        analyzer = ChiralMMPAnalyzer(scorer=scorer)

        flips = analyzer.find_beneficial_flips(SMILES_ACHIRAL)
        assert len(flips) == 0


# ── TestGenerativeFilter ───────────────────────────────────────────


class TestGenerativeFilter:
    def test_filter_basic(self):
        scorer = ChiralityScorer()
        gen_filter = GenerativeFilter(scorer=scorer)

        result = gen_filter.filter([SMILES_1CENTER, SMILES_1CENTER_S, SMILES_ACHIRAL])
        assert isinstance(result, FilterResult)
        assert result.input_count == 3
        assert result.output_count >= 1

    def test_deduplication(self):
        scorer = ChiralityScorer()
        gen_filter = GenerativeFilter(scorer=scorer)

        # Same molecule twice
        result = gen_filter.filter([SMILES_1CENTER, SMILES_1CENTER])
        assert result.output_count == 1  # Deduplicated

    def test_enumerate_stereo(self):
        scorer = ChiralityScorer()
        gen_filter = GenerativeFilter(
            scorer=scorer, enumerate_stereo=True,
        )

        result = gen_filter.filter([SMILES_1CENTER])
        # Should enumerate the other enantiomer too
        assert result.output_count >= 1

    def test_min_score_filter(self):
        scorer = ChiralityScorer()
        gen_filter = GenerativeFilter(scorer=scorer, min_score=999.0)

        result = gen_filter.filter([SMILES_1CENTER])
        assert result.output_count == 0  # All filtered out

    def test_filter_from_file(self, tmp_path):
        scorer = ChiralityScorer()
        gen_filter = GenerativeFilter(scorer=scorer)

        input_path = tmp_path / "input.txt"
        output_path = tmp_path / "output.tsv"
        input_path.write_text(f"{SMILES_1CENTER}\n{SMILES_ACHIRAL}\n")

        result = gen_filter.filter_from_file(input_path, output_path)
        assert result.input_count == 2
        assert output_path.exists()


# ── TestChiralOptimizer ─────────────────────────────────────────────


class TestChiralOptimizer:
    def test_optimize_single_center(self):
        scorer = ChiralityScorer()
        optimizer = ChiralOptimizer(scorer=scorer, max_iterations=3)

        result = optimizer.optimize(SMILES_1CENTER)
        assert isinstance(result, OptimizationResult)
        assert result.input_smiles == SMILES_1CENTER
        assert result.best_smiles is not None
        assert result.best_score >= 0
        assert result.n_iterations >= 1
        assert result.total_evaluated >= 1
        assert len(result.all_variants) >= 1

    def test_convergence(self):
        scorer = ChiralityScorer()
        optimizer = ChiralOptimizer(
            scorer=scorer, max_iterations=10, convergence_threshold=0.01,
        )

        result = optimizer.optimize(SMILES_1CENTER)
        # With no models, both enantiomers score the same → should converge quickly
        assert result.n_iterations <= 3

    def test_optimize_batch(self):
        scorer = ChiralityScorer()
        optimizer = ChiralOptimizer(scorer=scorer, max_iterations=2)

        results = optimizer.optimize_batch([SMILES_1CENTER, SMILES_ACHIRAL])
        assert len(results) == 2
        for r in results:
            assert isinstance(r, OptimizationResult)

    def test_optimize_achiral(self):
        scorer = ChiralityScorer()
        optimizer = ChiralOptimizer(scorer=scorer, max_iterations=2)

        result = optimizer.optimize(SMILES_ACHIRAL)
        # Achiral molecule has only one variant
        assert result.best_smiles == SMILES_ACHIRAL
