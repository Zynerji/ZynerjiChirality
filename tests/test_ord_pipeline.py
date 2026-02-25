"""Tests for the ORD enantioselectivity pipeline."""

from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path

from zynerji_chirality.ord.download import ORDDownloader, ORDReaction
from zynerji_chirality.ord.extractor import ORDExtractor, EnantioselectiveReaction
from zynerji_chirality.ord.enricher import ORDEnricher, ReactionEnrichment
from zynerji_chirality.ord.screen import CatalystScreen, CatalystScreeningHit


# ── ORDReaction dataclass tests ──────────────────────────────────────────


class TestORDReaction:

    def test_to_dict_from_dict_roundtrip(self):
        """ORDReaction should survive JSON roundtrip."""
        rxn = ORDReaction(
            reaction_id="ord-abc123",
            reaction_smiles="CC(=O)CC(=O)OCC>>C[C@@H](O)CC(=O)OCC",
            catalyst_smiles="C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",
            substrate_smiles="CC(=O)CC(=O)OCC",
            product_smiles="C[C@@H](O)CC(=O)OCC",
            ee_percent=99.0,
            yield_percent=95.0,
            temperature="25 C",
            solvent="methanol",
            source_doi="10.1234/test",
        )
        d = rxn.to_dict()
        rxn2 = ORDReaction.from_dict(d)
        assert rxn2.reaction_id == rxn.reaction_id
        assert rxn2.ee_percent == 99.0
        assert rxn2.catalyst_smiles == rxn.catalyst_smiles

    def test_save_load_reactions(self, tmp_path):
        """Save/load roundtrip should preserve reactions."""
        reactions = [
            ORDReaction(
                reaction_id="rxn-001",
                reaction_smiles="A>>B",
                ee_percent=95.0,
            ),
            ORDReaction(
                reaction_id="rxn-002",
                reaction_smiles="C>>D",
                ee_percent=88.0,
            ),
        ]
        path = str(tmp_path / "reactions.json")
        ORDDownloader.save_reactions(reactions, path)
        loaded = ORDDownloader.load_reactions(path)
        assert len(loaded) == 2
        assert loaded[0].reaction_id == "rxn-001"
        assert loaded[1].ee_percent == 88.0


# ── ORDExtractor tests ───────────────────────────────────────────────────


def _make_ord_reaction(rxn_id, catalyst=None, substrate=None, ee=None):
    """Helper to create test reactions."""
    return ORDReaction(
        reaction_id=rxn_id,
        reaction_smiles=f"{substrate or 'A'}>>{catalyst or 'B'}",
        catalyst_smiles=catalyst,
        substrate_smiles=substrate,
        ee_percent=ee,
    )


class TestORDExtractor:

    def test_extract_ee_reactions(self):
        """Should filter to reactions with valid ee data."""
        reactions = [
            _make_ord_reaction("001", ee=95.0, catalyst="CC", substrate="CCO"),
            _make_ord_reaction("002", ee=None),  # No ee
            _make_ord_reaction("003", ee=150.0),  # Invalid ee (>100)
            _make_ord_reaction("004", ee=50.0, catalyst="CC", substrate="CCC"),
        ]
        extractor = ORDExtractor()
        enriched = extractor.extract_ee_reactions(reactions)
        assert len(enriched) == 2
        assert enriched[0].reaction_id == "001"
        assert enriched[1].reaction_id == "004"

    def test_extract_catalyst_substrate_pairs(self):
        """Should format for EEPredictor.fit()."""
        reactions = [
            EnantioselectiveReaction(
                reaction_id="001",
                reaction_smiles="",
                catalyst_smiles="CC",
                substrate_smiles="CCO",
                product_smiles=None,
                ee_percent=95.0,
                yield_percent=None,
                temperature="",
                solvent="",
                source_doi="",
            ),
            EnantioselectiveReaction(
                reaction_id="002",
                reaction_smiles="",
                catalyst_smiles=None,  # No catalyst
                substrate_smiles="CCC",
                product_smiles=None,
                ee_percent=50.0,
                yield_percent=None,
                temperature="",
                solvent="",
                source_doi="",
            ),
        ]
        extractor = ORDExtractor()
        pairs = extractor.extract_catalyst_substrate_pairs(reactions)
        assert len(pairs) == 1  # Only one with both catalyst + substrate
        assert pairs[0]["catalyst"] == "CC"
        assert pairs[0]["ee"] == 95.0

    def test_group_by_catalyst(self):
        """Should group reactions by catalyst SMILES."""
        reactions = [
            EnantioselectiveReaction(
                reaction_id="001", reaction_smiles="", catalyst_smiles="CC",
                substrate_smiles="CCO", product_smiles=None, ee_percent=95.0,
                yield_percent=None, temperature="", solvent="", source_doi="",
            ),
            EnantioselectiveReaction(
                reaction_id="002", reaction_smiles="", catalyst_smiles="CC",
                substrate_smiles="CCC", product_smiles=None, ee_percent=88.0,
                yield_percent=None, temperature="", solvent="", source_doi="",
            ),
            EnantioselectiveReaction(
                reaction_id="003", reaction_smiles="", catalyst_smiles="CCC",
                substrate_smiles="CCO", product_smiles=None, ee_percent=70.0,
                yield_percent=None, temperature="", solvent="", source_doi="",
            ),
        ]
        groups = ORDExtractor.group_by_catalyst(reactions)
        assert len(groups) == 2
        assert len(groups["CC"]) == 2
        assert len(groups["CCC"]) == 1


# ── ORDEnricher tests ────────────────────────────────────────────────────


class TestORDEnricher:

    def test_enrich_reaction_with_fingerprint(self):
        """Should compute combined fingerprint for catalyst+substrate."""
        rxn = EnantioselectiveReaction(
            reaction_id="001",
            reaction_smiles="",
            catalyst_smiles="N[C@@H](C)C(=O)O",  # L-alanine
            substrate_smiles="CC(=O)CC(=O)OCC",   # ethyl acetoacetate
            product_smiles=None,
            ee_percent=99.0,
            yield_percent=None,
            temperature="",
            solvent="",
            source_doi="",
        )
        enricher = ORDEnricher(nbits=128)
        result = enricher.enrich_reaction(rxn)
        assert isinstance(result, ReactionEnrichment)
        assert result.catalyst_fp is not None
        assert result.substrate_fp is not None
        assert result.combined_fp is not None
        assert len(result.combined_fp) == 128 * 4  # 4*nbits

    def test_enrich_reaction_no_catalyst(self):
        """Missing catalyst should result in None combined FP."""
        rxn = EnantioselectiveReaction(
            reaction_id="002",
            reaction_smiles="",
            catalyst_smiles=None,
            substrate_smiles="CCO",
            product_smiles=None,
            ee_percent=50.0,
            yield_percent=None,
            temperature="",
            solvent="",
            source_doi="",
        )
        enricher = ORDEnricher(nbits=64)
        result = enricher.enrich_reaction(rxn)
        assert result.catalyst_fp is None
        assert result.combined_fp is None

    def test_enrichment_to_dict_from_dict(self):
        """ReactionEnrichment should survive serialization."""
        rxn = EnantioselectiveReaction(
            reaction_id="001",
            reaction_smiles="",
            catalyst_smiles="CC",
            substrate_smiles="CCO",
            product_smiles=None,
            ee_percent=95.0,
            yield_percent=None,
            temperature="",
            solvent="",
            source_doi="",
        )
        enrichment = ReactionEnrichment(
            reaction=rxn,
            catalyst_fp=np.ones(64),
            substrate_fp=np.zeros(64),
            combined_fp=np.ones(256),
        )
        d = enrichment.to_dict()
        e2 = ReactionEnrichment.from_dict(d)
        assert e2.reaction.reaction_id == "001"
        assert np.allclose(e2.catalyst_fp, np.ones(64))
        assert len(e2.combined_fp) == 256

    def test_enrichment_checkpoint(self, tmp_path):
        """Save/load checkpoint should preserve enrichments."""
        rxn = EnantioselectiveReaction(
            reaction_id="001",
            reaction_smiles="",
            catalyst_smiles="CC",
            substrate_smiles="CCO",
            product_smiles=None,
            ee_percent=90.0,
            yield_percent=None,
            temperature="",
            solvent="",
            source_doi="",
        )
        enrichment = ReactionEnrichment(
            reaction=rxn,
            catalyst_fp=np.ones(64),
            substrate_fp=np.zeros(64),
            combined_fp=np.ones(256),
        )
        path = str(tmp_path / "enriched.json")
        ORDEnricher.save_enriched([enrichment], path)
        loaded = ORDEnricher.load_enriched(path)
        assert len(loaded) == 1
        assert loaded[0].reaction.ee_percent == 90.0


# ── CatalystScreen tests ────────────────────────────────────────────────


def _make_enrichment(rxn_id, ee=95.0, catalyst="CC", substrate="CCO"):
    """Helper to create test enrichments."""
    rxn = EnantioselectiveReaction(
        reaction_id=rxn_id,
        reaction_smiles="",
        catalyst_smiles=catalyst,
        substrate_smiles=substrate,
        product_smiles=None,
        ee_percent=ee,
        yield_percent=None,
        temperature="",
        solvent="",
        source_doi="",
    )
    return ReactionEnrichment(
        reaction=rxn,
        catalyst_fp=np.ones(64),
        substrate_fp=np.zeros(64),
        combined_fp=np.ones(256),
    )


class TestCatalystScreen:

    def test_screen_high_ee(self):
        """Should filter by ee threshold."""
        enrichments = [
            _make_enrichment("001", ee=95.0),
            _make_enrichment("002", ee=50.0),
            _make_enrichment("003", ee=99.0),
        ]
        screen = CatalystScreen(enrichments)
        hits = screen.screen_high_ee(threshold=90.0)
        assert len(hits) == 2
        assert hits[0].reaction.reaction_id == "003"  # Highest first

    def test_screen_all_deduplicates(self):
        """Screen all should deduplicate by reaction_id."""
        enrichments = [
            _make_enrichment("001", ee=95.0),
        ]
        screen = CatalystScreen(enrichments)
        hits = screen.screen_all()
        rxn_ids = [h.reaction.reaction_id for h in hits]
        assert len(rxn_ids) == len(set(rxn_ids))

    def test_report_generates_text(self):
        """Report should produce non-empty text."""
        enrichments = [
            _make_enrichment("001", ee=95.0),
            _make_enrichment("002", ee=99.0),
        ]
        screen = CatalystScreen(enrichments)
        hits = screen.screen_all()
        report = screen.report(hits)
        assert "ORD Enantioselectivity Screening Report" in report
        assert "hits" in report.lower()

    def test_report_saves_to_file(self, tmp_path):
        """Report should save to file."""
        enrichments = [_make_enrichment("001", ee=95.0)]
        screen = CatalystScreen(enrichments)
        hits = screen.screen_all()
        path = str(tmp_path / "report.txt")
        screen.report(hits, output=path)
        assert Path(path).exists()

    def test_report_summary_statistics(self):
        """Report should include summary statistics."""
        enrichments = [
            _make_enrichment("001", ee=95.0),
            _make_enrichment("002", ee=50.0),
            _make_enrichment("003", ee=99.0),
        ]
        screen = CatalystScreen(enrichments)
        hits = screen.screen_all()
        report = screen.report(hits)
        assert "Mean ee" in report
        assert "Median ee" in report


# ── Train model test ─────────────────────────────────────────────────────


class TestTrainModel:

    def test_train_ee_model_synthetic(self):
        """Should train on synthetic data and return predictor."""
        from zynerji_chirality.ord.screen import train_ee_model

        # Create enough enrichments with both catalyst and substrate
        enrichments = []
        smiles_pairs = [
            ("N[C@@H](C)C(=O)O", "CC(=O)CC(=O)OCC"),
            ("N[C@@H](C)C(=O)O", "CC(=O)CC(=O)OC"),
            ("N[C@@H](CC1=CC=CC=C1)C(=O)O", "CC(=O)CC(=O)OCC"),
            ("N[C@@H](CC(C)C)C(=O)O", "CC(=O)CC(=O)OCC"),
            ("N[C@@H](CO)C(=O)O", "CC(=O)CC(=O)OCC"),
            ("N[C@@H](CS)C(=O)O", "CC(=O)OC"),
        ]
        for i, (cat, sub) in enumerate(smiles_pairs):
            rxn = EnantioselectiveReaction(
                reaction_id=f"rxn-{i:03d}",
                reaction_smiles="",
                catalyst_smiles=cat,
                substrate_smiles=sub,
                product_smiles=None,
                ee_percent=90.0 + i,
                yield_percent=None,
                temperature="",
                solvent="",
                source_doi="",
            )
            enrichments.append(ReactionEnrichment(
                reaction=rxn,
                catalyst_fp=None,
                substrate_fp=None,
                combined_fp=None,
            ))

        predictor = train_ee_model(enrichments)
        assert predictor is not None

    def test_train_ee_model_too_few_raises(self):
        """Should raise ValueError with fewer than 5 reactions."""
        from zynerji_chirality.ord.screen import train_ee_model

        enrichments = [
            ReactionEnrichment(
                reaction=EnantioselectiveReaction(
                    reaction_id="rxn-001",
                    reaction_smiles="",
                    catalyst_smiles="CC",
                    substrate_smiles="CCO",
                    product_smiles=None,
                    ee_percent=95.0,
                    yield_percent=None,
                    temperature="",
                    solvent="",
                    source_doi="",
                ),
                catalyst_fp=None,
                substrate_fp=None,
                combined_fp=None,
            )
        ]
        with pytest.raises(ValueError, match="at least 5"):
            train_ee_model(enrichments)
