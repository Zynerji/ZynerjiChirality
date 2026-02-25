"""Tests for the COD crystal structure pipeline."""

from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from zynerji_chirality.cod.download import (
    CODDownloader,
    CrystalStructure,
    SOHNCKE_GROUPS,
)
from zynerji_chirality.cod.enricher import CODEnricher, CrystalEnrichment
from zynerji_chirality.cod.screen import CrystalScreen, CrystalScreeningHit


# ── CrystalStructure dataclass tests ─────────────────────────────────────


class TestCrystalStructure:

    def test_to_dict_from_dict_roundtrip(self):
        """CrystalStructure should survive JSON roundtrip."""
        cs = CrystalStructure(
            cod_id="1234567",
            formula="Si O2",
            space_group="P 21 21 21",
            sg_number=19,
            a=5.0, b=5.0, c=5.0,
            alpha=90.0, beta=90.0, gamma=90.0,
            cif_url="https://www.crystallography.net/cod/1234567.cif",
        )
        d = cs.to_dict()
        cs2 = CrystalStructure.from_dict(d)
        assert cs2.cod_id == cs.cod_id
        assert cs2.formula == cs.formula
        assert cs2.sg_number == cs.sg_number
        assert cs2.is_chiral_sg == True  # SG 19 is Sohncke

    def test_non_chiral_sg(self):
        """Non-Sohncke space group should not be flagged chiral."""
        cs = CrystalStructure(
            cod_id="9999999",
            formula="Na Cl",
            space_group="Fm-3m",
            sg_number=225,  # NOT Sohncke
            a=5.6, b=5.6, c=5.6,
            alpha=90.0, beta=90.0, gamma=90.0,
            cif_url="https://example.com/9999999.cif",
        )
        assert cs.is_chiral_sg == False

    def test_sohncke_groups_count(self):
        """There should be exactly 65 Sohncke space groups."""
        assert len(SOHNCKE_GROUPS) == 65


# ── CODDownloader tests ──────────────────────────────────────────────────


class TestCODDownloader:

    def test_rate_limiting(self):
        """Downloader should respect rate limiting delay."""
        import time
        downloader = CODDownloader(delay=0.1)
        downloader._last_request = time.monotonic()
        t0 = time.monotonic()
        downloader._rate_limit()
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.05  # Some delay observed

    def test_save_load_structures(self, tmp_path):
        """Save/load roundtrip should preserve structures."""
        structures = [
            CrystalStructure(
                cod_id="1000000",
                formula="C H4",
                space_group="P 1",
                sg_number=1,
                a=3.0, b=3.0, c=3.0,
                alpha=90.0, beta=90.0, gamma=90.0,
                cif_url="https://example.com/1000000.cif",
            ),
            CrystalStructure(
                cod_id="2000000",
                formula="Si O2",
                space_group="P 21",
                sg_number=4,
                a=5.0, b=5.0, c=5.0,
                alpha=90.0, beta=90.0, gamma=90.0,
                cif_url="https://example.com/2000000.cif",
            ),
        ]
        path = str(tmp_path / "structures.json")
        CODDownloader.save_structures(structures, path)
        loaded = CODDownloader.load_structures(path)
        assert len(loaded) == 2
        assert loaded[0].cod_id == "1000000"
        assert loaded[1].formula == "Si O2"

    @patch("zynerji_chirality.cod.download.urlopen")
    def test_search_by_space_group_mock(self, mock_urlopen):
        """Mocked COD API response should parse to structures."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps([
            {
                "file": "1234567",
                "formula": "C6 H12 O6",
                "sg": "P 21 21 21",
                "a": 5.0, "b": 6.0, "c": 7.0,
                "alpha": 90, "beta": 90, "gamma": 90,
            }
        ]).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        downloader = CODDownloader(delay=0.0)
        structures = downloader.search_by_space_group(19, limit=10)
        assert len(structures) == 1
        assert structures[0].cod_id == "1234567"
        assert structures[0].formula == "C6 H12 O6"


# ── CODEnricher tests ────────────────────────────────────────────────────


class TestCODEnricher:

    def test_enrich_with_test_cif(self, tmp_path):
        """Enricher should analyze a CIF file successfully."""
        cif_content = """data_test
_cell_length_a  5.0
_cell_length_b  5.0
_cell_length_c  5.0
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0

loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si 0.0 0.0 0.0
Si 0.25 0.25 0.25
O  0.5 0.0 0.0
"""
        cif_dir = tmp_path / "cifs"
        cif_dir.mkdir()
        cif_file = cif_dir / "9999999.cif"
        cif_file.write_text(cif_content)

        structure = CrystalStructure(
            cod_id="9999999",
            formula="Si2 O",
            space_group="P 21 21 21",
            sg_number=19,
            a=5.0, b=5.0, c=5.0,
            alpha=90.0, beta=90.0, gamma=90.0,
            cif_url="https://example.com/9999999.cif",
        )

        enricher = CODEnricher(cutoff=3.0)
        enrichment = enricher.enrich_structure(structure, str(cif_dir))

        assert isinstance(enrichment, CrystalEnrichment)
        assert enrichment.n_atoms == 3
        assert enrichment.error is None
        assert isinstance(enrichment.chirality_score, float)
        assert isinstance(enrichment.ciss_score, float)
        assert 0.0 <= enrichment.ciss_score <= 1.0

    def test_enrichment_to_dict_from_dict(self):
        """Enrichment should survive serialization roundtrip."""
        structure = CrystalStructure(
            cod_id="1111111",
            formula="Al O3",
            space_group="R 3",
            sg_number=146,
            a=4.0, b=4.0, c=4.0,
            alpha=90.0, beta=90.0, gamma=120.0,
            cif_url="https://example.com/1111111.cif",
        )
        enrichment = CrystalEnrichment(
            structure=structure,
            atom_labels=["Al", "O", "O", "O"],
            n_atoms=4,
            cd_activity=0.35,
            cd_sign="positive",
            ciss_score=0.6,
            optical_rotation="dextrorotatory (+)",
            chirality_score=0.15,
            chirality_sign=1.0,
            is_chiral=True,
            band_gap_shift=0.02,
        )
        d = enrichment.to_dict()
        e2 = CrystalEnrichment.from_dict(d)
        assert e2.structure.cod_id == "1111111"
        assert abs(e2.cd_activity - 0.35) < 1e-6
        assert e2.is_chiral == True

    def test_enrichment_checkpoint(self, tmp_path):
        """Batch enrichment checkpoint should save/load correctly."""
        structure = CrystalStructure(
            cod_id="5555555",
            formula="Fe O",
            space_group="P 1",
            sg_number=1,
            a=3.0, b=3.0, c=3.0,
            alpha=90.0, beta=90.0, gamma=90.0,
            cif_url="https://example.com/5555555.cif",
        )
        enrichment = CrystalEnrichment(
            structure=structure,
            atom_labels=["Fe", "O"],
            n_atoms=2,
            cd_activity=0.1,
            cd_sign="negative",
            ciss_score=0.3,
            optical_rotation="levorotatory (-)",
            chirality_score=0.05,
            chirality_sign=-1.0,
            is_chiral=True,
            band_gap_shift=0.01,
        )
        path = str(tmp_path / "enriched.json")
        CODEnricher.save_enriched([enrichment], path)
        loaded = CODEnricher.load_enriched(path)
        assert len(loaded) == 1
        assert loaded[0].structure.cod_id == "5555555"


# ── CrystalScreen tests ─────────────────────────────────────────────────


def _make_enrichment(cod_id, cd_activity=0.1, ciss_score=0.3, chirality_score=0.05,
                     sg_number=19, formula="Si O2", is_chiral=True, error=None):
    """Helper to create test enrichments."""
    return CrystalEnrichment(
        structure=CrystalStructure(
            cod_id=cod_id,
            formula=formula,
            space_group=f"SG{sg_number}",
            sg_number=sg_number,
            a=5.0, b=5.0, c=5.0,
            alpha=90.0, beta=90.0, gamma=90.0,
            cif_url=f"https://example.com/{cod_id}.cif",
        ),
        atom_labels=["Si", "O", "O"],
        n_atoms=3,
        cd_activity=cd_activity,
        cd_sign="positive" if cd_activity > 0.1 else "inactive",
        ciss_score=ciss_score,
        optical_rotation="dextrorotatory (+)" if is_chiral else "inactive",
        chirality_score=chirality_score,
        chirality_sign=1.0 if is_chiral else 0.0,
        is_chiral=is_chiral,
        band_gap_shift=0.01,
        error=error,
    )


class TestCrystalScreen:

    def test_screen_high_cd(self):
        """Should filter by CD activity threshold."""
        enrichments = [
            _make_enrichment("001", cd_activity=0.5),
            _make_enrichment("002", cd_activity=0.1),
            _make_enrichment("003", cd_activity=0.8),
        ]
        screen = CrystalScreen(enrichments)
        hits = screen.screen_high_cd(threshold=0.3)
        assert len(hits) == 2
        assert hits[0].structure.cod_id == "003"  # Highest first

    def test_screen_high_ciss(self):
        """Should filter by CISS score threshold."""
        enrichments = [
            _make_enrichment("001", ciss_score=0.7),
            _make_enrichment("002", ciss_score=0.2),
            _make_enrichment("003", ciss_score=0.9),
        ]
        screen = CrystalScreen(enrichments)
        hits = screen.screen_high_ciss(threshold=0.5)
        assert len(hits) == 2
        assert hits[0].structure.cod_id == "003"

    def test_screen_all_deduplicates(self):
        """Screen all should deduplicate by COD ID."""
        enrichments = [
            _make_enrichment("001", cd_activity=0.5, ciss_score=0.7),
        ]
        screen = CrystalScreen(enrichments)
        hits = screen.screen_all()
        # Same structure may hit multiple screens but should be deduped
        cod_ids = [h.structure.cod_id for h in hits]
        assert len(cod_ids) == len(set(cod_ids))

    def test_screen_excludes_errors(self):
        """Structures with errors should be excluded."""
        enrichments = [
            _make_enrichment("001", cd_activity=0.5, error="parse failed"),
            _make_enrichment("002", cd_activity=0.8),
        ]
        screen = CrystalScreen(enrichments)
        hits = screen.screen_high_cd(threshold=0.3)
        assert len(hits) == 1
        assert hits[0].structure.cod_id == "002"

    def test_report_generates_text(self):
        """Report should produce non-empty text."""
        enrichments = [
            _make_enrichment("001", cd_activity=0.5),
            _make_enrichment("002", cd_activity=0.8),
        ]
        screen = CrystalScreen(enrichments)
        hits = screen.screen_all()
        report = screen.report(hits)
        assert "COD Crystal Chirality Screening Report" in report
        assert "hits" in report.lower()

    def test_report_saves_to_file(self, tmp_path):
        """Report should save to file when output path given."""
        enrichments = [_make_enrichment("001", cd_activity=0.5)]
        screen = CrystalScreen(enrichments)
        hits = screen.screen_all()
        path = str(tmp_path / "report.txt")
        screen.report(hits, output=path)
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "COD" in content
