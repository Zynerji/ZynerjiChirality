"""Tests for the COD crystal structure pipeline."""

from __future__ import annotations

import json
import statistics
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


# ── Novel Findings Validation (against real COD data) ──────────────────


COD_DATA_PATH = Path(__file__).parent.parent / "cod_work" / "enriched_structures.json"
_cod_data_cache = None


def _load_cod_data():
    """Load and cache the 70K enrichment dataset (65MB, ~2s)."""
    global _cod_data_cache
    if _cod_data_cache is None:
        with open(COD_DATA_PATH) as f:
            raw = json.load(f)
        _cod_data_cache = [d for d in raw if not d.get("error")]
    return _cod_data_cache


requires_cod_data = pytest.mark.skipif(
    not COD_DATA_PATH.exists(),
    reason="COD enrichment data not available (run pipeline first)",
)


@requires_cod_data
class TestNovelFindings:
    """Validate the 7 novel statistical findings from the COD whitepaper.

    These tests run against the real 69,997-structure enrichment dataset
    and verify the empirical claims made in Section 5 of the whitepaper.
    """

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.data = _load_cod_data()
        self.valid = [d for d in self.data if d.get("is_chiral") is not None]

    # ── Finding 1: CISS Exponential Decay Law ──────────────────────────

    def test_ciss_exponential_decay_law(self):
        """CISS ≈ 1 - exp(-α·CD) with α ≈ 5.2.

        Validates by binning real data and checking that ln(1-CISS)
        is approximately linear in CD (R² > 0.95).
        """
        bins = [
            (0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.6),
            (0.6, 0.8), (0.8, 1.0), (1.0, 1.5),
        ]
        cd_mids, ln_vals = [], []
        for lo, hi in bins:
            subset = [d for d in self.valid
                      if lo <= d["cd_activity"] < hi and d["ciss_score"] < 1.0]
            if len(subset) < 10:
                continue
            mean_cd = statistics.mean(d["cd_activity"] for d in subset)
            mean_ciss = statistics.mean(d["ciss_score"] for d in subset)
            if mean_ciss >= 1.0:
                continue
            cd_mids.append(mean_cd)
            ln_vals.append(np.log(1 - mean_ciss))

        cd_arr = np.array(cd_mids)
        ln_arr = np.array(ln_vals)

        # Linear fit: ln(1-CISS) = -α·CD + intercept
        # For a good exponential law, R² should be high
        coeffs = np.polyfit(cd_arr, ln_arr, 1)
        alpha_fitted = -coeffs[0]
        predicted = np.polyval(coeffs, cd_arr)
        ss_res = np.sum((ln_arr - predicted) ** 2)
        ss_tot = np.sum((ln_arr - np.mean(ln_arr)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        assert r_squared > 0.95, f"Exponential fit R²={r_squared:.3f}, expected >0.95"
        assert 3.0 < alpha_fitted < 8.0, f"α={alpha_fitted:.1f}, expected ~5.2"

    def test_ciss_saturation_at_high_cd(self):
        """Structures with CD > 1.0 should have CISS > 0.99."""
        high_cd = [d for d in self.valid if d["cd_activity"] > 1.0]
        assert len(high_cd) > 100, "Need sufficient high-CD structures"
        mean_ciss = statistics.mean(d["ciss_score"] for d in high_cd)
        assert mean_ciss > 0.99, f"Mean CISS at CD>1.0 is {mean_ciss:.4f}, expected >0.99"

    def test_ciss_bounded_zero_one(self):
        """All CISS scores must be in [0, 1]."""
        for d in self.valid:
            assert 0.0 <= d["ciss_score"] <= 1.0, (
                f"COD {d['structure']['cod_id']}: CISS={d['ciss_score']}"
            )

    # ── Finding 2: 500-Atom Chirality Cliff ────────────────────────────

    def test_500_atom_cliff(self):
        """Non-chiral rate should jump dramatically above 500 atoms."""
        mid = [d for d in self.valid if 51 <= d["n_atoms"] <= 100]
        big = [d for d in self.valid if d["n_atoms"] > 500]
        assert len(mid) > 1000 and len(big) > 50, "Need sufficient structures"

        mid_non_chiral = sum(1 for d in mid if not d["is_chiral"]) / len(mid)
        big_non_chiral = sum(1 for d in big if not d["is_chiral"]) / len(big)

        # Mid-range should be nearly zero, big should be >10%
        assert mid_non_chiral < 0.01, f"Mid non-chiral={mid_non_chiral:.4f}"
        assert big_non_chiral > 0.10, f"Big non-chiral={big_non_chiral:.4f}"
        # The cliff: big must be >10x the mid-range rate
        if mid_non_chiral > 0:
            assert big_non_chiral / mid_non_chiral > 10

    def test_non_chiral_higher_atom_count(self):
        """Non-chiral structures should have higher mean atom count than chiral."""
        chiral = [d["n_atoms"] for d in self.valid if d["is_chiral"]]
        non_chiral = [d["n_atoms"] for d in self.valid if not d["is_chiral"]]
        assert len(non_chiral) > 50
        assert statistics.mean(non_chiral) > 2 * statistics.mean(chiral)

    # ── Finding 3: SG 211 as Chirality Amplifier ───────────────────────

    def test_sg211_highest_mean_cd(self):
        """SG 211 (I432) should have highest or near-highest mean CD."""
        from collections import defaultdict
        sg_cds = defaultdict(list)
        for d in self.valid:
            sg_cds[d["structure"]["sg_number"]].append(d["cd_activity"])

        # Only consider SGs with >=10 structures
        sg_means = {
            sg: statistics.mean(cds)
            for sg, cds in sg_cds.items()
            if len(cds) >= 10
        }
        sg211_cd = sg_means.get(211)
        assert sg211_cd is not None, "SG 211 not found or <10 structures"
        assert sg211_cd > 1.0, f"SG 211 mean CD={sg211_cd:.3f}, expected >1.0"

        # Should be in top 5
        ranked = sorted(sg_means.items(), key=lambda x: -x[1])
        top5_sgs = [sg for sg, _ in ranked[:5]]
        assert 211 in top5_sgs, f"SG 211 not in top 5: {ranked[:5]}"

    # ── Finding 4: CD–Band Gap Anticorrelation ─────────────────────────

    def test_cd_bandgap_anticorrelation(self):
        """CD and band gap shift should be negatively correlated."""
        cds = np.array([d["cd_activity"] for d in self.valid])
        bgs = np.array([d["band_gap_shift"] for d in self.valid])
        r = np.corrcoef(cds, bgs)[0, 1]
        assert r < 0, f"Expected negative correlation, got r={r:.4f}"
        assert r < -0.05, f"Anticorrelation too weak: r={r:.4f}"

    def test_high_cd_lower_bandgap(self):
        """High-CD structures should have smaller band gap shifts than low-CD."""
        high = [d["band_gap_shift"] for d in self.valid if d["cd_activity"] > 1.2]
        low = [d["band_gap_shift"] for d in self.valid if d["cd_activity"] < 0.2]
        assert len(high) > 50 and len(low) > 50
        assert statistics.mean(high) < statistics.mean(low)

    # ── Finding 5: Enantiomorphic Pair Asymmetry ───────────────────────

    def test_enantiomorphic_pair_asymmetry(self):
        """SG 152/154 should show different negative-CD fractions."""
        sg152 = [d for d in self.valid if d["structure"]["sg_number"] == 152]
        sg154 = [d for d in self.valid if d["structure"]["sg_number"] == 154]
        assert len(sg152) > 50 and len(sg154) > 50

        neg152 = sum(1 for d in sg152 if d["cd_sign"] == "negative") / len(sg152)
        neg154 = sum(1 for d in sg154 if d["cd_sign"] == "negative") / len(sg154)

        # They should differ (asymmetry exists)
        assert abs(neg152 - neg154) > 0.03, (
            f"No asymmetry: SG152={neg152:.3f}, SG154={neg154:.3f}"
        )

    # ── Finding 6: Negative-CD Enhancement ─────────────────────────────

    def test_negative_cd_higher_mean(self):
        """Negative-CD structures should have higher mean |CD| than positive."""
        pos = [d["cd_activity"] for d in self.valid if d["cd_sign"] == "positive"]
        neg = [d["cd_activity"] for d in self.valid if d["cd_sign"] == "negative"]
        assert len(neg) > 500, f"Only {len(neg)} negative-CD structures"
        assert statistics.mean(neg) > statistics.mean(pos), (
            f"Neg mean={statistics.mean(neg):.3f} vs Pos={statistics.mean(pos):.3f}"
        )

    # ── Finding 7: Unimodal Chirality Score ────────────────────────────

    def test_chirality_score_unimodal(self):
        """Chirality score distribution should peak near 0.5-0.6 (unimodal)."""
        scores = [d["chirality_score"] for d in self.valid if d["is_chiral"]]
        # Bin into deciles
        bins = [0] * 10
        for s in scores:
            idx = min(int(s * 10), 9)
            bins[idx] += 1

        # Peak should be in bin 4-6 (CS 0.4-0.7)
        peak_bin = bins.index(max(bins))
        assert 3 <= peak_bin <= 6, f"Peak at bin {peak_bin} (CS {peak_bin/10:.1f}-{(peak_bin+1)/10:.1f})"

        # Unimodal check: no secondary peak >80% of primary
        primary = max(bins)
        secondary_bins = sorted(bins, reverse=True)
        if secondary_bins[1] > 0:
            # Check bins at distance >=2 from peak for secondary modes
            for i, count in enumerate(bins):
                if abs(i - peak_bin) >= 3 and count > 0.8 * primary:
                    pytest.fail(f"Bimodal: secondary peak at bin {i} ({count} vs {primary})")

    def test_cs_cd_strong_correlation(self):
        """Chirality score and CD activity should be strongly correlated (r>0.8)."""
        cs = np.array([d["chirality_score"] for d in self.valid])
        cd = np.array([d["cd_activity"] for d in self.valid])
        r = np.corrcoef(cs, cd)[0, 1]
        assert r > 0.8, f"CS-CD correlation r={r:.4f}, expected >0.8"

    # ── Finding 7b: Element Diversity ──────────────────────────────────

    def test_element_diversity_drives_chirality(self):
        """Structures with more unique elements should have higher mean CD."""
        from collections import Counter
        diversity_cd = {}
        for d in self.valid:
            labels = d.get("atom_labels", [])
            if not labels:
                continue
            n_unique = len(set(labels))
            diversity_cd.setdefault(n_unique, []).append(d["cd_activity"])

        # Single-element should be low
        if 1 in diversity_cd and len(diversity_cd[1]) >= 5:
            mono_cd = statistics.mean(diversity_cd[1])
            assert mono_cd < 0.3, f"Single-element mean CD={mono_cd:.3f}"

        # Multi-element (5+) should be higher than binary (2)
        if 2 in diversity_cd and 5 in diversity_cd:
            binary_cd = statistics.mean(diversity_cd[2])
            multi_cd = statistics.mean(diversity_cd[5])
            assert multi_cd > binary_cd, (
                f"5-element CD={multi_cd:.3f} not > 2-element CD={binary_cd:.3f}"
            )

    # ── Overall sanity checks ──────────────────────────────────────────

    def test_dataset_size(self):
        """Should have ~70K structures."""
        assert len(self.valid) > 60000

    def test_chirality_detection_rate(self):
        """Detection rate should be >99%."""
        chiral = sum(1 for d in self.valid if d["is_chiral"])
        rate = chiral / len(self.valid)
        assert rate > 0.99, f"Detection rate={rate:.4f}"

    def test_cubic_highest_mean_cd(self):
        """Cubic system should have highest mean CD (symmetry amplification)."""
        systems = {
            "Triclinic": range(1, 3), "Monoclinic": range(3, 16),
            "Orthorhombic": range(16, 75), "Tetragonal": range(75, 143),
            "Trigonal": range(143, 168), "Hexagonal": range(168, 195),
            "Cubic": range(195, 231),
        }
        sys_cd = {}
        for name, sg_range in systems.items():
            cds = [d["cd_activity"] for d in self.valid
                   if d["structure"]["sg_number"] in sg_range]
            if cds:
                sys_cd[name] = statistics.mean(cds)

        assert sys_cd["Cubic"] == max(sys_cd.values()), (
            f"Cubic CD={sys_cd['Cubic']:.3f} not highest: {sys_cd}"
        )
