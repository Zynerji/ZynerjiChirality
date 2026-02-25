"""Tests for the chiral materials science module."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from zynerji_chirality.materials.detector import (
    ChiralMaterialDetector,
    MaterialChiralityResult,
)
from zynerji_chirality.materials.properties import (
    estimate_cd_activity,
    estimate_ciss_score,
    classify_optical_rotation,
    band_gap_chirality_shift,
)
from zynerji_chirality.materials.crystal import (
    perovskite_graph,
    polymer_repeat_graph,
    _parse_cell_params,
    _frac_to_cart_matrix,
    _parse_atom_sites,
    cif_to_adjacency,
)


@pytest.fixture
def detector():
    return ChiralMaterialDetector()


# ── ChiralMaterialDetector tests ──────────────────────────────────────────

class TestChiralMaterialDetector:

    def test_detect_chiral_molecule(self, detector):
        """L-alanine should be detected as chiral with non-zero CD."""
        result = detector.detect_material("N[C@@H](C)C(=O)O")
        assert isinstance(result, MaterialChiralityResult)
        assert result.is_chiral
        assert result.chirality_score > 0
        assert result.cd_activity > 0
        assert result.cd_sign in ("positive", "negative")
        assert result.material_class == "molecular"

    def test_detect_achiral_molecule(self, detector):
        """Glycine (achiral) should have near-zero CD activity."""
        result = detector.detect_material("NCC(=O)O")
        assert not result.is_chiral
        assert result.chirality_score < 0.01
        assert result.optical_rotation == "inactive"

    def test_cd_sign_opposite_enantiomers(self, detector):
        """L-alanine and D-alanine should have opposite chirality signs and rotation."""
        l_result = detector.detect_material("N[C@@H](C)C(=O)O")
        d_result = detector.detect_material("N[C@H](C)C(=O)O")
        # Enantiomers have opposite chirality signs
        if l_result.is_chiral and d_result.is_chiral:
            assert l_result.chirality_sign * d_result.chirality_sign < 0
        # Optical rotation should be opposite
        if l_result.optical_rotation != "inactive" and d_result.optical_rotation != "inactive":
            assert l_result.optical_rotation != d_result.optical_rotation

    def test_ciss_score_range(self, detector):
        """CISS score should be in [0, 1]."""
        result = detector.detect_material("N[C@@H](C)C(=O)O")
        assert 0.0 <= result.ciss_score <= 1.0

    def test_optical_rotation_classification(self, detector):
        """Should classify optical rotation direction."""
        result = detector.detect_material("N[C@@H](C)C(=O)O")
        assert result.optical_rotation in (
            "dextrorotatory (+)", "levorotatory (-)", "inactive"
        )
        if result.is_chiral:
            assert result.optical_rotation != "inactive"

    def test_detect_from_adjacency(self, detector):
        """Raw adjacency matrix input should work."""
        # Simple 4-node graph with asymmetric weights
        adj = np.array([
            [0, 1, 0, 0.5],
            [1, 0, 1.5, 0],
            [0, 1.5, 0, 1],
            [0.5, 0, 1, 0],
        ])
        result = detector.detect_from_adjacency(adj, atom_labels=["C", "C", "N", "O"])
        assert isinstance(result, MaterialChiralityResult)
        assert isinstance(result.chirality_score, float)
        assert 0.0 <= result.ciss_score <= 1.0

    def test_detect_from_sparse_adjacency(self, detector):
        """Sparse csr_matrix adjacency should work."""
        adj = csr_matrix(np.array([
            [0, 1, 0.5],
            [1, 0, 1],
            [0.5, 1, 0],
        ]))
        result = detector.detect_from_adjacency(adj)
        assert isinstance(result, MaterialChiralityResult)

    def test_predict_cd_spectrum(self, detector):
        """predict_cd_spectrum should return spectral features."""
        cd = detector.predict_cd_spectrum("N[C@@H](C)C(=O)O")
        assert "peaks" in cd
        assert "sign" in cd
        assert "intensity" in cd
        assert isinstance(cd["peaks"], list)

    def test_predict_spin_selectivity(self, detector):
        """predict_spin_selectivity should return float in [0, 1]."""
        score = detector.predict_spin_selectivity("N[C@@H](C)C(=O)O")
        assert 0.0 <= score <= 1.0

    def test_band_gap_chirality_shift(self, detector):
        """Band gap shift should return a float."""
        result = detector.detect_material("N[C@@H](C)C(=O)O")
        assert isinstance(result.band_gap_shift, float)

    def test_material_class_molecular(self, detector):
        """SMILES input should classify as molecular."""
        result = detector.detect_material("N[C@@H](C)C(=O)O")
        assert result.material_class == "molecular"

    def test_material_class_from_adjacency(self, detector):
        """Large adjacency should classify as crystalline."""
        n = 25
        # Random symmetric adjacency
        rng = np.random.RandomState(42)
        adj = rng.rand(n, n)
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0)
        adj[adj < 0.7] = 0  # Sparsify
        result = detector.detect_from_adjacency(adj)
        assert result.material_class == "crystalline"

    def test_smiles_field_set(self, detector):
        """SMILES input should populate smiles field."""
        result = detector.detect_material("N[C@@H](C)C(=O)O")
        assert result.smiles == "N[C@@H](C)C(=O)O"


# ── Crystal utilities ─────────────────────────────────────────────────────

class TestCrystalUtils:

    def test_perovskite_graph(self):
        """Perovskite graph should have 5 nodes and valid adjacency."""
        adj, labels = perovskite_graph("Cs", "Pb", "I")
        assert adj.shape == (5, 5)
        assert len(labels) == 5
        assert labels[0] == "Cs"
        assert labels[1] == "Pb"
        assert labels[2] == "I"
        # Should be symmetric
        diff = abs(adj - adj.T)
        assert diff.max() < 1e-10

    def test_perovskite_custom_lattice(self):
        """Custom lattice parameter should change edge weights."""
        adj1, _ = perovskite_graph("Cs", "Pb", "I", a=3.9)
        adj2, _ = perovskite_graph("Cs", "Pb", "I", a=5.0)
        # Larger lattice = smaller weights (1/d)
        assert adj2.sum() < adj1.sum()

    def test_polymer_repeat_graph(self):
        """Should build oligomer from monomer SMILES."""
        mol = polymer_repeat_graph("C=C", n_repeat=3)
        assert mol is not None
        assert mol.GetNumAtoms() > 2

    def test_cif_parser_cell_params(self):
        """Should parse CIF cell parameters."""
        cif_text = """
_cell_length_a  5.431
_cell_length_b  5.431
_cell_length_c  5.431
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
"""
        params = _parse_cell_params(cif_text)
        assert abs(params["a"] - 5.431) < 1e-6
        assert abs(params["alpha"] - 90.0) < 1e-6

    def test_frac_to_cart_cubic(self):
        """Cubic cell should give orthogonal axes."""
        cell = {"a": 5.0, "b": 5.0, "c": 5.0,
                "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
        M = _frac_to_cart_matrix(cell)
        # For cubic: should be diagonal
        assert abs(M[0, 0] - 5.0) < 1e-6
        assert abs(M[1, 1] - 5.0) < 1e-6
        assert abs(M[2, 2] - 5.0) < 1e-6
        assert abs(M[0, 1]) < 1e-6

    def test_cif_to_adjacency_with_temp_file(self, tmp_path):
        """CIF file parsing end-to-end."""
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
        cif_file = tmp_path / "test.cif"
        cif_file.write_text(cif_content)

        adj, labels = cif_to_adjacency(str(cif_file), cutoff=3.0)
        assert len(labels) == 3
        assert "Si" in labels
        assert "O" in labels
        assert adj.shape == (3, 3)

    def test_cutoff_distance_affects_adjacency(self, tmp_path):
        """Smaller cutoff should have fewer edges."""
        cif_content = """data_test
_cell_length_a  10.0
_cell_length_b  10.0
_cell_length_c  10.0
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0

loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si 0.0 0.0 0.0
O  0.1 0.0 0.0
N  0.5 0.5 0.5
"""
        cif_file = tmp_path / "test2.cif"
        cif_file.write_text(cif_content)

        adj_small, _ = cif_to_adjacency(str(cif_file), cutoff=2.0)
        adj_large, _ = cif_to_adjacency(str(cif_file), cutoff=10.0)
        assert adj_large.nnz >= adj_small.nnz


# ── Visualization (smoke tests) ──────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    _HAS_MPL = True
except (ImportError, Exception):
    _HAS_MPL = False


@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib not available or broken")
class TestMaterialVisualization:

    def test_plot_cd_spectrum(self, detector):
        """Should generate matplotlib figure without error."""
        from zynerji_chirality.materials.visualization import plot_cd_spectrum

        result = detector.detect_material("N[C@@H](C)C(=O)O")
        ax = plot_cd_spectrum(result)
        assert ax is not None

    def test_plot_ciss_diagram(self, detector):
        """Should generate CISS diagram without error."""
        from zynerji_chirality.materials.visualization import plot_ciss_diagram

        result = detector.detect_material("N[C@@H](C)C(=O)O")
        ax = plot_ciss_diagram(result)
        assert ax is not None

    def test_plot_material_comparison(self, detector):
        """Should compare multiple materials without error."""
        from zynerji_chirality.materials.visualization import plot_material_comparison

        r1 = detector.detect_material("N[C@@H](C)C(=O)O")
        r2 = detector.detect_material("NCC(=O)O")
        ax = plot_material_comparison([r1, r2], ["L-Ala", "Gly"])
        assert ax is not None


# ── Properties unit tests ─────────────────────────────────────────────────

class TestProperties:

    def test_classify_optical_rotation_R(self):
        assert classify_optical_rotation(1.0, 0.1) == "dextrorotatory (+)"

    def test_classify_optical_rotation_S(self):
        assert classify_optical_rotation(-1.0, 0.1) == "levorotatory (-)"

    def test_classify_optical_rotation_achiral(self):
        assert classify_optical_rotation(0.0, 0.0) == "inactive"
