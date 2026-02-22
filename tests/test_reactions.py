"""Tests for reaction stereochemistry analysis."""

import pytest

from zynerji_chirality.reactions.reaction_graph import (
    ReactionStereoAnalyzer,
    ReactionStereoResult,
    CenterFate,
)


@pytest.fixture
def analyzer():
    return ReactionStereoAnalyzer()


class TestReactionStereoAnalyzer:
    def test_simple_reaction(self, analyzer):
        """Simple reaction with retained stereocenter."""
        # L-Alanine to N-acetyl-L-alanine (stereocenter retained)
        rxn = "[NH2:1][C@@H:2]([CH3:3])[C:4](=O)[OH:5]>>[CH3:6][C:7](=O)[NH:1][C@@H:2]([CH3:3])[C:4](=O)[OH:5]"
        result = analyzer.analyze(rxn)
        assert isinstance(result, ReactionStereoResult)

    def test_no_stereocenters(self, analyzer):
        """Reaction without stereocenters."""
        rxn = "[CH3:1][OH:2]>>[CH3:1][O:2][CH3:3]"
        result = analyzer.analyze(rxn)
        assert len(result.new_centers) == 0
        assert len(result.lost_centers) == 0

    def test_new_stereocenter(self, analyzer):
        """Reaction creating a new stereocenter."""
        # Ketone to alcohol — creates new center
        rxn = "[CH3:1][C:2](=O)[CH3:3]>>[CH3:1][C@@H:2]([OH:4])[CH3:3]"
        result = analyzer.analyze(rxn)
        assert 2 in result.new_centers

    def test_lost_stereocenter(self, analyzer):
        """Reaction destroying a stereocenter."""
        rxn = "[C@@H:1]([OH:2])([CH3:3])[CH2:4][CH3:5]>>[C:1](=O)([CH3:3])[CH2:4][CH3:5]"
        result = analyzer.analyze(rxn)
        assert 1 in result.lost_centers

    def test_repr(self, analyzer):
        """Repr should show center counts."""
        rxn = "[CH3:1][OH:2]>>[CH3:1][O:2][CH3:3]"
        result = analyzer.analyze(rxn)
        s = repr(result)
        assert "new=" in s
        assert "inverted=" in s

    def test_invalid_reaction(self, analyzer):
        """Invalid reaction should raise."""
        with pytest.raises(ValueError):
            analyzer.analyze("not a reaction")

    def test_center_fate_dataclass(self):
        """CenterFate should store fate information."""
        fate = CenterFate(
            atom_map_num=1,
            reactant_label="R",
            product_label="S",
            fate="inverted",
        )
        assert fate.fate == "inverted"
        assert fate.atom_map_num == 1
