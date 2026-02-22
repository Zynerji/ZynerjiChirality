"""Tests for benchmark molecule databases.

Validates that all SMILES in the benchmark databases are parseable
and that the database structure is correct.
"""

import pytest

from rdkit import Chem

from zynerji_chirality.benchmarks.amino_acids import (
    get_all_pairs as get_amino_pairs,
    get_achiral as get_amino_achiral,
)
from zynerji_chirality.benchmarks.rs_pairs import get_all_pairs as get_drug_pairs
from zynerji_chirality.benchmarks.known_molecules import (
    get_achiral,
    get_meso,
    get_constitutional_isomers,
)


class TestAminoAcidDatabase:
    def test_has_19_pairs(self):
        pairs = get_amino_pairs()
        assert len(pairs) == 19

    def test_all_smiles_parseable(self):
        for name, l_smi, d_smi in get_amino_pairs():
            mol_l = Chem.MolFromSmiles(l_smi)
            mol_d = Chem.MolFromSmiles(d_smi)
            assert mol_l is not None, f"Cannot parse L-{name}: {l_smi}"
            assert mol_d is not None, f"Cannot parse D-{name}: {d_smi}"

    def test_l_and_d_same_formula(self):
        for name, l_smi, d_smi in get_amino_pairs():
            mol_l = Chem.MolFromSmiles(l_smi)
            mol_d = Chem.MolFromSmiles(d_smi)
            assert mol_l.GetNumAtoms() == mol_d.GetNumAtoms(), \
                f"{name}: L and D have different atom counts"

    def test_glycine_achiral(self):
        achiral = get_amino_achiral()
        assert len(achiral) == 1
        name, smiles = achiral[0]
        assert name == "Glycine"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None


class TestDrugPairDatabase:
    def test_has_pairs(self):
        pairs = get_drug_pairs()
        assert len(pairs) >= 12

    def test_all_smiles_parseable(self):
        for name, r_smi, s_smi, note in get_drug_pairs():
            mol_r = Chem.MolFromSmiles(r_smi)
            mol_s = Chem.MolFromSmiles(s_smi)
            assert mol_r is not None, f"Cannot parse R-{name}: {r_smi}"
            assert mol_s is not None, f"Cannot parse S-{name}: {s_smi}"

    def test_has_notes(self):
        for name, _, _, note in get_drug_pairs():
            assert len(note) > 0, f"{name} missing pharmacological note"


class TestKnownMolecules:
    def test_achiral_parseable(self):
        for name, smiles in get_achiral():
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"Cannot parse {name}: {smiles}"

    def test_meso_parseable(self):
        for name, smiles in get_meso():
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"Cannot parse {name}: {smiles}"

    def test_constitutional_isomers_parseable(self):
        for name, smi_a, smi_b in get_constitutional_isomers():
            mol_a = Chem.MolFromSmiles(smi_a)
            mol_b = Chem.MolFromSmiles(smi_b)
            assert mol_a is not None, f"Cannot parse {name} A: {smi_a}"
            assert mol_b is not None, f"Cannot parse {name} B: {smi_b}"

    def test_achiral_have_no_chiral_centers(self):
        for name, smiles in get_achiral():
            mol = Chem.MolFromSmiles(smiles)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            centers = Chem.FindMolChiralCenters(mol)
            assert len(centers) == 0, f"{name} should be achiral but has centers: {centers}"
