"""Retrosynthetic chirality planning.

Suggests synthetic strategies for introducing stereocenters based on
local chemical environment analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from zynerji_chirality.chirality.detector import HelixChiralityDetector


@dataclass
class RetroSuggestion:
    """A suggested synthetic strategy for a stereocenter."""
    atom_idx: int
    cip_label: str
    environment: str          # Description of local environment
    strategies: list[str]     # Applicable synthetic strategies
    confidence: float         # How well the strategies match (0-1)


# Strategy lookup: maps local environment patterns to synthetic approaches
_STRATEGY_TABLE = {
    "alcohol": [
        "Asymmetric hydrogenation of ketone precursor",
        "CBS reduction (oxazaborolidine catalyst)",
        "Enzymatic resolution (lipase, hydrolase)",
    ],
    "amine": [
        "Asymmetric reductive amination",
        "Chiral auxiliary (Evans oxazolidinone)",
        "Enzymatic transamination",
    ],
    "alpha_amino_acid": [
        "Strecker synthesis with chiral catalyst",
        "Enzymatic resolution (acylase)",
        "Asymmetric hydrogenation of dehydroamino acid",
    ],
    "epoxide": [
        "Sharpless asymmetric epoxidation",
        "Jacobsen hydrolytic kinetic resolution",
    ],
    "allyl_alcohol": [
        "Sharpless asymmetric epoxidation",
        "Asymmetric dihydroxylation (AD-mix)",
    ],
    "olefin": [
        "Asymmetric hydrogenation (Rh-BINAP, Ir-PHOX)",
        "Asymmetric dihydroxylation",
        "Asymmetric cyclopropanation",
    ],
    "carbanion_alkylation": [
        "Chiral auxiliary (Evans, Myers, Ellman)",
        "Asymmetric phase-transfer catalysis",
        "Chiral enolate alkylation",
    ],
    "generic": [
        "Enzymatic resolution (kinetic resolution)",
        "Chiral HPLC separation",
        "Diastereomeric salt resolution",
    ],
}


class RetroChiralityPlanner:
    """Suggest stereoselective synthetic strategies for target molecules.

    Analyzes each stereocenter's local chemical environment and
    recommends compatible asymmetric synthesis approaches.
    """

    def __init__(self, detector: HelixChiralityDetector | None = None):
        self.detector = detector or HelixChiralityDetector()

    def plan(self, smiles: str) -> list[RetroSuggestion]:
        """Suggest synthetic strategies for each stereocenter.

        Parameters
        ----------
        smiles : str
            Target molecule SMILES.

        Returns
        -------
        list[RetroSuggestion]
            One suggestion per stereocenter.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: {smiles}")

        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        if not centers:
            return []

        suggestions = []
        for atom_idx, label in centers:
            env = self._classify_environment(mol, atom_idx)
            strategies = _STRATEGY_TABLE.get(env, _STRATEGY_TABLE["generic"])
            confidence = 0.8 if env != "generic" else 0.3

            suggestions.append(RetroSuggestion(
                atom_idx=atom_idx,
                cip_label=label,
                environment=env,
                strategies=list(strategies),
                confidence=confidence,
            ))

        return suggestions

    def _classify_environment(self, mol: Chem.Mol, atom_idx: int) -> str:
        """Classify the local chemical environment of a stereocenter."""
        atom = mol.GetAtomWithIdx(atom_idx)
        neighbors = [nb for nb in atom.GetNeighbors()]
        neighbor_symbols = set(nb.GetSymbol() for nb in neighbors)
        neighbor_nums = set(nb.GetAtomicNum() for nb in neighbors)

        # Check for common functional group patterns
        has_oh = False
        has_nh = False
        has_cooh = False
        has_double_bond_neighbor = False

        for nb in neighbors:
            nb_sym = nb.GetSymbol()
            nb_idx = nb.GetIdx()

            if nb_sym == "O":
                # Check if it's -OH (alcohol) or C=O or COOH
                if nb.GetDegree() == 1 or (nb.GetDegree() == 2 and nb.GetTotalNumHs() > 0):
                    has_oh = True
            elif nb_sym == "N":
                has_nh = True

            # Check for adjacent C=O (alpha to carbonyl)
            for nb2 in nb.GetNeighbors():
                if nb2.GetIdx() == atom_idx:
                    continue
                bond = mol.GetBondBetweenAtoms(nb.GetIdx(), nb2.GetIdx())
                if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                    if nb2.GetSymbol() == "O":
                        has_cooh = True
                    has_double_bond_neighbor = True

        # Classify
        if has_nh and has_cooh:
            return "alpha_amino_acid"
        if has_nh:
            return "amine"
        if has_oh:
            return "alcohol"

        # Check for epoxide (3-membered ring with O)
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if atom_idx in ring and len(ring) == 3:
                ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
                if any(a.GetSymbol() == "O" for a in ring_atoms):
                    return "epoxide"

        if has_double_bond_neighbor:
            return "carbanion_alkylation"

        return "generic"
