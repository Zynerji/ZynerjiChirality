"""Reaction stereochemistry analysis via atom-mapping.

Tracks the fate of stereocenters through chemical reactions by comparing
chirality analysis of reactants and products linked by atom-mapping numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import AllChem

from zynerji_chirality.chirality.detector import (
    HelixChiralityDetector,
    ChiralityResult,
)


@dataclass
class CenterFate:
    """Describes what happened to a stereocenter through a reaction."""
    atom_map_num: int
    reactant_label: str     # "R", "S", "?", or "none"
    product_label: str      # "R", "S", "?", or "none"
    fate: str               # "retained", "inverted", "new", "lost", "unchanged"


@dataclass
class ReactionStereoResult:
    """Result of stereochemical analysis of a reaction."""
    reaction_smarts: str
    reactant_results: list[ChiralityResult]
    product_results: list[ChiralityResult]
    center_fates: list[CenterFate]
    new_centers: list[int]        # atom map nums of new stereocenters
    inverted_centers: list[int]   # atom map nums of inverted centers
    retained_centers: list[int]   # atom map nums of retained centers
    lost_centers: list[int]       # atom map nums of lost centers

    def __repr__(self) -> str:
        return (
            f"ReactionStereoResult(new={len(self.new_centers)}, "
            f"inverted={len(self.inverted_centers)}, "
            f"retained={len(self.retained_centers)}, "
            f"lost={len(self.lost_centers)})"
        )


class ReactionStereoAnalyzer:
    """Analyze stereochemical changes in chemical reactions.

    Parses reaction SMARTS, runs chirality detection on each reactant
    and product, then tracks stereocenter fate via atom-mapping numbers.
    """

    def __init__(self, detector: HelixChiralityDetector | None = None):
        self.detector = detector or HelixChiralityDetector()

    def analyze(self, reaction_smarts: str) -> ReactionStereoResult:
        """Analyze stereochemistry of a reaction.

        Parameters
        ----------
        reaction_smarts : str
            Reaction SMARTS/SMILES with atom mapping (e.g., "[C:1]>>[C:1]").

        Returns
        -------
        ReactionStereoResult
            Complete stereochemical analysis.
        """
        rxn = AllChem.ReactionFromSmarts(reaction_smarts)
        if rxn is None:
            raise ValueError(f"Cannot parse reaction: {reaction_smarts}")

        # Extract reactant and product molecules
        reactant_mols = []
        product_mols = []

        # Parse reactants and products from SMARTS string
        parts = reaction_smarts.split(">>")
        if len(parts) != 2:
            raise ValueError("Reaction must have exactly one '>>' separator")

        reactant_smiles_list = parts[0].split(".")
        product_smiles_list = parts[1].split(".")

        reactant_results = []
        product_results = []

        # Build atom map -> CIP label maps
        reactant_centers = {}  # atom_map_num -> CIP label
        product_centers = {}

        for smi in reactant_smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                result = self.detector.detect(smi)
                reactant_results.append(result)
            except Exception:
                continue

            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            for atom_idx, label in centers:
                atom = mol.GetAtomWithIdx(atom_idx)
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    reactant_centers[map_num] = label

        for smi in product_smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                result = self.detector.detect(smi)
                product_results.append(result)
            except Exception:
                continue

            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            for atom_idx, label in centers:
                atom = mol.GetAtomWithIdx(atom_idx)
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    product_centers[map_num] = label

        # Determine center fates
        all_map_nums = set(reactant_centers.keys()) | set(product_centers.keys())
        center_fates = []
        new_centers = []
        inverted_centers = []
        retained_centers = []
        lost_centers = []

        for map_num in sorted(all_map_nums):
            r_label = reactant_centers.get(map_num, "none")
            p_label = product_centers.get(map_num, "none")

            if r_label == "none" and p_label != "none":
                fate = "new"
                new_centers.append(map_num)
            elif r_label != "none" and p_label == "none":
                fate = "lost"
                lost_centers.append(map_num)
            elif r_label == p_label:
                fate = "retained"
                retained_centers.append(map_num)
            elif (r_label in ("R", "S") and p_label in ("R", "S")
                  and r_label != p_label):
                fate = "inverted"
                inverted_centers.append(map_num)
            else:
                fate = "unchanged"

            center_fates.append(CenterFate(
                atom_map_num=map_num,
                reactant_label=r_label,
                product_label=p_label,
                fate=fate,
            ))

        return ReactionStereoResult(
            reaction_smarts=reaction_smarts,
            reactant_results=reactant_results,
            product_results=product_results,
            center_fates=center_fates,
            new_centers=new_centers,
            inverted_centers=inverted_centers,
            retained_centers=retained_centers,
            lost_centers=lost_centers,
        )
