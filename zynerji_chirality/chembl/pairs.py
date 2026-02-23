"""Enantiomer pair discovery from ChEMBL molecules.

Groups molecules by stereo-stripped canonical SMILES (same connectivity),
then verifies enantiomeric relationships by checking CIP label inversions.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Iterable

from rdkit import Chem

from zynerji_chirality.chembl.download import ChiralMolecule

logger = logging.getLogger(__name__)


@dataclass
class EnantiomerPair:
    """A pair of stereoisomers with same connectivity."""
    connectivity_smiles: str     # stereo-stripped canonical
    mol_a: ChiralMolecule       # first stereoisomer
    mol_b: ChiralMolecule       # second stereoisomer
    relationship: str            # "enantiomer" or "diastereomer"

    def to_dict(self) -> dict:
        return {
            "connectivity_smiles": self.connectivity_smiles,
            "mol_a": asdict(self.mol_a),
            "mol_b": asdict(self.mol_b),
            "relationship": self.relationship,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EnantiomerPair:
        return cls(
            connectivity_smiles=d["connectivity_smiles"],
            mol_a=ChiralMolecule(**d["mol_a"]),
            mol_b=ChiralMolecule(**d["mol_b"]),
            relationship=d["relationship"],
        )


def _classify_relationship(mol_a: ChiralMolecule, mol_b: ChiralMolecule) -> str | None:
    """Classify the stereoisomeric relationship between two molecules.

    Returns "enantiomer" if ALL chiral centers are inverted,
    "diastereomer" if SOME centers are inverted, or None if no inversion.
    """
    # Both must have the same number of centers
    if mol_a.n_centers != mol_b.n_centers:
        return None

    # Parse with RDKit to get atom-mapped CIP labels
    mol_a_rdkit = Chem.MolFromSmiles(mol_a.canonical_smiles)
    mol_b_rdkit = Chem.MolFromSmiles(mol_b.canonical_smiles)

    if mol_a_rdkit is None or mol_b_rdkit is None:
        return None

    Chem.AssignStereochemistry(mol_a_rdkit, cleanIt=True, force=True)
    Chem.AssignStereochemistry(mol_b_rdkit, cleanIt=True, force=True)

    centers_a = Chem.FindMolChiralCenters(mol_a_rdkit, includeUnassigned=False)
    centers_b = Chem.FindMolChiralCenters(mol_b_rdkit, includeUnassigned=False)

    if not centers_a or not centers_b:
        return None

    # Use canonical atom ordering to map centers between molecules
    # Since they have the same stereo-stripped SMILES, canonical ranks should match
    ranks_a = list(Chem.CanonicalRankAtoms(mol_a_rdkit))
    ranks_b = list(Chem.CanonicalRankAtoms(mol_b_rdkit))

    # Build rank → CIP label maps
    rank_to_cip_a = {}
    for atom_idx, label in centers_a:
        rank_to_cip_a[ranks_a[atom_idx]] = label

    rank_to_cip_b = {}
    for atom_idx, label in centers_b:
        rank_to_cip_b[ranks_b[atom_idx]] = label

    # Compare CIP labels at matching ranks
    common_ranks = set(rank_to_cip_a.keys()) & set(rank_to_cip_b.keys())
    if not common_ranks:
        return None

    n_inverted = 0
    n_same = 0
    for rank in common_ranks:
        label_a = rank_to_cip_a[rank]
        label_b = rank_to_cip_b[rank]
        if label_a != label_b:
            n_inverted += 1
        else:
            n_same += 1

    if n_inverted == 0:
        return None  # Same stereoisomer
    elif n_same == 0:
        return "enantiomer"  # ALL centers inverted
    else:
        return "diastereomer"  # Some inverted, some same


class EnantiomerPairFinder:
    """Discover enantiomer and diastereomer pairs from a set of chiral molecules."""

    def find_pairs(
        self,
        molecules: Iterable[ChiralMolecule],
        include_diastereomers: bool = True,
    ) -> list[EnantiomerPair]:
        """Group by stereo-stripped SMILES, verify opposite CIP labels.

        Parameters
        ----------
        molecules : Iterable[ChiralMolecule]
            Chiral molecules to search for pairs.
        include_diastereomers : bool
            If True, also include diastereomer pairs.

        Returns
        -------
        list[EnantiomerPair]
            Discovered stereoisomer pairs.
        """
        # Group by stereo-stripped SMILES
        groups: dict[str, list[ChiralMolecule]] = defaultdict(list)
        for mol in molecules:
            groups[mol.stereo_stripped_smiles].append(mol)

        logger.info(
            "Grouped into %d connectivity classes (%d with 2+ members)",
            len(groups),
            sum(1 for g in groups.values() if len(g) >= 2),
        )

        pairs = []
        for connectivity, members in groups.items():
            if len(members) < 2:
                continue

            # Deduplicate by canonical SMILES
            seen_smiles: set[str] = set()
            unique_members: list[ChiralMolecule] = []
            for m in members:
                if m.canonical_smiles not in seen_smiles:
                    seen_smiles.add(m.canonical_smiles)
                    unique_members.append(m)

            if len(unique_members) < 2:
                continue

            # Check all pairs within the group
            for i in range(len(unique_members)):
                for j in range(i + 1, len(unique_members)):
                    rel = _classify_relationship(unique_members[i], unique_members[j])
                    if rel is None:
                        continue
                    if rel == "diastereomer" and not include_diastereomers:
                        continue

                    pairs.append(EnantiomerPair(
                        connectivity_smiles=connectivity,
                        mol_a=unique_members[i],
                        mol_b=unique_members[j],
                        relationship=rel,
                    ))

        n_enant = sum(1 for p in pairs if p.relationship == "enantiomer")
        n_diast = sum(1 for p in pairs if p.relationship == "diastereomer")
        logger.info(
            "Found %d pairs: %d enantiomer, %d diastereomer",
            len(pairs), n_enant, n_diast,
        )
        return pairs

    def save_pairs(self, pairs: list[EnantiomerPair], path: str) -> None:
        """Save discovered pairs to JSON for checkpointing."""
        data = [p.to_dict() for p in pairs]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d pairs to %s", len(pairs), path)

    def load_pairs(self, path: str) -> list[EnantiomerPair]:
        """Load previously discovered pairs."""
        with open(path) as f:
            data = json.load(f)
        pairs = [EnantiomerPair.from_dict(d) for d in data]
        logger.info("Loaded %d pairs from %s", len(pairs), path)
        return pairs
