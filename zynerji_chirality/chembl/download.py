"""ChEMBL molecule download and chiral filtering.

Uses chembl-downloader to iterate over the full ChEMBL database,
filtering to molecules with defined stereocenters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

from rdkit import Chem
from rdkit.Chem import rdmolops

logger = logging.getLogger(__name__)


@dataclass
class ChiralMolecule:
    """A molecule with defined stereochemistry from ChEMBL."""
    chembl_id: str
    smiles: str
    canonical_smiles: str
    stereo_stripped_smiles: str
    n_centers: int
    cip_labels: list[tuple[int, str]] = field(default_factory=list)

    def __hash__(self):
        return hash(self.chembl_id)

    def __eq__(self, other):
        if not isinstance(other, ChiralMolecule):
            return NotImplemented
        return self.chembl_id == other.chembl_id


def strip_stereo(mol: Chem.Mol) -> str:
    """Remove all stereochemistry and return canonical SMILES."""
    mol_copy = Chem.RWMol(mol)
    Chem.RemoveStereochemistry(mol_copy)
    return Chem.MolToSmiles(mol_copy.GetMol())


class ChEMBLDownloader:
    """Download and iterate ChEMBL molecules with stereochemistry.

    Parameters
    ----------
    version : str
        ChEMBL version to download. Use "latest" for most recent.
    cache_dir : str, optional
        Directory to cache downloaded files. Uses chembl-downloader default if None.
    """

    def __init__(self, version: str = "latest", cache_dir: str | None = None):
        self.version = version
        self.cache_dir = cache_dir

    def iterate_chiral_molecules(self) -> Iterator[ChiralMolecule]:
        """Yield only molecules with defined stereocenters.

        Uses chembl-downloader's supplier() to iterate RDKit Mol objects,
        then filters to those with assigned CIP stereocenters.
        """
        try:
            import chembl_downloader
        except ImportError:
            raise ImportError(
                "chembl-downloader required. Install with: "
                "pip install zynerji-chirality[chembl]"
            )

        logger.info("Starting ChEMBL molecule iteration (version=%s)", self.version)
        n_total = 0
        n_chiral = 0

        kwargs = {}
        if self.version != "latest":
            kwargs["version"] = self.version

        with chembl_downloader.supplier(**kwargs) as suppl:
            for mol in suppl:
                if mol is None:
                    continue
                n_total += 1

                try:
                    # Assign stereochemistry
                    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
                    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)

                    if not centers:
                        continue

                    smiles = Chem.MolToSmiles(mol)
                    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                    stereo_stripped = strip_stereo(mol)

                    # Get ChEMBL ID from molecule properties
                    chembl_id = ""
                    if mol.HasProp("chembl_id"):
                        chembl_id = mol.GetProp("chembl_id")
                    elif mol.HasProp("_Name"):
                        chembl_id = mol.GetProp("_Name")

                    n_chiral += 1
                    yield ChiralMolecule(
                        chembl_id=chembl_id,
                        smiles=smiles,
                        canonical_smiles=canonical,
                        stereo_stripped_smiles=stereo_stripped,
                        n_centers=len(centers),
                        cip_labels=list(centers),
                    )

                except Exception as e:
                    logger.debug("Failed to process molecule %d: %s", n_total, e)
                    continue

                if n_total % 100000 == 0:
                    logger.info(
                        "Processed %d molecules, %d chiral (%.1f%%)",
                        n_total, n_chiral, 100.0 * n_chiral / n_total,
                    )

        logger.info(
            "Done: %d total, %d chiral (%.1f%%)",
            n_total, n_chiral, 100.0 * n_chiral / max(n_total, 1),
        )

    def iterate_from_smiles_file(self, path: str) -> Iterator[ChiralMolecule]:
        """Iterate chiral molecules from a SMILES file (one per line).

        Format: SMILES<tab>CHEMBL_ID or just SMILES per line.
        Useful for pre-downloaded or filtered datasets.
        """
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")
                smiles = parts[0].strip()
                chembl_id = parts[1].strip() if len(parts) > 1 else f"MOL_{i}"

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                try:
                    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
                    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)

                    if not centers:
                        continue

                    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                    stereo_stripped = strip_stereo(mol)

                    yield ChiralMolecule(
                        chembl_id=chembl_id,
                        smiles=smiles,
                        canonical_smiles=canonical,
                        stereo_stripped_smiles=stereo_stripped,
                        n_centers=len(centers),
                        cip_labels=list(centers),
                    )
                except Exception:
                    continue

    def count_total(self) -> int:
        """Quick count of total molecules in release.

        Note: requires downloading the full dataset.
        """
        try:
            import chembl_downloader
        except ImportError:
            raise ImportError(
                "chembl-downloader required. Install with: "
                "pip install zynerji-chirality[chembl]"
            )

        count = 0
        kwargs = {}
        if self.version != "latest":
            kwargs["version"] = self.version

        with chembl_downloader.supplier(**kwargs) as suppl:
            for mol in suppl:
                if mol is not None:
                    count += 1
        return count
