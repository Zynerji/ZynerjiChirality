"""Stereoisomer enumeration using RDKit.

Enumerates all possible stereoisomers of a molecule, including
single-center flips for targeted optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

from zynerji_chirality.chirality.fingerprint import chirality_fingerprint

logger = logging.getLogger(__name__)


@dataclass
class StereoisomerVariant:
    """A single stereoisomer variant."""
    smiles: str
    canonical_smiles: str
    n_centers: int
    configuration: dict[int, str] = field(default_factory=dict)  # atom_idx -> R/S
    is_original: bool = False
    fingerprint: np.ndarray | None = None


class StereoisomerEnumerator:
    """Enumerate stereoisomers of a molecule.

    Parameters
    ----------
    max_centers : int
        Maximum stereocenters to enumerate (2^max_centers limit).
    compute_fingerprints : bool
        Whether to compute chirality fingerprints for each variant.
    nbits : int
        Fingerprint bits (if computing).
    """

    def __init__(
        self,
        max_centers: int = 8,
        compute_fingerprints: bool = True,
        nbits: int = 128,
    ):
        self.max_centers = max_centers
        self.compute_fingerprints = compute_fingerprints
        self.nbits = nbits

    def enumerate(self, smiles: str) -> list[StereoisomerVariant]:
        """Enumerate all stereoisomers of a molecule.

        Parameters
        ----------
        smiles : str
            Input SMILES.

        Returns
        -------
        list[StereoisomerVariant]
            All unique stereoisomers (up to 2^max_centers).
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Count stereocenters
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        n_centers = len(centers)

        if n_centers == 0:
            # Achiral molecule — return single variant
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            fp = None
            if self.compute_fingerprints:
                try:
                    fp = chirality_fingerprint(canonical, nbits=self.nbits)
                except Exception:
                    pass
            return [StereoisomerVariant(
                smiles=canonical,
                canonical_smiles=canonical,
                n_centers=0,
                configuration={},
                is_original=True,
                fingerprint=fp,
            )]

        if n_centers > self.max_centers:
            logger.warning(
                "Molecule has %d centers (max %d). Truncating enumeration.",
                n_centers, self.max_centers,
            )

        # Remove existing stereo assignments so EnumerateStereoisomers
        # will enumerate ALL possible configurations, not just unassigned ones
        Chem.RemoveStereochemistry(mol)

        # Enumerate stereoisomers
        opts = StereoEnumerationOptions(
            tryEmbedding=False,
            unique=True,
            maxIsomers=2 ** min(n_centers, self.max_centers),
        )
        isomers = list(EnumerateStereoisomers(mol, options=opts))

        # Get canonical SMILES of original for comparison
        original_canonical = Chem.MolToSmiles(mol, isomericSmiles=True)

        variants = []
        seen_smiles: set[str] = set()

        for iso_mol in isomers:
            iso_smiles = Chem.MolToSmiles(iso_mol, isomericSmiles=True)
            if iso_smiles in seen_smiles:
                continue
            seen_smiles.add(iso_smiles)

            # Get CIP configurations
            Chem.AssignStereochemistry(iso_mol, cleanIt=True, force=True)
            iso_centers = Chem.FindMolChiralCenters(iso_mol, includeUnassigned=False)
            config = {idx: label for idx, label in iso_centers}

            fp = None
            if self.compute_fingerprints:
                try:
                    fp = chirality_fingerprint(iso_smiles, nbits=self.nbits)
                except Exception:
                    pass

            variants.append(StereoisomerVariant(
                smiles=iso_smiles,
                canonical_smiles=iso_smiles,
                n_centers=len(iso_centers),
                configuration=config,
                is_original=(iso_smiles == original_canonical),
                fingerprint=fp,
            ))

        logger.debug(
            "Enumerated %d stereoisomers for %s (%d centers)",
            len(variants), smiles, n_centers,
        )
        return variants

    def enumerate_single_flips(self, smiles: str) -> list[StereoisomerVariant]:
        """Enumerate variants with exactly one center flipped (Hamming distance 1).

        Produces N variants for N stereocenters, each with exactly one
        center inverted relative to the input.

        Parameters
        ----------
        smiles : str
            Input SMILES with defined stereochemistry.

        Returns
        -------
        list[StereoisomerVariant]
            Variants that differ at exactly one stereocenter.
        """
        all_variants = self.enumerate(smiles)

        if len(all_variants) <= 1:
            return []

        # Find the original
        original = None
        for v in all_variants:
            if v.is_original:
                original = v
                break
        if original is None:
            original = all_variants[0]

        # Filter to Hamming distance 1
        single_flips = []
        for v in all_variants:
            if v.canonical_smiles == original.canonical_smiles:
                continue

            # Count configuration differences
            n_diff = 0
            all_keys = set(original.configuration.keys()) | set(v.configuration.keys())
            for key in all_keys:
                if original.configuration.get(key) != v.configuration.get(key):
                    n_diff += 1

            if n_diff == 1:
                single_flips.append(v)

        return single_flips
