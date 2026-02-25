"""Reaction data extraction and classification for ORD ee reactions.

Filters, validates, and enriches ORD reactions with chirality analysis
of catalysts and substrates.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, asdict

from zynerji_chirality.ord.download import ORDReaction

logger = logging.getLogger(__name__)


@dataclass
class EnantioselectiveReaction:
    """An ORD reaction enriched with chirality analysis."""

    # Base reaction fields
    reaction_id: str
    reaction_smiles: str
    catalyst_smiles: str | None
    substrate_smiles: str | None
    product_smiles: str | None
    ee_percent: float
    yield_percent: float | None
    temperature: str
    solvent: str
    source_doi: str

    # Chirality enrichment
    catalyst_chirality_score: float = 0.0
    catalyst_chirality_sign: float = 0.0
    catalyst_is_chiral: bool = False
    substrate_chirality_score: float = 0.0
    substrate_chirality_sign: float = 0.0
    substrate_is_chiral: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> EnantioselectiveReaction:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ORDExtractor:
    """Extract and classify enantioselective reactions from ORD data.

    Filters valid ee data, computes chirality of catalysts/substrates,
    and groups reactions for analysis.
    """

    def __init__(self):
        self._detector = None

    @property
    def detector(self):
        if self._detector is None:
            from zynerji_chirality.chirality.detector import HelixChiralityDetector
            self._detector = HelixChiralityDetector()
        return self._detector

    def extract_ee_reactions(
        self, reactions: list[ORDReaction]
    ) -> list[EnantioselectiveReaction]:
        """Filter reactions with valid ee data and enrich with chirality.

        Parameters
        ----------
        reactions : list[ORDReaction]
            Raw ORD reactions (may include ones without ee).

        Returns
        -------
        list[EnantioselectiveReaction]
            Reactions with valid ee data, enriched with chirality analysis.
        """
        enriched = []
        for rxn in reactions:
            if rxn.ee_percent is None:
                continue
            if not (0 <= rxn.ee_percent <= 100):
                continue

            cat_score, cat_sign, cat_chiral = 0.0, 0.0, False
            sub_score, sub_sign, sub_chiral = 0.0, 0.0, False

            if rxn.catalyst_smiles:
                try:
                    result = self.detector.detect(rxn.catalyst_smiles)
                    cat_score = result.chirality_score
                    cat_sign = result.chirality_sign
                    cat_chiral = result.is_chiral
                except Exception:
                    pass

            if rxn.substrate_smiles:
                try:
                    result = self.detector.detect(rxn.substrate_smiles)
                    sub_score = result.chirality_score
                    sub_sign = result.chirality_sign
                    sub_chiral = result.is_chiral
                except Exception:
                    pass

            enriched.append(EnantioselectiveReaction(
                reaction_id=rxn.reaction_id,
                reaction_smiles=rxn.reaction_smiles,
                catalyst_smiles=rxn.catalyst_smiles,
                substrate_smiles=rxn.substrate_smiles,
                product_smiles=rxn.product_smiles,
                ee_percent=rxn.ee_percent,
                yield_percent=rxn.yield_percent,
                temperature=rxn.temperature,
                solvent=rxn.solvent,
                source_doi=rxn.source_doi,
                catalyst_chirality_score=cat_score,
                catalyst_chirality_sign=cat_sign,
                catalyst_is_chiral=cat_chiral,
                substrate_chirality_score=sub_score,
                substrate_chirality_sign=sub_sign,
                substrate_is_chiral=sub_chiral,
            ))

        logger.info(
            "Extracted %d ee reactions from %d total", len(enriched), len(reactions)
        )
        return enriched

    def extract_catalyst_substrate_pairs(
        self, reactions: list[EnantioselectiveReaction]
    ) -> list[dict]:
        """Format reactions for EEPredictor.fit().

        Returns
        -------
        list[dict]
            Dicts with keys: catalyst, substrate, ee, major.
        """
        pairs = []
        for rxn in reactions:
            if not rxn.catalyst_smiles or not rxn.substrate_smiles:
                continue
            major = "R" if rxn.ee_percent >= 0 else "S"
            pairs.append({
                "catalyst": rxn.catalyst_smiles,
                "substrate": rxn.substrate_smiles,
                "ee": abs(rxn.ee_percent),
                "major": major,
            })
        return pairs

    @staticmethod
    def group_by_catalyst(
        reactions: list[EnantioselectiveReaction],
    ) -> dict[str, list[EnantioselectiveReaction]]:
        """Group reactions by catalyst SMILES."""
        groups: dict[str, list[EnantioselectiveReaction]] = defaultdict(list)
        for rxn in reactions:
            key = rxn.catalyst_smiles or "unknown"
            groups[key].append(rxn)
        return dict(groups)

    @staticmethod
    def group_by_reaction_type(
        reactions: list[EnantioselectiveReaction],
    ) -> dict[str, list[EnantioselectiveReaction]]:
        """Classify and group reactions by type (heuristic).

        Uses reaction SMILES patterns to classify:
        - hydrogenation (H2 or [H][H])
        - oxidation (common oxidants)
        - C-C coupling
        - other
        """
        groups: dict[str, list[EnantioselectiveReaction]] = defaultdict(list)
        for rxn in reactions:
            rs = rxn.reaction_smiles.lower() if rxn.reaction_smiles else ""
            if "[h][h]" in rs or "[h]" in rs:
                rtype = "hydrogenation"
            elif "o=o" in rs or "[o]" in rs:
                rtype = "oxidation"
            elif ">>" in rs:
                rtype = "transformation"
            else:
                rtype = "other"
            groups[rtype].append(rxn)
        return dict(groups)
