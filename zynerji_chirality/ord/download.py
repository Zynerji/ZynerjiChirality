"""ORD reaction data download and ee-selective filtering.

Accesses the Open Reaction Database (ORD) to extract enantioselective
reactions with enantiomeric excess (ee%) data.
"""

from __future__ import annotations

import glob
import gzip
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class ORDReaction:
    """An enantioselective reaction from the ORD."""

    reaction_id: str
    reaction_smiles: str
    catalyst_smiles: str | None = None
    substrate_smiles: str | None = None
    product_smiles: str | None = None
    ee_percent: float | None = None
    yield_percent: float | None = None
    temperature: str = ""
    solvent: str = ""
    source_doi: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ORDReaction:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ORDDownloader:
    """Download and parse ORD data for enantioselective reactions.

    Supports two modes:
    1. Full protobuf parsing via ord-schema (if installed)
    2. Pre-extracted JSON file loading (fallback)
    """

    def __init__(self):
        self._has_ord_schema = False
        try:
            from ord_schema import message_helpers
            from ord_schema.proto import reaction_pb2
            self._has_ord_schema = True
        except ImportError:
            logger.info("ord-schema not installed; use load_from_json() or install with: pip install ord-schema")

    def iterate_from_pb_dir(self, data_dir: str) -> Iterator[ORDReaction]:
        """Parse .pb.gz files from ORD data directory.

        Requires ord-schema. Filters for reactions with ee selectivity data.

        Parameters
        ----------
        data_dir : str
            Directory containing .pb.gz or .pb files from ORD.
        """
        if not self._has_ord_schema:
            raise ImportError(
                "ord-schema required for protobuf parsing. "
                "Install with: pip install ord-schema"
            )

        from ord_schema.proto import reaction_pb2

        pb_files = (
            glob.glob(str(Path(data_dir) / "**/*.pb.gz"), recursive=True)
            + glob.glob(str(Path(data_dir) / "**/*.pb"), recursive=True)
        )
        logger.info("Found %d protobuf files in %s", len(pb_files), data_dir)

        for pb_path in pb_files:
            try:
                dataset = reaction_pb2.Dataset()
                if pb_path.endswith(".gz"):
                    with gzip.open(pb_path, "rb") as f:
                        dataset.ParseFromString(f.read())
                else:
                    with open(pb_path, "rb") as f:
                        dataset.ParseFromString(f.read())

                for reaction in dataset.reactions:
                    ord_reaction = self._extract_ee_reaction(reaction)
                    if ord_reaction is not None:
                        yield ord_reaction
            except Exception as e:
                logger.debug("Error parsing %s: %s", pb_path, e)

    def _extract_ee_reaction(self, reaction) -> ORDReaction | None:
        """Extract ee data from an ORD reaction protobuf message.

        Returns None if reaction has no ee selectivity data.
        """
        from ord_schema.proto import reaction_pb2

        # Look for ee in outcomes
        ee_value = None
        yield_value = None
        product_smiles = None

        for outcome in reaction.outcomes:
            for product in outcome.products:
                # Check selectivity measurements
                for measurement in product.measurements:
                    if measurement.type == reaction_pb2.ProductMeasurement.SELECTIVITY:
                        if measurement.selectivity.type == reaction_pb2.Selectivity.EE:
                            ee_value = measurement.percentage.value if measurement.HasField("percentage") else None
                    elif measurement.type == reaction_pb2.ProductMeasurement.YIELD:
                        yield_value = measurement.percentage.value if measurement.HasField("percentage") else None

                # Get product SMILES
                for identifier in product.identifiers:
                    if identifier.type == reaction_pb2.CompoundIdentifier.SMILES:
                        product_smiles = identifier.value

        if ee_value is None:
            return None

        # Extract catalyst, substrate, reaction SMILES
        catalyst_smiles = None
        substrate_smiles = None
        reaction_smiles = ""

        for identifier in reaction.identifiers:
            if identifier.type == reaction_pb2.ReactionIdentifier.REACTION_SMILES:
                reaction_smiles = identifier.value

        # Inputs: look for catalysts and substrates
        for key, inp in reaction.inputs.items():
            for component in inp.components:
                smiles = None
                for identifier in component.identifiers:
                    if identifier.type == reaction_pb2.CompoundIdentifier.SMILES:
                        smiles = identifier.value
                        break
                if smiles is None:
                    continue

                role = component.reaction_role
                if role == reaction_pb2.ReactionRole.CATALYST:
                    catalyst_smiles = smiles
                elif role == reaction_pb2.ReactionRole.REACTANT:
                    if substrate_smiles is None:
                        substrate_smiles = smiles

        # Extract conditions
        temperature = ""
        solvent = ""
        if reaction.conditions.HasField("temperature"):
            temp = reaction.conditions.temperature
            if temp.HasField("setpoint"):
                temperature = f"{temp.setpoint.value} {temp.setpoint.units}"

        # Source DOI
        source_doi = ""
        if reaction.provenance and reaction.provenance.doi:
            source_doi = reaction.provenance.doi

        return ORDReaction(
            reaction_id=reaction.reaction_id,
            reaction_smiles=reaction_smiles,
            catalyst_smiles=catalyst_smiles,
            substrate_smiles=substrate_smiles,
            product_smiles=product_smiles,
            ee_percent=ee_value,
            yield_percent=yield_value,
            temperature=temperature,
            solvent=solvent,
            source_doi=source_doi,
        )

    def iterate_from_json(self, json_path: str) -> Iterator[ORDReaction]:
        """Load pre-extracted reactions from a JSON file.

        Parameters
        ----------
        json_path : str
            Path to JSON file with list of reaction dicts.
        """
        with open(json_path) as f:
            data = json.load(f)
        for entry in data:
            yield ORDReaction.from_dict(entry)

    @staticmethod
    def save_reactions(reactions: list[ORDReaction], path: str) -> None:
        """Save reactions to JSON checkpoint."""
        data = [r.to_dict() for r in reactions]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info("Saved %d reactions to %s", len(data), path)

    @staticmethod
    def load_reactions(path: str) -> list[ORDReaction]:
        """Load reactions from JSON checkpoint."""
        with open(path) as f:
            data = json.load(f)
        return [ORDReaction.from_dict(d) for d in data]
