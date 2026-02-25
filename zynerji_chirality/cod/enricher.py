"""CIF enrichment and chirality analysis for COD structures.

Downloads CIF files, converts to adjacency matrices, and runs chirality
detection via the materials module.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from zynerji_chirality.cod.download import CODDownloader, CrystalStructure
from zynerji_chirality.materials.crystal import cif_to_adjacency, lattice_to_adjacency
from zynerji_chirality.materials.detector import (
    ChiralMaterialDetector,
    MaterialChiralityResult,
)

logger = logging.getLogger(__name__)


@dataclass
class CrystalEnrichment:
    """Chirality-enriched crystal structure."""

    structure: CrystalStructure
    atom_labels: list[str]
    n_atoms: int
    cd_activity: float
    cd_sign: str
    ciss_score: float
    optical_rotation: str
    chirality_score: float
    chirality_sign: float
    is_chiral: bool
    band_gap_shift: float
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "structure": self.structure.to_dict(),
            "atom_labels": self.atom_labels,
            "n_atoms": self.n_atoms,
            "cd_activity": self.cd_activity,
            "cd_sign": self.cd_sign,
            "ciss_score": self.ciss_score,
            "optical_rotation": self.optical_rotation,
            "chirality_score": self.chirality_score,
            "chirality_sign": self.chirality_sign,
            "is_chiral": self.is_chiral,
            "band_gap_shift": self.band_gap_shift,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CrystalEnrichment:
        return cls(
            structure=CrystalStructure.from_dict(d["structure"]),
            atom_labels=d["atom_labels"],
            n_atoms=d["n_atoms"],
            cd_activity=d["cd_activity"],
            cd_sign=d["cd_sign"],
            ciss_score=d["ciss_score"],
            optical_rotation=d["optical_rotation"],
            chirality_score=d["chirality_score"],
            chirality_sign=d["chirality_sign"],
            is_chiral=d["is_chiral"],
            band_gap_shift=d["band_gap_shift"],
            error=d.get("error"),
        )


class CODEnricher:
    """Enrich COD structures with chirality analysis.

    Parameters
    ----------
    cutoff : float
        Distance cutoff (Angstroms) for adjacency matrix construction.
    """

    def __init__(self, cutoff: float = 3.0, skip_cif: bool = False):
        self.cutoff = cutoff
        self.skip_cif = skip_cif
        self.detector = ChiralMaterialDetector()
        self.downloader = CODDownloader()

    def enrich_structure(
        self, structure: CrystalStructure, cif_dir: str
    ) -> CrystalEnrichment:
        """Enrich a single structure by downloading CIF and analyzing chirality.

        Falls back to lattice-parameter-based graph if CIF download fails.
        """
        try:
            # Try CIF-based enrichment first (unless skip_cif)
            adj, labels = None, None
            if not self.skip_cif:
                cif_path = self.downloader.download_cif(structure.cod_id, cif_dir)
                if cif_path:
                    adj, labels = cif_to_adjacency(cif_path, cutoff=self.cutoff)
                    if adj.shape[0] == 0:
                        adj, labels = None, None

            # Fallback: build graph from lattice parameters + formula
            if adj is None or labels is None:
                adj, labels = lattice_to_adjacency(
                    structure.formula,
                    structure.a, structure.b, structure.c,
                    structure.alpha, structure.beta, structure.gamma,
                )
                if adj.shape[0] == 0:
                    return self._error_enrichment(structure, "Empty adjacency matrix")

            result = self.detector.detect_from_adjacency(
                adj.toarray() if hasattr(adj, "toarray") else adj,
                atom_labels=labels,
            )

            return CrystalEnrichment(
                structure=structure,
                atom_labels=labels,
                n_atoms=len(labels),
                cd_activity=result.cd_activity,
                cd_sign=result.cd_sign,
                ciss_score=result.ciss_score,
                optical_rotation=result.optical_rotation,
                chirality_score=result.chirality_score,
                chirality_sign=result.chirality_sign,
                is_chiral=result.is_chiral,
                band_gap_shift=result.band_gap_shift,
            )
        except Exception as e:
            logger.debug("Error enriching %s: %s", structure.cod_id, e)
            return self._error_enrichment(structure, str(e))

    def enrich_batch(
        self,
        structures: list[CrystalStructure],
        cif_dir: str,
        chunk_size: int = 100,
        checkpoint_path: str | None = None,
        progress_path: str | None = None,
    ) -> list[CrystalEnrichment]:
        """Enrich a batch of structures with incremental checkpointing.

        Parameters
        ----------
        structures : list
            Structures to enrich.
        cif_dir : str
            Directory to store downloaded CIF files.
        chunk_size : int
            Save checkpoint after this many structures.
        checkpoint_path : str, optional
            Path to save/load enrichment checkpoint JSON.
        progress_path : str, optional
            Path to write lightweight progress JSON.
        """
        enrichments: list[CrystalEnrichment] = []
        start_idx = 0

        # Resume from checkpoint
        if checkpoint_path and Path(checkpoint_path).exists():
            enrichments = self.load_enriched(checkpoint_path)
            if len(enrichments) >= len(structures):
                logger.info("Enrichment complete: %d structures", len(enrichments))
                return enrichments
            start_idx = len(enrichments)
            logger.info("Resuming enrichment from %d/%d", start_idx, len(structures))

        total = len(structures)
        Path(cif_dir).mkdir(parents=True, exist_ok=True)

        for chunk_start in range(start_idx, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            logger.info(
                "Enriching %d-%d of %d (%.1f%%)",
                chunk_start, chunk_end, total,
                chunk_end / total * 100,
            )

            for structure in structures[chunk_start:chunk_end]:
                enrichment = self.enrich_structure(structure, cif_dir)
                enrichments.append(enrichment)

            # Save checkpoint
            if checkpoint_path:
                self.save_enriched(enrichments, checkpoint_path)

            # Write progress
            if progress_path:
                n_chiral = sum(1 for e in enrichments if e.is_chiral)
                n_errors = sum(1 for e in enrichments if e.error)
                progress = {
                    "stage": "enrich",
                    "structures_enriched": len(enrichments),
                    "structures_total": total,
                    "structures_chiral": n_chiral,
                    "structures_error": n_errors,
                    "percent_complete": round(len(enrichments) / total * 100, 1),
                }
                with open(progress_path, "w") as f:
                    json.dump(progress, f)

        return enrichments

    @staticmethod
    def save_enriched(enrichments: list[CrystalEnrichment], path: str) -> None:
        """Save enrichments to JSON."""
        data = [e.to_dict() for e in enrichments]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load_enriched(path: str) -> list[CrystalEnrichment]:
        """Load enrichments from JSON."""
        with open(path) as f:
            data = json.load(f)
        return [CrystalEnrichment.from_dict(d) for d in data]

    @staticmethod
    def _error_enrichment(
        structure: CrystalStructure, error: str
    ) -> CrystalEnrichment:
        return CrystalEnrichment(
            structure=structure,
            atom_labels=[],
            n_atoms=0,
            cd_activity=0.0,
            cd_sign="inactive",
            ciss_score=0.0,
            optical_rotation="inactive",
            chirality_score=0.0,
            chirality_sign=0.0,
            is_chiral=False,
            band_gap_shift=0.0,
            error=error,
        )
