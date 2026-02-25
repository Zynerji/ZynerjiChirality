"""Crystal structure screening for chiral materials discovery.

Screens COD-enriched structures for materials with interesting chirality
properties: high CD activity, high CISS, chiral perovskites, etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from zynerji_chirality.cod.download import CrystalStructure
from zynerji_chirality.cod.enricher import CrystalEnrichment

logger = logging.getLogger(__name__)


@dataclass
class CrystalScreeningHit:
    """A hit from the crystal screening pipeline."""

    structure: CrystalStructure
    enrichment: CrystalEnrichment
    hit_type: str
    description: str
    priority_score: float

    def __repr__(self) -> str:
        return (
            f"CrystalScreeningHit({self.hit_type}, score={self.priority_score:.2f}, "
            f"COD {self.structure.cod_id}, {self.structure.formula})"
        )


class CrystalScreen:
    """Screen enriched crystal structures for materials discovery.

    Parameters
    ----------
    enrichments : list[CrystalEnrichment]
        Enriched crystal structures.
    """

    def __init__(self, enrichments: list[CrystalEnrichment]):
        self.enrichments = enrichments

    def screen_high_cd(
        self, threshold: float = 0.3
    ) -> list[CrystalScreeningHit]:
        """Find structures with high circular dichroism activity."""
        hits = []
        for e in self.enrichments:
            if e.error or e.cd_activity < threshold:
                continue
            hits.append(CrystalScreeningHit(
                structure=e.structure,
                enrichment=e,
                hit_type="high_cd",
                description=(
                    f"High CD activity ({e.cd_activity:.3f}, {e.cd_sign}) "
                    f"in {e.structure.formula} (SG {e.structure.sg_number})"
                ),
                priority_score=e.cd_activity,
            ))
        hits.sort(key=lambda h: h.priority_score, reverse=True)
        return hits

    def screen_high_ciss(
        self, threshold: float = 0.5
    ) -> list[CrystalScreeningHit]:
        """Find structures with high chirality-induced spin selectivity."""
        hits = []
        for e in self.enrichments:
            if e.error or e.ciss_score < threshold:
                continue
            hits.append(CrystalScreeningHit(
                structure=e.structure,
                enrichment=e,
                hit_type="high_ciss",
                description=(
                    f"High CISS score ({e.ciss_score:.3f}) "
                    f"in {e.structure.formula} (SG {e.structure.sg_number})"
                ),
                priority_score=e.ciss_score,
            ))
        hits.sort(key=lambda h: h.priority_score, reverse=True)
        return hits

    def screen_chiral_perovskites(self) -> list[CrystalScreeningHit]:
        """Find chiral structures with perovskite-like formulas (ABX3)."""
        hits = []
        for e in self.enrichments:
            if e.error or not e.is_chiral:
                continue
            formula = e.structure.formula.strip()
            # Simple heuristic: look for 3:1 element ratios or known perovskite SGs
            perovskite_sgs = {99, 107, 143, 144, 145, 146, 150, 152, 154, 168, 169}
            if e.structure.sg_number in perovskite_sgs or "3" in formula:
                hits.append(CrystalScreeningHit(
                    structure=e.structure,
                    enrichment=e,
                    hit_type="chiral_perovskite",
                    description=(
                        f"Possible chiral perovskite: {formula} "
                        f"(SG {e.structure.sg_number}, chirality={e.chirality_score:.3f})"
                    ),
                    priority_score=e.chirality_score + e.cd_activity,
                ))
        hits.sort(key=lambda h: h.priority_score, reverse=True)
        return hits

    def screen_all(self) -> list[CrystalScreeningHit]:
        """Run all screens and return combined hits."""
        hits = []
        hits.extend(self.screen_high_cd())
        hits.extend(self.screen_high_ciss())
        hits.extend(self.screen_chiral_perovskites())

        # Deduplicate by COD ID, keeping highest priority
        seen: dict[str, CrystalScreeningHit] = {}
        for h in hits:
            cid = h.structure.cod_id
            if cid not in seen or h.priority_score > seen[cid].priority_score:
                seen[cid] = h
        unique = sorted(seen.values(), key=lambda h: h.priority_score, reverse=True)

        logger.info("Total screening hits: %d (from %d candidates)", len(unique), len(hits))
        return unique

    def report(
        self, hits: list[CrystalScreeningHit], output: str | None = None
    ) -> str:
        """Generate a text report of screening hits.

        Parameters
        ----------
        hits : list
            Screening hits to report.
        output : str, optional
            Path to save report file.
        """
        lines = [
            "=" * 72,
            "COD Crystal Chirality Screening Report",
            f"Generated: {datetime.now().isoformat()}",
            f"Structures analyzed: {len(self.enrichments)}",
            f"Total hits: {len(hits)}",
            "=" * 72,
            "",
        ]

        # Group by type
        by_type: dict[str, list[CrystalScreeningHit]] = {}
        for h in hits:
            by_type.setdefault(h.hit_type, []).append(h)

        for hit_type, type_hits in sorted(by_type.items()):
            lines.append(f"--- {hit_type.upper()} ({len(type_hits)} hits) ---")
            lines.append("")
            for h in type_hits[:20]:  # Top 20 per type
                lines.append(
                    f"  COD {h.structure.cod_id:>8s}  "
                    f"{h.structure.formula:<20s}  "
                    f"SG {h.structure.sg_number:>3d}  "
                    f"score={h.priority_score:.3f}  "
                    f"{h.description}"
                )
            if len(type_hits) > 20:
                lines.append(f"  ... and {len(type_hits) - 20} more")
            lines.append("")

        report_text = "\n".join(lines)

        if output:
            with open(output, "w") as f:
                f.write(report_text)
            logger.info("Report saved to %s", output)

        return report_text
