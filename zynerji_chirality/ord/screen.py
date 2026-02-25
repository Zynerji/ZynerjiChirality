"""Catalyst screening for enantioselective reaction discovery.

Screens ORD-enriched reactions for interesting catalysts and
trains ee prediction models on the dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from zynerji_chirality.ord.extractor import EnantioselectiveReaction
from zynerji_chirality.ord.enricher import ReactionEnrichment

logger = logging.getLogger(__name__)


@dataclass
class CatalystScreeningHit:
    """A hit from the catalyst screening pipeline."""

    reaction: EnantioselectiveReaction
    enrichment: ReactionEnrichment | None
    hit_type: str
    description: str
    priority_score: float

    def __repr__(self) -> str:
        return (
            f"CatalystScreeningHit({self.hit_type}, score={self.priority_score:.2f}, "
            f"ee={self.reaction.ee_percent:.1f}%, {self.reaction.reaction_id})"
        )


class CatalystScreen:
    """Screen enriched reactions for catalyst discovery.

    Parameters
    ----------
    enrichments : list[ReactionEnrichment]
        Enriched enantioselective reactions.
    """

    def __init__(self, enrichments: list[ReactionEnrichment]):
        self.enrichments = enrichments

    def screen_high_ee(
        self, threshold: float = 90.0
    ) -> list[CatalystScreeningHit]:
        """Find reactions with high enantiomeric excess."""
        hits = []
        for e in self.enrichments:
            if e.reaction.ee_percent is None or e.reaction.ee_percent < threshold:
                continue
            hits.append(CatalystScreeningHit(
                reaction=e.reaction,
                enrichment=e,
                hit_type="high_ee",
                description=(
                    f"High ee ({e.reaction.ee_percent:.1f}%) "
                    f"catalyst={e.reaction.catalyst_smiles or 'unknown'}"
                ),
                priority_score=e.reaction.ee_percent,
            ))
        hits.sort(key=lambda h: h.priority_score, reverse=True)
        return hits

    def screen_novel_catalysts(
        self, store=None, similarity_threshold: float = 0.8
    ) -> list[CatalystScreeningHit]:
        """Find catalysts with no close match in the fingerprint store.

        Parameters
        ----------
        store : FingerprintStore, optional
            Fingerprint database for similarity search.
        similarity_threshold : float
            Below this similarity score, the catalyst is considered novel.
        """
        if store is None:
            return []

        hits = []
        seen_catalysts: set[str] = set()

        for e in self.enrichments:
            cat = e.reaction.catalyst_smiles
            if not cat or cat in seen_catalysts:
                continue
            seen_catalysts.add(cat)

            if e.catalyst_fp is None:
                continue

            try:
                # Search for similar catalysts in DB
                results = store.search_similar(
                    cat, k=1, mode="similar"
                )
                if not results or results[0][1] < similarity_threshold:
                    hits.append(CatalystScreeningHit(
                        reaction=e.reaction,
                        enrichment=e,
                        hit_type="novel_catalyst",
                        description=(
                            f"Novel catalyst (no close DB match): {cat[:60]}"
                        ),
                        priority_score=(
                            e.reaction.ee_percent or 0
                        ) / 100.0 + 0.5,
                    ))
            except Exception:
                pass

        hits.sort(key=lambda h: h.priority_score, reverse=True)
        return hits

    def screen_ee_prediction_gaps(
        self, predictor=None
    ) -> list[CatalystScreeningHit]:
        """Find reactions where predicted ee disagrees with actual.

        These are scientifically interesting — the model's spectral
        analysis disagrees with the experimental outcome.

        Parameters
        ----------
        predictor : EEPredictor, optional
            A fitted ee predictor.
        """
        if predictor is None:
            return []

        hits = []
        for e in self.enrichments:
            if (
                not e.reaction.catalyst_smiles
                or not e.reaction.substrate_smiles
                or e.reaction.ee_percent is None
            ):
                continue

            try:
                pred = predictor.predict_ee(
                    e.reaction.catalyst_smiles, e.reaction.substrate_smiles
                )
                gap = abs(pred.ee_percent - e.reaction.ee_percent)
                if gap > 20:  # >20% disagreement is interesting
                    hits.append(CatalystScreeningHit(
                        reaction=e.reaction,
                        enrichment=e,
                        hit_type="prediction_gap",
                        description=(
                            f"Prediction gap: predicted {pred.ee_percent:.1f}% "
                            f"vs actual {e.reaction.ee_percent:.1f}% "
                            f"(gap={gap:.1f}%)"
                        ),
                        priority_score=gap,
                    ))
            except Exception:
                pass

        hits.sort(key=lambda h: h.priority_score, reverse=True)
        return hits

    def screen_all(
        self, store=None, predictor=None
    ) -> list[CatalystScreeningHit]:
        """Run all screens and return combined hits."""
        hits = []
        hits.extend(self.screen_high_ee())
        hits.extend(self.screen_novel_catalysts(store))
        hits.extend(self.screen_ee_prediction_gaps(predictor))

        # Deduplicate by reaction_id, keeping highest priority
        seen: dict[str, CatalystScreeningHit] = {}
        for h in hits:
            rid = h.reaction.reaction_id
            if rid not in seen or h.priority_score > seen[rid].priority_score:
                seen[rid] = h
        unique = sorted(seen.values(), key=lambda h: h.priority_score, reverse=True)

        logger.info(
            "Total screening hits: %d (from %d candidates)", len(unique), len(hits)
        )
        return unique

    def report(
        self, hits: list[CatalystScreeningHit], output: str | None = None
    ) -> str:
        """Generate a text report of screening hits."""
        lines = [
            "=" * 72,
            "ORD Enantioselectivity Screening Report",
            f"Generated: {datetime.now().isoformat()}",
            f"Reactions analyzed: {len(self.enrichments)}",
            f"Total hits: {len(hits)}",
            "=" * 72,
            "",
        ]

        by_type: dict[str, list[CatalystScreeningHit]] = {}
        for h in hits:
            by_type.setdefault(h.hit_type, []).append(h)

        for hit_type, type_hits in sorted(by_type.items()):
            lines.append(f"--- {hit_type.upper()} ({len(type_hits)} hits) ---")
            lines.append("")
            for h in type_hits[:20]:
                lines.append(
                    f"  {h.reaction.reaction_id[:30]:<30s}  "
                    f"ee={h.reaction.ee_percent or 0:.1f}%  "
                    f"score={h.priority_score:.3f}  "
                    f"{h.description[:60]}"
                )
            if len(type_hits) > 20:
                lines.append(f"  ... and {len(type_hits) - 20} more")
            lines.append("")

        # Summary statistics
        if self.enrichments:
            ees = [
                e.reaction.ee_percent
                for e in self.enrichments
                if e.reaction.ee_percent is not None
            ]
            if ees:
                lines.append("--- SUMMARY STATISTICS ---")
                lines.append(f"  Mean ee: {np.mean(ees):.1f}%")
                lines.append(f"  Median ee: {np.median(ees):.1f}%")
                lines.append(f"  >90% ee: {sum(1 for x in ees if x > 90)} reactions")
                lines.append(f"  >95% ee: {sum(1 for x in ees if x > 95)} reactions")
                lines.append("")

        report_text = "\n".join(lines)

        if output:
            with open(output, "w") as f:
                f.write(report_text)
            logger.info("Report saved to %s", output)

        return report_text


def train_ee_model(
    enrichments: list[ReactionEnrichment],
) -> object:
    """Train an EEPredictor on ORD enrichment data.

    Parameters
    ----------
    enrichments : list[ReactionEnrichment]
        Enriched reactions with combined fingerprints.

    Returns
    -------
    EEPredictor
        A fitted ee prediction model.
    """
    from zynerji_chirality.catalysts.ee_predictor import EEPredictor

    # Filter to reactions with both catalyst and substrate
    training_data = []
    for e in enrichments:
        rxn = e.reaction
        if (
            rxn.catalyst_smiles
            and rxn.substrate_smiles
            and rxn.ee_percent is not None
        ):
            training_data.append({
                "catalyst": rxn.catalyst_smiles,
                "substrate": rxn.substrate_smiles,
                "ee": abs(rxn.ee_percent),
                "major": "R" if rxn.ee_percent >= 0 else "S",
            })

    if len(training_data) < 5:
        raise ValueError(
            f"Need at least 5 training reactions, got {len(training_data)}"
        )

    logger.info("Training ee model on %d reactions", len(training_data))
    predictor = EEPredictor()
    predictor.fit(training_data)
    logger.info("Model trained successfully")
    return predictor
