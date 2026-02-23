"""Enantiomer screening interface.

Screens enantiomer pairs for drug discovery opportunities:
- Activity gaps: one enantiomer tested, the other not
- Differential activity: significant potency difference on same target
- Query-based: find enantiomers similar to a query molecule
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from zynerji_chirality.chembl.activity import PairActivity
from zynerji_chirality.chembl.pairs import EnantiomerPair
from zynerji_chirality.chirality.fingerprint import chirality_fingerprint
from zynerji_chirality.db.store import FingerprintStore

logger = logging.getLogger(__name__)


@dataclass
class ScreeningHit:
    """A hit from the enantiomer screen."""
    pair: EnantiomerPair
    pair_activity: PairActivity | None
    hit_type: str             # "activity_gap", "differential", "novel_analog"
    description: str
    priority_score: float     # higher = more interesting

    def __repr__(self) -> str:
        return (
            f"ScreeningHit({self.hit_type}, score={self.priority_score:.2f}, "
            f"{self.pair.mol_a.chembl_id} vs {self.pair.mol_b.chembl_id})"
        )


class EnantiomerScreen:
    """Screen enantiomer pairs for drug discovery opportunities.

    Parameters
    ----------
    store : FingerprintStore
        Fingerprint database for similarity search.
    pair_activities : list[PairActivity]
        Enriched enantiomer pairs with bioactivity data.
    """

    def __init__(
        self,
        store: FingerprintStore,
        pair_activities: list[PairActivity],
    ):
        self.store = store
        self.pair_activities = pair_activities

    def screen_activity_gaps(self) -> list[ScreeningHit]:
        """Find pairs where one enantiomer has activity data but other doesn't.

        These are candidates for testing the untested enantiomer.
        """
        hits = []
        for pa in self.pair_activities:
            if pa.gap_targets_a or pa.gap_targets_b:
                n_gaps = len(pa.gap_targets_a) + len(pa.gap_targets_b)
                n_activities = len(pa.activities_a) + len(pa.activities_b)

                # Priority: more gaps + more activity data = higher priority
                priority = n_gaps * (1 + 0.1 * n_activities)

                # Determine which side has the gap
                if pa.gap_targets_a and not pa.gap_targets_b:
                    desc = (
                        f"{pa.pair.mol_a.chembl_id} tested on {len(pa.gap_targets_a)} targets, "
                        f"{pa.pair.mol_b.chembl_id} untested. "
                        f"Recommend testing {pa.pair.mol_b.chembl_id}."
                    )
                elif pa.gap_targets_b and not pa.gap_targets_a:
                    desc = (
                        f"{pa.pair.mol_b.chembl_id} tested on {len(pa.gap_targets_b)} targets, "
                        f"{pa.pair.mol_a.chembl_id} untested. "
                        f"Recommend testing {pa.pair.mol_a.chembl_id}."
                    )
                else:
                    desc = (
                        f"Activity gaps in both directions: "
                        f"{pa.pair.mol_a.chembl_id} unique targets={len(pa.gap_targets_a)}, "
                        f"{pa.pair.mol_b.chembl_id} unique targets={len(pa.gap_targets_b)}."
                    )

                hits.append(ScreeningHit(
                    pair=pa.pair,
                    pair_activity=pa,
                    hit_type="activity_gap",
                    description=desc,
                    priority_score=priority,
                ))

        hits.sort(key=lambda h: h.priority_score, reverse=True)
        logger.info("Found %d activity gap hits", len(hits))
        return hits

    def screen_differential(self, min_fold_change: float = 3.0) -> list[ScreeningHit]:
        """Find pairs with significant activity difference on shared targets.

        These reveal chirality-sensitive drug targets where the handedness
        of the molecule critically affects its pharmacological action.
        """
        hits = []
        for pa in self.pair_activities:
            significant = [
                dt for dt in pa.differential_targets
                if dt.get("fold_change", 0) >= min_fold_change
            ]
            if not significant:
                continue

            # Priority: higher fold change = more interesting
            max_fold = max(dt["fold_change"] for dt in significant)
            priority = max_fold * len(significant)

            target_descs = []
            for dt in sorted(significant, key=lambda d: d["fold_change"], reverse=True):
                target_descs.append(
                    f"  {dt['target_name']} ({dt['target_id']}): "
                    f"{dt.get('fold_change', 0):.1f}x differential"
                )

            desc = (
                f"{len(significant)} chirality-sensitive target(s):\n"
                + "\n".join(target_descs)
            )

            hits.append(ScreeningHit(
                pair=pa.pair,
                pair_activity=pa,
                hit_type="differential",
                description=desc,
                priority_score=priority,
            ))

        hits.sort(key=lambda h: h.priority_score, reverse=True)
        logger.info("Found %d differential activity hits", len(hits))
        return hits

    def screen_by_query(
        self,
        query_smiles: str,
        k: int = 10,
    ) -> list[ScreeningHit]:
        """Given a molecule, find enantiomers/analogs with known activity.

        Parameters
        ----------
        query_smiles : str
            SMILES of query molecule.
        k : int
            Number of results to return.

        Returns
        -------
        list[ScreeningHit]
            Hits ranked by similarity to query.
        """
        try:
            query_fp = chirality_fingerprint(query_smiles)
        except Exception as e:
            logger.error("Failed to fingerprint query: %s", e)
            return []

        results = self.store.search_similar(query_fp, k=k * 2)  # get extras for matching

        # Build ChEMBL ID → PairActivity lookup
        id_to_pa: dict[str, PairActivity] = {}
        for pa in self.pair_activities:
            id_to_pa[pa.pair.mol_a.chembl_id] = pa
            id_to_pa[pa.pair.mol_b.chembl_id] = pa

        hits = []
        seen_pairs: set[str] = set()
        for sr in results:
            chembl_id = sr.name or ""
            pa = id_to_pa.get(chembl_id)
            if pa is None:
                continue

            pair_key = pa.pair.connectivity_smiles
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            n_activities = len(pa.activities_a) + len(pa.activities_b)
            desc = (
                f"Similar to query ({sr.similarity:.3f}): "
                f"{pa.pair.mol_a.chembl_id} vs {pa.pair.mol_b.chembl_id} "
                f"({pa.pair.relationship}), {n_activities} activity records"
            )

            hits.append(ScreeningHit(
                pair=pa.pair,
                pair_activity=pa,
                hit_type="novel_analog",
                description=desc,
                priority_score=sr.similarity * (1 + 0.1 * n_activities),
            ))

            if len(hits) >= k:
                break

        logger.info("Found %d query hits for %s", len(hits), query_smiles)
        return hits

    def screen_all(self, min_fold_change: float = 3.0) -> list[ScreeningHit]:
        """Run all screens and return combined, deduplicated hits."""
        all_hits = []
        all_hits.extend(self.screen_activity_gaps())
        all_hits.extend(self.screen_differential(min_fold_change=min_fold_change))

        # Deduplicate by pair connectivity
        seen: set[str] = set()
        unique_hits = []
        for hit in sorted(all_hits, key=lambda h: h.priority_score, reverse=True):
            key = f"{hit.hit_type}:{hit.pair.connectivity_smiles}"
            if key not in seen:
                seen.add(key)
                unique_hits.append(hit)

        return unique_hits

    def report(self, hits: list[ScreeningHit], output: str | None = None) -> str:
        """Generate human-readable screening report.

        Parameters
        ----------
        hits : list[ScreeningHit]
            Screening hits to report.
        output : str, optional
            Path to write report file. If None, returns string only.

        Returns
        -------
        str
            Formatted report text.
        """
        lines = [
            "=" * 72,
            "ZynerjiChirality Enantiomer Screening Report",
            f"Generated: {datetime.now().isoformat()}",
            f"Total hits: {len(hits)}",
            "=" * 72,
            "",
        ]

        # Summary by type
        by_type: dict[str, list[ScreeningHit]] = {}
        for hit in hits:
            by_type.setdefault(hit.hit_type, []).append(hit)

        for hit_type, type_hits in sorted(by_type.items()):
            lines.append(f"## {hit_type.upper()} ({len(type_hits)} hits)")
            lines.append("-" * 40)

            for i, hit in enumerate(type_hits[:20]):  # Top 20 per type
                lines.append(f"\n### Hit {i + 1} (priority={hit.priority_score:.2f})")
                lines.append(f"Pair: {hit.pair.mol_a.chembl_id} vs {hit.pair.mol_b.chembl_id}")
                lines.append(f"Relationship: {hit.pair.relationship}")
                lines.append(f"Connectivity: {hit.pair.connectivity_smiles}")
                lines.append(f"Mol A: {hit.pair.mol_a.canonical_smiles}")
                lines.append(f"Mol B: {hit.pair.mol_b.canonical_smiles}")
                lines.append(f"Description: {hit.description}")

            if len(type_hits) > 20:
                lines.append(f"\n... and {len(type_hits) - 20} more {hit_type} hits")
            lines.append("")

        # Stats
        lines.extend([
            "=" * 72,
            "Summary Statistics",
            f"  Total pairs screened: {len(self.pair_activities)}",
            f"  Total hits: {len(hits)}",
            f"  Activity gap hits: {len(by_type.get('activity_gap', []))}",
            f"  Differential hits: {len(by_type.get('differential', []))}",
            f"  Novel analog hits: {len(by_type.get('novel_analog', []))}",
            f"  Molecules in store: {self.store.count()}",
            "=" * 72,
        ])

        report_text = "\n".join(lines)

        if output:
            with open(output, "w") as f:
                f.write(report_text)
            logger.info("Report written to %s", output)

        return report_text
