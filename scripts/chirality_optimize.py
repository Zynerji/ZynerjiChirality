#!/usr/bin/env python3
"""Chirality-aware molecular optimization.

Usage:
    python scripts/chirality_optimize.py --smiles "N[C@@H](C)C(=O)O"
    python scripts/chirality_optimize.py --input molecules.txt --output ranked.tsv
    python scripts/chirality_optimize.py --filter molecules.txt --min-score 0.3
    python scripts/chirality_optimize.py --mmp "N[C@@H](C)C(=O)O"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _build_scorer(args):
    """Build ChiralityScorer from args, loading optional models."""
    from zynerji_chirality.generative.scorer import ChiralityScorer

    target_profiler = None
    admet_profiler = None
    store = None

    if args.target_models_dir and Path(args.target_models_dir).exists():
        from zynerji_chirality.targets.profiler import TargetProfiler
        target_profiler = TargetProfiler.load_from_directory(args.target_models_dir)

    if args.admet_models_dir and Path(args.admet_models_dir).exists():
        from zynerji_chirality.admet.profiler import ADMETProfiler
        admet_profiler = ADMETProfiler.load_from_directory(args.admet_models_dir)

    if args.db and Path(args.db).exists():
        from zynerji_chirality.db.store import FingerprintStore
        store = FingerprintStore(args.db)

    return ChiralityScorer(
        target_profiler=target_profiler,
        admet_profiler=admet_profiler,
        store=store,
    )


def main():
    parser = argparse.ArgumentParser(description="Chirality-aware molecular optimization")
    parser.add_argument("--smiles", help="Single SMILES to optimize")
    parser.add_argument("--input", help="Input file (one SMILES per line)")
    parser.add_argument("--filter", dest="filter_file", help="Filter and rank molecules from file")
    parser.add_argument("--mmp", help="Find beneficial chirality flips")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--work-dir", default=".", help="Working directory")
    parser.add_argument("--db", help="Fingerprint database for novelty")
    parser.add_argument("--target-models-dir", help="Target models directory")
    parser.add_argument("--admet-models-dir", help="ADMET models directory")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    scorer = _build_scorer(args)

    if args.smiles:
        from zynerji_chirality.generative.optimizer import ChiralOptimizer

        optimizer = ChiralOptimizer(scorer=scorer, max_iterations=args.max_iterations)
        result = optimizer.optimize(args.smiles)

        print(f"\nStereoisomer Optimization")
        print(f"  Input:  {result.input_smiles}")
        print(f"  Best:   {result.best_smiles}")
        print(f"  Score:  {result.best_score:.4f}")
        print(f"  Iterations: {result.n_iterations}")
        print(f"  Evaluated:  {result.total_evaluated}")
        print()
        for step in result.steps:
            print(
                f"  Step {step.iteration}: score={step.best_score:.4f} "
                f"(+{step.improvement:.4f}) "
                f"candidates={step.n_candidates_evaluated}"
            )
        print(f"\nAll variants:")
        for smiles, score in sorted(result.all_variants, key=lambda x: -x[1]):
            marker = " <-- best" if smiles == result.best_smiles else ""
            print(f"  {score:.4f}  {smiles}{marker}")
        return

    if args.mmp:
        from zynerji_chirality.generative.mmp import ChiralMMPAnalyzer

        analyzer = ChiralMMPAnalyzer(scorer=scorer)
        flips = analyzer.find_beneficial_flips(args.mmp)

        print(f"\nBeneficial Chirality Flips for: {args.mmp}")
        if not flips:
            print("  No beneficial flips found.")
        else:
            for mmp in flips:
                print(
                    f"  {mmp.transformation}: "
                    f"{mmp.score_a.combined_score:.4f} -> {mmp.score_b.combined_score:.4f} "
                    f"(+{mmp.score_delta:.4f})"
                )
                print(f"    {mmp.smiles_b}")
        return

    if args.filter_file:
        from zynerji_chirality.generative.filter import GenerativeFilter

        gen_filter = GenerativeFilter(
            scorer=scorer, min_score=args.min_score, enumerate_stereo=True,
        )
        result = gen_filter.filter_from_file(args.filter_file, args.output)

        print(f"\nFilter Results")
        print(f"  Input:    {result.input_count}")
        print(f"  Output:   {result.output_count}")
        print(f"  Filtered: {result.filtered_count}")
        for score in result.ranked_molecules[:20]:
            print(f"  {score.combined_score:.4f}  {score.smiles}")
        return

    if args.input:
        from zynerji_chirality.generative.optimizer import ChiralOptimizer

        optimizer = ChiralOptimizer(scorer=scorer, max_iterations=args.max_iterations)
        with open(args.input) as f:
            smiles_list = [line.strip() for line in f if line.strip()]

        results = optimizer.optimize_batch(smiles_list)

        if args.output:
            with open(args.output, "w") as f:
                f.write("input_smiles\tbest_smiles\tscore\titerations\n")
                for r in results:
                    f.write(f"{r.input_smiles}\t{r.best_smiles}\t{r.best_score:.4f}\t{r.n_iterations}\n")
            print(f"Wrote {len(results)} results to {args.output}")
        else:
            for r in results:
                print(f"  {r.input_smiles} -> {r.best_smiles} (score={r.best_score:.4f})")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
