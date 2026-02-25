"""Iterative chirality optimization for stereoisomer selection.

Enumerates stereoisomers, scores them, and iteratively selects
the best configuration until convergence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from zynerji_chirality.generative.enumerator import (
    StereoisomerEnumerator,
    StereoisomerVariant,
)
from zynerji_chirality.generative.scorer import ChiralityScorer, ChiralityScore

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStep:
    """A single iteration of stereoisomer optimization."""
    iteration: int
    best_smiles: str
    best_score: float
    n_candidates_evaluated: int
    improvement: float


@dataclass
class OptimizationResult:
    """Complete result of stereoisomer optimization."""
    input_smiles: str
    best_smiles: str
    best_score: float
    steps: list[OptimizationStep] = field(default_factory=list)
    n_iterations: int = 0
    total_evaluated: int = 0
    all_variants: list[tuple[str, float]] = field(default_factory=list)


class ChiralOptimizer:
    """Iterative stereoisomer optimizer.

    Strategy: enumerate stereo -> score -> select best -> repeat with
    single-center flips until improvement < threshold.

    Parameters
    ----------
    scorer : ChiralityScorer
        Scoring function for molecules.
    enumerator : StereoisomerEnumerator or None
        Stereoisomer enumerator.
    max_iterations : int
        Maximum optimization iterations.
    convergence_threshold : float
        Stop when improvement per iteration < threshold.
    """

    def __init__(
        self,
        scorer: ChiralityScorer,
        enumerator: StereoisomerEnumerator | None = None,
        max_iterations: int = 5,
        convergence_threshold: float = 0.01,
    ):
        self.scorer = scorer
        self.enumerator = enumerator or StereoisomerEnumerator(
            compute_fingerprints=False,
        )
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def optimize(self, smiles: str) -> OptimizationResult:
        """Optimize the stereochemistry of a molecule.

        First iteration: enumerate all stereoisomers and score.
        Subsequent iterations: try single-center flips from current best.
        Stops when no improvement exceeds convergence_threshold.

        Parameters
        ----------
        smiles : str
            Input SMILES.

        Returns
        -------
        OptimizationResult
            Best stereoisomer found with optimization trace.
        """
        steps = []
        all_variants: list[tuple[str, float]] = []
        total_evaluated = 0
        current_smiles = smiles

        # Score original
        current_score_obj = self.scorer.score(current_smiles)
        current_best_score = current_score_obj.combined_score
        all_variants.append((current_smiles, current_best_score))
        total_evaluated += 1

        for iteration in range(self.max_iterations):
            if iteration == 0:
                # First iteration: enumerate all stereoisomers
                candidates = self.enumerator.enumerate(current_smiles)
            else:
                # Subsequent: single-center flips from current best
                candidates = self.enumerator.enumerate_single_flips(current_smiles)

            if not candidates:
                break

            # Score all candidates
            best_candidate = current_smiles
            best_score = current_best_score
            n_evaluated = 0

            for variant in candidates:
                try:
                    score = self.scorer.score(variant.canonical_smiles)
                    all_variants.append((variant.canonical_smiles, score.combined_score))
                    n_evaluated += 1

                    if score.combined_score > best_score:
                        best_score = score.combined_score
                        best_candidate = variant.canonical_smiles
                except Exception:
                    continue

            total_evaluated += n_evaluated
            improvement = best_score - current_best_score

            steps.append(OptimizationStep(
                iteration=iteration,
                best_smiles=best_candidate,
                best_score=best_score,
                n_candidates_evaluated=n_evaluated,
                improvement=improvement,
            ))

            if improvement < self.convergence_threshold:
                logger.debug(
                    "Converged at iteration %d (improvement %.4f < %.4f)",
                    iteration, improvement, self.convergence_threshold,
                )
                break

            current_smiles = best_candidate
            current_best_score = best_score

        return OptimizationResult(
            input_smiles=smiles,
            best_smiles=current_smiles,
            best_score=current_best_score,
            steps=steps,
            n_iterations=len(steps),
            total_evaluated=total_evaluated,
            all_variants=all_variants,
        )

    def optimize_batch(
        self,
        smiles_list: list[str],
        n_workers: int = 1,
    ) -> list[OptimizationResult]:
        """Optimize multiple molecules.

        Parameters
        ----------
        smiles_list : list[str]
            Input SMILES list.
        n_workers : int
            Number of parallel workers (currently sequential).

        Returns
        -------
        list[OptimizationResult]
            Optimization results, one per input.
        """
        # Sequential for now — parallelism would require serializable scorer
        results = []
        for smiles in smiles_list:
            try:
                result = self.optimize(smiles)
                results.append(result)
            except Exception as e:
                logger.warning("Optimization failed for %s: %s", smiles, e)
                results.append(OptimizationResult(
                    input_smiles=smiles,
                    best_smiles=smiles,
                    best_score=0.0,
                ))
        return results
