"""Spectral embedding visualization for chirality analysis.

Plots 2D/3D spectral embeddings showing R vs S separation in
dual-helix spectral space.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from zynerji_chirality.chirality.detector import (
    ChiralityResult,
    HelixChiralityDetector,
)


def plot_eigenvalue_comparison(
    result_a: ChiralityResult,
    result_b: ChiralityResult,
    label_a: str = "Molecule A",
    label_b: str = "Molecule B",
    title: str = "Eigenvalue Comparison",
) -> Figure:
    """Plot cos/sin eigenvalue spectra for two molecules side by side.

    Shows the asymmetry between cos and sin helices that encodes chirality.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, result, label in [
        (axes[0], result_a, label_a),
        (axes[1], result_b, label_b),
    ]:
        k = min(len(result.cos_eigenvalues), len(result.sin_eigenvalues))
        if k == 0:
            ax.set_title(f"{label}\n(no eigenvalues)")
            continue

        indices = np.arange(k)
        width = 0.35

        ax.bar(indices - width / 2, result.cos_eigenvalues[:k], width,
               label="cos (right helix)", color="#2196F3", alpha=0.8)
        ax.bar(indices + width / 2, result.sin_eigenvalues[:k], width,
               label="sin (left helix)", color="#FF5722", alpha=0.8)

        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(f"{label}\nscore={result.chirality_score:.4f}, "
                     f"sign={result.chirality_sign:+.0f}")
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_spectral_embedding_2d(
    results: list[tuple[str, ChiralityResult]],
    title: str = "Spectral Embedding (2D)",
    dim_x: int = 0,
    dim_y: int = 1,
) -> Figure:
    """Plot 2D spectral embedding colored by chirality sign.

    Each molecule's atoms are plotted in spectral space. R-dominant
    molecules are blue, S-dominant are red, achiral are gray.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {1.0: "#2196F3", -1.0: "#FF5722", 0.0: "#9E9E9E"}

    for label, result in results:
        coords = result.spectral_coords.coords
        if coords.shape[1] <= max(dim_x, dim_y):
            continue

        color = colors.get(result.chirality_sign, "#9E9E9E")
        ax.scatter(
            coords[:, dim_x], coords[:, dim_y],
            c=color, alpha=0.6, s=30, label=label,
        )

    ax.set_xlabel(f"Spectral dimension {dim_x}")
    ax.set_ylabel(f"Spectral dimension {dim_y}")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_chirality_scores(
    results: list[tuple[str, ChiralityResult]],
    title: str = "Chirality Scores",
    threshold: float = 0.01,
) -> Figure:
    """Bar chart of chirality scores with threshold line."""
    fig, ax = plt.subplots(figsize=(12, 6))

    names = [name for name, _ in results]
    scores = [r.chirality_score for _, r in results]
    signs = [r.chirality_sign for _, r in results]

    colors = ["#2196F3" if s > 0 else "#FF5722" if s < 0 else "#9E9E9E" for s in signs]

    bars = ax.bar(range(len(names)), scores, color=colors, alpha=0.8)
    ax.axhline(y=threshold, color="black", linestyle="--", linewidth=1,
               label=f"Threshold ({threshold})")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Chirality Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_enantiomer_pair(
    smiles_a: str,
    smiles_b: str,
    label_a: str = "R",
    label_b: str = "S",
    detector: Optional[HelixChiralityDetector] = None,
) -> Figure:
    """Complete visualization for an enantiomer pair.

    Shows eigenvalue comparison, asymmetry vectors, and spectral embedding.
    """
    if detector is None:
        detector = HelixChiralityDetector()

    comp = detector.compare_enantiomers(smiles_a, smiles_b)
    result_a = comp.result_a
    result_b = comp.result_b

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Eigenvalue bars for A
    ax = axes[0, 0]
    k = min(len(result_a.cos_eigenvalues), len(result_a.sin_eigenvalues))
    if k > 0:
        idx = np.arange(k)
        w = 0.35
        ax.bar(idx - w / 2, result_a.cos_eigenvalues[:k], w,
               label="cos", color="#2196F3", alpha=0.8)
        ax.bar(idx + w / 2, result_a.sin_eigenvalues[:k], w,
               label="sin", color="#FF5722", alpha=0.8)
    ax.set_title(f"{label_a}: score={result_a.chirality_score:.4f}, "
                 f"sign={result_a.chirality_sign:+.0f}")
    ax.legend(fontsize=8)

    # Top-right: Eigenvalue bars for B
    ax = axes[0, 1]
    k = min(len(result_b.cos_eigenvalues), len(result_b.sin_eigenvalues))
    if k > 0:
        idx = np.arange(k)
        ax.bar(idx - w / 2, result_b.cos_eigenvalues[:k], w,
               label="cos", color="#2196F3", alpha=0.8)
        ax.bar(idx + w / 2, result_b.sin_eigenvalues[:k], w,
               label="sin", color="#FF5722", alpha=0.8)
    ax.set_title(f"{label_b}: score={result_b.chirality_score:.4f}, "
                 f"sign={result_b.chirality_sign:+.0f}")
    ax.legend(fontsize=8)

    # Bottom-left: Asymmetry comparison
    ax = axes[1, 0]
    k_a = min(len(result_a.cos_eigenvalues), len(result_a.sin_eigenvalues))
    k_b = min(len(result_b.cos_eigenvalues), len(result_b.sin_eigenvalues))
    if k_a > 0:
        asym_a = result_a.cos_eigenvalues[:k_a] - result_a.sin_eigenvalues[:k_a]
        ax.plot(asym_a, "o-", label=label_a, color="#2196F3", alpha=0.8)
    if k_b > 0:
        asym_b = result_b.cos_eigenvalues[:k_b] - result_b.sin_eigenvalues[:k_b]
        ax.plot(asym_b, "s-", label=label_b, color="#FF5722", alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("cos - sin asymmetry")
    ax.set_title("Asymmetry Vectors")
    ax.legend()

    # Bottom-right: Spectral embedding overlay
    ax = axes[1, 1]
    coords_a = result_a.spectral_coords.coords
    coords_b = result_b.spectral_coords.coords
    if coords_a.shape[1] >= 2:
        ax.scatter(coords_a[:, 0], coords_a[:, 1], c="#2196F3",
                   alpha=0.6, s=40, label=label_a)
    if coords_b.shape[1] >= 2:
        ax.scatter(coords_b[:, 0], coords_b[:, 1], c="#FF5722",
                   alpha=0.6, s=40, label=label_b)
    ax.set_xlabel("Spectral dim 0")
    ax.set_ylabel("Spectral dim 1")
    ax.set_title("Spectral Embedding")
    ax.legend()

    enantiomer_str = "ENANTIOMERS" if comp.are_enantiomers else "NOT ENANTIOMERS"
    fig.suptitle(
        f"{label_a} vs {label_b} — {enantiomer_str}\n"
        f"Spectral dist={comp.spectral_distance:.4f}, "
        f"Signs opposite={comp.signs_opposite}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return fig
