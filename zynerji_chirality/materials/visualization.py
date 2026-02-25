"""Material-specific visualization functions.

Plots for circular dichroism spectra, CISS diagrams, and material comparisons.
"""

from __future__ import annotations

import numpy as np

from zynerji_chirality.materials.detector import MaterialChiralityResult


def plot_cd_spectrum(result: MaterialChiralityResult, ax=None):
    """Plot a simulated CD spectrum from eigenvalue differences.

    Parameters
    ----------
    result : MaterialChiralityResult
        Material chirality analysis result.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates new figure if None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    spectral = result.spectral_coords
    ecos = spectral.eigenvalues_cos
    esin = spectral.eigenvalues_sin
    k = min(len(ecos), len(esin))

    if k == 0:
        ax.text(0.5, 0.5, "No spectral data", ha="center", va="center",
                transform=ax.transAxes, color="#888")
        return ax

    diff = ecos[:k] - esin[:k]

    # Create pseudo-wavelength axis from eigenvalue indices
    wavelengths = np.linspace(200, 400, 200)  # nm
    cd_spectrum = np.zeros_like(wavelengths)

    for i in range(k):
        center = 200 + (i / max(k - 1, 1)) * 200
        width = 20.0
        gaussian = np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
        cd_spectrum += diff[i] * gaussian

    # Normalize
    max_val = np.max(np.abs(cd_spectrum))
    if max_val > 0:
        cd_spectrum *= result.cd_activity / max_val

    ax.plot(wavelengths, cd_spectrum, color="#00d4ff", linewidth=1.5)
    ax.axhline(y=0, color="#333", linewidth=0.5, linestyle="--")
    ax.fill_between(wavelengths, cd_spectrum, where=cd_spectrum > 0,
                    alpha=0.3, color="#00ff88", label="Positive")
    ax.fill_between(wavelengths, cd_spectrum, where=cd_spectrum < 0,
                    alpha=0.3, color="#ff4444", label="Negative")

    ax.set_xlabel("Pseudo-wavelength (nm)")
    ax.set_ylabel("CD Intensity (a.u.)")
    title = f"Simulated CD Spectrum — {result.cd_sign} (intensity={result.cd_activity:.4f})"
    if result.smiles:
        title += f"\n{result.smiles[:50]}"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_facecolor("#0a0a0f")

    return ax


def plot_ciss_diagram(result: MaterialChiralityResult, ax=None):
    """Plot a spin selectivity visualization.

    Shows the CISS score as a polar-style diagram indicating
    helical electron transport asymmetry.

    Parameters
    ----------
    result : MaterialChiralityResult
        Material chirality analysis result.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates new figure if None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    score = result.ciss_score

    # Draw concentric circles for scale
    for r in [0.25, 0.5, 0.75, 1.0]:
        circle = plt.Circle((0, 0), r, fill=False, color="#333", linewidth=0.5)
        ax.add_patch(circle)
        ax.text(r + 0.02, 0, f"{r:.0%}", fontsize=7, color="#666", va="center")

    # Draw CISS score as filled circle
    color = "#00ff88" if score > 0.5 else "#ffaa00" if score > 0.2 else "#888"
    filled = plt.Circle((0, 0), score, fill=True, alpha=0.4, color=color)
    ax.add_patch(filled)

    # Arrow indicating spin direction
    if result.chirality_sign != 0:
        direction = 1 if result.chirality_sign > 0 else -1
        theta = np.linspace(0, direction * 2 * np.pi * score, 100)
        r = np.linspace(0.1, score, 100)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, color=color, linewidth=2)
        if len(x) > 1:
            ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2))

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_title(f"CISS Score: {score:.3f}", fontsize=11)
    ax.set_facecolor("#0a0a0f")
    ax.axis("off")

    return ax


def plot_material_comparison(
    results: list[MaterialChiralityResult],
    labels: list[str],
    ax=None,
):
    """Compare multiple materials side by side.

    Bar chart of CD activity, CISS score, and chirality score for each material.

    Parameters
    ----------
    results : list[MaterialChiralityResult]
        List of material results to compare.
    labels : list[str]
        Labels for each material.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates new figure if None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    n = len(results)
    x = np.arange(n)
    width = 0.25

    cd_vals = [r.cd_activity for r in results]
    ciss_vals = [r.ciss_score for r in results]
    chiral_vals = [r.chirality_score for r in results]

    ax.bar(x - width, cd_vals, width, label="CD Activity", color="#00d4ff", alpha=0.8)
    ax.bar(x, ciss_vals, width, label="CISS Score", color="#00ff88", alpha=0.8)
    ax.bar(x + width, chiral_vals, width, label="Chirality Score", color="#c084fc", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Material Chirality Comparison")
    ax.legend(fontsize=9)
    ax.set_facecolor("#0a0a0f")

    return ax
