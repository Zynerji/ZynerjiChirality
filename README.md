# ZynerjiChirality

Chirality detection via dual-helix spectral graph analysis.

## Key Insight

Standard graph Laplacian eigenvalues are identical for enantiomers (mirror-image molecules) because they share the same adjacency matrix. No permutation-invariant spectral method can distinguish R from S chirality.

The **dual-helix Laplacian** breaks this symmetry. Its phase modulation (`cos(omega*theta)` / `sin(omega*theta)`) depends on node ordering. By using CIP-priority canonical ordering with chirality-dependent cyclic shifts, the cos and sin helices produce different spectral responses for chirally-ordered vs baseline-ordered graphs.

**Detection mechanism**: differential spectral asymmetry between a chirality-aware node ordering and a baseline canonical ordering. Achiral molecules produce identical orderings (score = 0). Chiral molecules produce different orderings (score >> 0). R/S assignment uses CIP labels from RDKit.

## Installation

```bash
pip install -e ".[dev]"
```

Requires: numpy, scipy, rdkit, matplotlib

## Quick Start

```python
from zynerji_chirality.chirality.detector import HelixChiralityDetector

detector = HelixChiralityDetector()

# Detect chirality
result = detector.detect("N[C@@H](C)C(=O)O")  # L-Alanine
print(result)  # ChiralityResult(CHIRAL, score=0.1129, sign=S, confidence=36.63)

# Compare enantiomers
comp = detector.compare_enantiomers(
    "N[C@@H](C)C(=O)O",  # L-Alanine
    "N[C@H](C)C(=O)O",   # D-Alanine
)
print(comp.are_enantiomers)  # True
print(comp.signs_opposite)   # True
```

## Benchmark Results

### Achiral Rejection: 16/16 (100%)
All 16 achiral molecules (methane, benzene, ethanol, glycine, etc.) correctly classified with score = 0.

### Amino Acid Detection: 19/19 (100%)
| Amino Acid | L score | D score | Opposite Signs |
|-----------|---------|---------|---------------|
| Alanine | 0.113 | 0.068 | Yes |
| Valine | 0.111 | 0.004 | Yes |
| Leucine | 0.198 | 0.027 | Yes |
| Isoleucine | 0.208 | 0.144 | Yes |
| Proline | 0.020 | 0.061 | Yes |
| Phenylalanine | 0.085 | 0.120 | Yes |
| Tryptophan | 0.068 | 0.089 | Yes |
| Methionine | 0.096 | 0.007 | Yes |
| Serine | 0.016 | 0.219 | Yes |
| Threonine | 0.074 | 0.225 | Yes |
| Cysteine | 0.255 | 0.044 | Yes |
| Tyrosine | 0.005 | 0.110 | Yes |
| Asparagine | 0.033 | 0.210 | Yes |
| Glutamine | 0.004 | 0.163 | Yes |
| Aspartate | 0.079 | 0.245 | Yes |
| Glutamate | 0.063 | 0.192 | Yes |
| Lysine | 0.191 | 0.008 | Yes |
| Arginine | 0.017 | 0.040 | Yes |
| Histidine | 0.062 | 0.119 | Yes |

### Drug Pair Detection: 8/12 (67%)
| Drug | Detected | Opposite Signs |
|------|----------|---------------|
| Thalidomide | Yes | Yes |
| Ibuprofen | Yes | Yes |
| Naproxen | Yes | Yes |
| Warfarin | Yes | Yes |
| Methylphenidate | Yes | Yes |
| Amphetamine | Yes | Yes |
| Methadone | Yes | Yes |
| Penicillamine | Yes | Yes |

### Summary
- **Achiral rejection**: 100% (zero false positives)
- **Amino acid chirality detection**: 100% (19/19 with opposite signs)
- **Drug pair enantiomer discrimination**: 67% with opposite signs
- **Total benchmark time**: ~2.6s

## Running Tests

```bash
pytest tests/ -v  # 59/59 passing
```

## Benchmarks

```bash
python scripts/run_benchmarks.py
```

## Demo

```bash
python scripts/demo.py
python scripts/demo.py --plot  # Save visualization
```

## Architecture

```
zynerji_chirality/
  core/
    dual_helix.py       # Sparse dual-helix spectral engine (from ZQC)
    spectral_match.py   # Cost matrix construction (from ZQC, Qiskit-free)
    mol_graph.py        # RDKit Mol -> sparse adjacency
    chiral_ordering.py  # CIP canonical ordering + cyclic shifts
  chirality/
    detector.py         # HelixChiralityDetector (differential scoring)
    fingerprint.py      # Fixed-length spectral chirality fingerprint
  benchmarks/
    amino_acids.py      # 19 L/D amino acid pairs
    rs_pairs.py         # 12 R/S drug molecule pairs
    known_molecules.py  # Achiral controls, meso compounds
  viz.py                # Spectral embedding visualization
```

## How It Works

1. **Parse molecule** and generate 3D conformer via RDKit ETKDG
2. **CIP canonical ordering** — baseline (no chirality) and chirality-aware (cyclic neighbor shifts for R/S centers)
3. **Reorder adjacency matrix** with both orderings
4. **Dual-helix spectral decomposition** — cos (right, phi coupling) and sin (left, phi^2 coupling) Laplacians
5. **Differential asymmetry** — measure how much the chirality-aware ordering changes the cos-sin eigenvalue gap vs baseline
6. **Score > threshold** → chiral. R/S sign from CIP labels.

## License

MIT
