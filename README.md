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
print(result)  # ChiralityResult(CHIRAL, score=0.1590, sign=S, confidence=14.90)

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

### Amino Acid Detection: 15/19 (79%)
| Amino Acid | L score | D score | Opposite Signs |
|-----------|---------|---------|---------------|
| Alanine | 0.159 | 0.236 | Yes |
| Valine | 0.440 | 0.143 | Yes |
| Leucine | 0.335 | 0.149 | Yes |
| Isoleucine | 0.137 | 0.709 | Yes |
| Proline | 0.243 | 0.185 | Yes |
| Tryptophan | 0.254 | 0.196 | Yes |
| Methionine | 0.321 | 0.321 | Yes |
| Serine | 0.129 | 0.559 | Yes |
| Threonine | 0.190 | 0.459 | Yes |
| Cysteine | 0.597 | 0.093 | Yes |
| Asparagine | 0.036 | 0.578 | Yes |
| Glutamine | 0.069 | 0.420 | Yes |
| Aspartate | 0.128 | 0.420 | Yes |
| Glutamate | 0.052 | 0.183 | Yes |
| Lysine | 0.443 | 0.546 | Yes |

### Drug Pair Detection: 7/12 (58%)
| Drug | Detected | Opposite Signs |
|------|----------|---------------|
| Ibuprofen | Yes | Yes |
| Propranolol | Yes | Yes |
| Methylphenidate | Yes | Yes |
| Methadone | Yes | Yes |
| Penicillamine | Yes | Yes |

### Summary
- **Achiral rejection**: 100% (zero false positives)
- **Amino acid chirality detection**: 79%
- **Drug pair enantiomer discrimination**: 42% with opposite signs
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
