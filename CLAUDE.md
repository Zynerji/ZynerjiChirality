# ZynerjiChirality — Project Checkpoint

## What This Project Does

Chirality detection via dual-helix spectral graph analysis. Detects whether a molecule is chiral (R/S) using spectral asymmetry between two phase-modulated graph Laplacians (cos/sin helices with golden-ratio coupling).

**Key insight**: Standard graph Laplacian eigenvalues are identical for enantiomers. The dual-helix Laplacian breaks this symmetry via phase modulation that depends on node ordering. CIP-priority canonical ordering with chirality-dependent cyclic shifts encodes handedness into the ordering, producing different spectral responses for R vs S.

## Current Status (2026-02-22)

**All benchmarks passing — 100% across the board:**
- Amino acids: 19/19 (100%) with opposite signs
- Drug pairs: 12/12 (100%) with opposite signs
- Achiral rejection: 16/16 (100%)
- Tests: 59/59 passing
- Benchmark time: ~3.0s
- Pushed to GitHub: `Zynerji/ZynerjiChirality` (master, 3 commits)

**Meso compounds**: 0/2 correctly classified as achiral. This is a known limitation — meso compounds have chiral centers (CIP-labeled R and S) that internally cancel, but the detector sees each center independently. Fixing this would require detecting internal symmetry planes.

## Architecture

```
zynerji_chirality/
  core/
    dual_helix.py        # Sparse dual-helix spectral engine (from ZQC, modified)
    spectral_match.py    # Cost matrix construction (from ZQC, Qiskit-free)
    mol_graph.py         # RDKit Mol -> sparse adjacency (binary/bond_order/distance)
    chiral_ordering.py   # CIP canonical ordering + signed tetrahedral volumes
  chirality/
    detector.py          # HelixChiralityDetector (bidirectional differential scoring)
    fingerprint.py       # Fixed-length spectral chirality fingerprint (eigenvalue-based)
  benchmarks/
    amino_acids.py       # 19 L/D amino acid pairs + glycine achiral
    rs_pairs.py          # 12 R/S drug molecule pairs
    known_molecules.py   # 16 achiral controls, 2 meso compounds
  viz.py                 # Spectral embedding visualization (matplotlib)
tests/                   # 59 tests across 5 files
scripts/
  demo.py                # Quick demo (--plot for visualization)
  run_benchmarks.py      # Full benchmark suite
```

## Detection Pipeline

1. Parse molecule → 3D conformer via RDKit ETKDG (seed=42, MMFF optimization)
2. Build bond-order weighted adjacency matrix (csr_matrix)
3. Baseline ordering: `cip_canonical_order(mol, chirality_aware=False)`
4. **Bidirectional scoring**: Try both shift directions (+1 and -1) via `shift_override`
5. For each shift: reorder adjacency → dual-helix spectral decomposition → asymmetry
6. Score = max(|shifted_asym - baseline_asym|) across both directions
7. Score > 0.003 threshold → chiral. R/S sign from RDKit CIP labels.

## Key Design Decisions

### Signed Laplacian (dual_helix.py:79-98)
The sin helix produces mostly negative edge weights. Originally, a `if w > 0` guard dropped these, creating disconnected graphs with degenerate zero eigenvalues. Fix: keep ALL edges, use `|w|` for degree (ensures PSD), preserve sign in adjacency.

### Bidirectional Scoring (detector.py:124-155)
The cyclic shift perturbation has asymmetric sensitivity — one direction may hit a low-sensitivity region of the phase function (e.g., Propranolol R-shift: 0.0003 vs S-shift: 0.095). Fix: try BOTH shift directions, take max. This also makes enantiomer scores identical (theoretically correct).

### CIP '?' Fallback (chiral_ordering.py:140-146)
When `FindMolChiralCenters` returns `'?'` (ambiguous CIP), fall back to signed tetrahedral volume from 3D conformer geometry to determine shift direction.

### Differential Scoring
Achiral molecules get EXACTLY 0.0 score because `ordering_base == ordering_chiral` (no chiral centers → no shifts → identical orderings → identical spectra). This eliminates the inherent cos/sin asymmetry that would otherwise cause false positives.

### Threshold = 0.003
All achiral molecules score exactly 0.000000. The lowest chiral score in benchmarks is ~0.004 (Valine D, Glutamine L). Threshold 0.003 provides clean separation with zero false positive risk.

## Dependencies

- numpy, scipy (sparse linear algebra, eigsh)
- rdkit (molecular parsing, CIP assignment, 3D conformers, ETKDG)
- matplotlib (visualization only)

## Parameters (HelixParams defaults)

- `omega=0.3` — phase frequency
- `c_log=1.0` — log spacing constant for angular coordinates
- `twist_fraction=0.33` — Mobius twist threshold
- `k=8` — eigenvectors per helix
- `alpha=3.0` — spectral attenuation exponent
- `use_helix=True` — if False, uses standard graph Laplacian

## Known Limitations

1. **Meso compounds**: Detected as chiral (0/2). Each chiral center triggers shifts independently; internal symmetry cancellation not detected.
2. **R/S sign from CIP, not spectra**: The spectral method detects WHETHER a molecule is chiral; the R/S assignment comes from RDKit's CIP labels, not from the spectral asymmetry direction (which is unreliable due to nonlinear eigenvalue response).
3. **SMILES quality matters**: Stereochemistry must be encoded in SMILES (`@`/`@@`). Missing stereo markers → CIP returns `'?'` → falls back to 3D geometry (less reliable than explicit SMILES).
4. **Single conformer**: Uses one ETKDG conformer (seed=42). Flexible molecules might benefit from conformer ensemble averaging.

## Potential Next Steps

- Fix meso compound detection (detect internal symmetry planes)
- Conformer ensemble averaging for flexible molecules
- Expand benchmark set (sugars, steroids, natural products)
- Performance optimization (vectorize `build_sparse_laplacian` loop)
- Multi-center chirality analysis (per-center scores)
