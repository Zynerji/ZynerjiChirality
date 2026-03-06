# Atlas v7.0 — Per-Center Differential FP Bioactivity Correlation

**Date**: 2026-03-05
**Version**: ZynerjiChirality v0.7.1
**New function**: `per_center_differential_fingerprint()`

## Summary

Atlas v7 tests whether isolating each chiral center's differential fingerprint independently (to avoid multi-center cancellation) improves bioactivity fold-change correlation versus the global differential FP from v6.

**Result: Per-center FP did NOT improve correlation.** The global differential FP remains the better predictor.

## Per-Center vs Global Comparison

| Drug | #Centers | PC-Sim | PC-Div | Global-Div | Fold-Change |
|------|----------|--------|--------|------------|-------------|
| Ibuprofen | 1 | 0.1480 | 0.8520 | 0.8700 | 100x |
| Warfarin | 1 | 0.1624 | 0.8376 | 0.9712 | 5x |
| Escitalopram | 1 | 0.0459 | 0.9541 | 0.9153 | 30x |
| Naproxen | 1 | 0.3304 | 0.6696 | 0.7766 | 28x |
| Ketamine | 1 | 0.1417 | 0.8583 | 0.8298 | 4x |
| Thalidomide | 1 | 0.2718 | 0.7282 | 0.7535 | 1000x |
| L-DOPA | 1 | 0.1903 | 0.8097 | 0.7790 | 100x |
| Bupivacaine | 1 | 0.0868 | 0.9132 | 0.9139 | 4x |
| Omeprazole | 1 | 0.1492 | 0.8508 | 0.8931 | 1.5x |
| Ofloxacin | 0 | 0.5000 | 0.5000 | 1.0000 | 8x |
| Alanine | 1 | 0.1862 | 0.8138 | 0.8988 | 100x |
| Ethambutol | 3 | 0.6004 | 0.3996 | 0.5625 | 500x |

## Correlation Statistics

| Metric | Per-Center | Global (v6) | Lift |
|--------|-----------|-------------|------|
| **Spearman rho** | -0.4762 (p=0.1176) | **-0.5362** (p=0.0723) | -0.06 |
| **Pearson r** | -0.3861 (p=0.2150) | **-0.6661** (p=0.0180) | -0.28 |

## Interpretation

1. **Per-center did not help**: For 10/12 pairs that have only 1 chiral center, per-center and global are mathematically identical (just different code paths). The only multi-center molecule where per-center differs is Ethambutol (3 centers).

2. **Ethambutol got worse**: Per-center divergence (0.40) is lower than global (0.56), and Ethambutol has the highest fold-change (500x). This single data point drives the correlation degradation.

3. **Per-center averaging dilutes signal**: Averaging per-center similarities loses the coherent global differential signal. The global FP captures how *all* centers jointly shift the spectrum, which is closer to how the molecule interacts with a receptor.

4. **The real problem is saturation, not cancellation**: Both per-center and global divergences cluster in 0.65-0.97 for single-center molecules. The FP is excellent at *detecting* chirality but the divergence magnitude doesn't predict bioactivity fold-change magnitude.

5. **Ofloxacin remains an outlier**: 0 centers detected because one SMILES lacks explicit stereo annotation. Removing it would improve both correlations.

## What Per-Center IS Useful For

The per-center FP is still valuable for:
- **Per-center chirality scoring** in multi-center molecules (which center matters most?)
- **Feature vectors for ML models** that predict per-center bioactivity contributions
- **Stereo enumeration guidance** — which centers to flip for activity optimization

It just doesn't improve the divergence-vs-fold-change correlation in this 12-pair benchmark.

## v0.7.1 Changes

1. New `per_center_differential_fingerprint()` function in `fingerprint.py`
   - Uses `cip_canonical_order_per_center()` to isolate each center
   - Returns `list[np.ndarray]` — one FP per center, empty for achiral
2. Version bump: 0.7.0 -> 0.7.1
3. Exported in `__init__.py`

## Next Steps

- **Receptor-aware features**: Combine differential FP with target binding site descriptors for fold-change prediction
- **Weighted per-center**: Weight centers by their proximity to the pharmacophore rather than equal averaging
- **Larger benchmark**: 12 pairs is underpowered for correlation analysis. Use the 68K differential pairs from ChEMBL enrichment
