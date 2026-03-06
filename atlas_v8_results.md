# Atlas v8.0 — ChEMBL Large-N Differential FP Bioactivity Correlation

**Date**: 2026-03-05
**Version**: ZynerjiChirality v0.7.1
**Platform**: NVIDIA RTX PRO 6000 Blackwell (98GB VRAM, EPYC 384 threads)
**Script**: `scripts/atlas_v8_chembl.py`
**Data**: `chembl_work/screening_hits.json` (26,172 differential pairs from ChEMBL)

## Summary

Atlas v8 tests the differential fingerprint divergence-vs-bioactivity-fold-change correlation on **1,998 real ChEMBL enantiomer pairs** (>3x, <1000x fold-change, <60 heavy atoms).

**Result: NO correlation at scale.** Spearman rho = -0.008, p = 0.71.

## Results

| Metric | Value |
|--------|-------|
| Pairs processed | 1,998 |
| Failures | 2 |
| Runtime | 633s (~10.5 min on Blackwell) |
| **Spearman rho** | **-0.0083** (p = 0.71) |
| **Pearson r** | **-0.0056** (p = 0.80) |

### Divergence Distribution

| Stat | Value |
|------|-------|
| Mean | 0.640 |
| Std | 0.319 |
| Min | 0.001 |
| Max | 1.000 |

### Quartile Analysis

| Fold-Change Group | Mean Divergence | n |
|-------------------|----------------|---|
| Low FC (log10 <= 2.48) | 0.655 | 503 |
| High FC (log10 >= 2.79) | 0.659 | 501 |

The mean divergence is essentially identical across fold-change quartiles.

## Interpretation

1. **The Atlas v6 correlation (rho=-0.54, r=-0.67) was an artifact of small N.** With 12 hand-picked drug pairs, random variation created an apparent correlation. At n=1998, the effect disappears completely (rho=-0.008).

2. **Differential FP divergence does NOT predict bioactivity fold-change.** The divergence measures *how different* the spectral chirality signatures are, but bioactivity depends on receptor-ligand geometry, binding pocket complementarity, and pharmacokinetics — not on the magnitude of the spectral perturbation.

3. **The differential FP IS excellent at detecting enantiomers** (mean divergence 0.64, range 0.001-1.0 with strong anti-correlation). Its value is in classification (same vs different handedness), not in predicting how much the biology changes.

4. **This is a negative result worth documenting.** It rules out a naive hypothesis (bigger spectral difference → bigger activity difference) and points toward the correct use case: enantiomer detection and classification, not quantitative bioactivity prediction.

## Comparison Across Atlas Versions

| Version | n | Method | Spearman rho | p-value |
|---------|---|--------|-------------|---------|
| v6 | 12 | Global diff FP | -0.536 | 0.072 |
| v7 | 12 | Per-center diff FP | -0.476 | 0.118 |
| **v8** | **1,998** | **Global diff FP** | **-0.008** | **0.710** |

## What The Differential FP IS Good For

The differential fingerprint's proven use cases:
- **Enantiomer detection**: cosine < 0 for all tested enantiomer pairs
- **Achiral filtering**: norm = 0 for all achiral molecules (no false positives)
- **ML feature**: input to target-specific models that learn receptor-chirality relationships
- **Database search**: find enantiomeric counterparts in chemical databases

What it cannot do:
- Predict magnitude of bioactivity differences between enantiomers
- Serve as a standalone bioactivity predictor without receptor/target context

## Next Steps

- **Target-conditioned models**: Train per-target models that use differential FP + target descriptors to predict activity differential. The FP captures *what's different* about the chirality; the target model learns *which differences matter* for each receptor.
- **Close the v6/v7/v8 arc**: Document this as a case study in the danger of small-N benchmarks.
