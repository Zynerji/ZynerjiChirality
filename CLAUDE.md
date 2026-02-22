# ZynerjiChirality — Project Checkpoint

## What This Project Does

Chirality detection via dual-helix spectral graph analysis. Detects whether a molecule is chiral (R/S) using spectral asymmetry between two phase-modulated graph Laplacians (cos/sin helices with golden-ratio coupling).

**Key insight**: Standard graph Laplacian eigenvalues are identical for enantiomers. The dual-helix Laplacian breaks this symmetry via phase modulation that depends on node ordering. CIP-priority canonical ordering with chirality-dependent cyclic shifts encodes handedness into the ordering, producing different spectral responses for R vs S.

## Current Status (v0.2.0, 2026-02-22)

**All core benchmarks passing — 100%:**
- Amino acids: 19/19 (100%) with opposite signs
- Drug pairs: 12/12 (100%) with opposite signs
- Achiral rejection: 16/16 (100%)
- Meso compounds: 2/2 correctly achiral (FIXED in v0.2.0)
- Tests: 100+ passing
- Pushed to GitHub: `Zynerji/ZynerjiChirality`

**Expanded benchmarks:**
- Sugars: 4/4 detected, 3/4 opposite signs (glucose sign issue due to equal R/S count)
- Natural products: 3/3 (menthol, camphor, carvone)
- Steroids: 2/2 (cholesterol, testosterone)
- Tough achiral: 3/4 (norbornane embedding issue)

## Architecture

```
zynerji_chirality/
  core/
    dual_helix.py        # Sparse dual-helix spectral engine
    spectral_match.py    # Cost matrix construction
    mol_graph.py         # RDKit → sparse adjacency + axial perturbation + ensemble
    chiral_ordering.py   # CIP ordering + per-center + axial/planar detection
  chirality/
    detector.py          # HelixChiralityDetector (detect, detect_per_center, detect_ensemble)
    fingerprint.py       # Spectral fingerprint + batch + cached projection
  benchmarks/
    amino_acids.py       # 19 L/D amino acid pairs + glycine
    rs_pairs.py          # 12 R/S drug pairs
    known_molecules.py   # 16 achiral + 2 meso compounds
    expanded.py          # Sugars, natural products, steroids, tough achiral
    axial_chirality.py   # Biaryl atropisomer benchmarks
    planar_chirality.py  # Paracyclophane benchmarks
  db/
    store.py             # SQLite FingerprintStore with similarity search
  ml/
    features.py          # Combined feature vectors (spectral + RDKit + per-center + ensemble)
    base_model.py        # BaseChiralPredictor (sklearn RF/GB)
    activity_model.py    # ChiralActivityPredictor
    admet_model.py       # ChiralADMETPredictor
  reactions/
    reaction_graph.py    # ReactionStereoAnalyzer (center fate tracking)
    transfer.py          # ProchiralFaceDetector, ChiralityTransferPredictor
    retro.py             # RetroChiralityPlanner (strategy suggestions)
  viz.py                 # Spectral embedding visualization
tests/                   # 100+ tests across 10 files
scripts/
  demo.py                # Quick demo with all features
  run_benchmarks.py      # Full benchmark suite (5 categories)
  ingest.py              # CLI for batch SMILES ingestion
  compare_fingerprints.py # ZynerjiChirality vs ECFP4 vs MACCS comparison
  train_activity_model.py # Demo activity model training
```

## Detection Pipeline

1. Parse molecule → 3D conformer via RDKit ETKDG (seed=42, MMFF optimization)
2. **Meso check**: equal R/S + stereo-stripped canonical rank matching → achiral
3. Build bond-order weighted adjacency matrix (csr_matrix)
4. Baseline ordering: `cip_canonical_order(mol, chirality_aware=False)`
5. **Bidirectional scoring**: Try both shift directions (+1 and -1) via `shift_override`
6. For each shift: reorder adjacency → dual-helix spectral decomposition → asymmetry
7. Score = max(|shifted_asym - baseline_asym|) across both directions
8. **Axial chirality fallback**: if tetrahedral scoring = 0, check for atropisomers
9. Score > 0.003 threshold → chiral. R/S sign from RDKit CIP labels.

## Key Features (v0.2.0)

### Phase 1: Core Engine Hardening
- **Meso detection**: `_is_meso()` — stereo-stripped canonical rank matching
- **Per-center scores**: `detect_per_center()` — isolates each center's contribution
- **Conformer ensemble**: `detect_ensemble()` — multi-conformer aggregation

### Phase 2: Extended Chirality Types
- **Axial chirality**: `find_axial_centers()` + RDKit atropisomer API + geometry fallback
- **Planar chirality**: `find_planar_centers()` — ring plane + substituent face analysis

### Phase 3: Fingerprint Database
- **FingerprintStore**: SQLite-backed with `add()`, `get()`, `batch_add()`, `search_similar()`
- **Similarity search**: cosine k-NN ("similar") or opposite-sign filter ("enantiomer")
- **Batch fingerprinting**: `batch_fingerprint()` with multiprocessing
- **Cached projection matrix**: `_PROJ_CACHE` for deterministic, fast random projection

### Phase 4: ML Property Prediction
- **Feature pipeline**: `build_feature_vector()` — spectral FP + RDKit descriptors + per-center + ensemble
- **BaseChiralPredictor**: sklearn RF/GB with fit/predict/cross_validate/save/load
- **ChiralActivityPredictor**: enantioselective pharmacological activity
- **ChiralADMETPredictor**: ADMET property prediction
- Optional dependency: `pip install zynerji-chirality[ml]`

### Phase 5: Reaction Stereochemistry
- **ReactionStereoAnalyzer**: tracks stereocenter fate (new/inverted/retained/lost) via atom mapping
- **ProchiralFaceDetector**: identifies prochiral sp2 carbons + face features
- **ChiralityTransferPredictor**: framework for face selectivity prediction
- **RetroChiralityPlanner**: strategy suggestions for stereocenter synthesis

## Dependencies

- numpy, scipy (sparse linear algebra, eigsh)
- rdkit (molecular parsing, CIP, 3D conformers, ETKDG)
- matplotlib (visualization only)
- scikit-learn (optional, for ML models)

## Known Limitations

1. **Glucose sign**: Equal R/S count in multi-center sugars → `_cip_sign` returns 0.0
2. **Axial chirality**: SMILES encoding non-standardized; relies on 3D geometry
3. **Planar chirality**: Pure geometry approach; no direct RDKit API
4. **ML models**: Require training data for real predictions (demo uses synthetic labels)
5. **Chirality transfer**: Framework only — needs experimental reaction data
