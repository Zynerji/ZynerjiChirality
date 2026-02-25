# ZynerjiChirality — Project Checkpoint

## What This Project Does

Chirality detection via dual-helix spectral graph analysis. Detects whether a molecule is chiral (R/S) using spectral asymmetry between two phase-modulated graph Laplacians (cos/sin helices with golden-ratio coupling).

**Key insight**: Standard graph Laplacian eigenvalues are identical for enantiomers. The dual-helix Laplacian breaks this symmetry via phase modulation that depends on node ordering. CIP-priority canonical ordering with chirality-dependent cyclic shifts encodes handedness into the ordering, producing different spectral responses for R vs S.

## Current Status (v0.4.0, 2026-02-24)

**All core benchmarks passing — 100%:**
- Amino acids: 19/19 (100%) with opposite signs
- Drug pairs: 12/12 (100%) with opposite signs
- Achiral rejection: 16/16 (100%)
- Meso compounds: 2/2 correctly achiral (FIXED in v0.2.0)
- Tests: 134 passing (108 core + 26 ChEMBL pipeline)
- Pushed to GitHub: `Zynerji/ZynerjiChirality`

**Expanded benchmarks:**
- Sugars: 4/4 detected, 3/4 opposite signs (glucose sign issue due to equal R/S count)
- Natural products: 3/3 (menthol, camphor, carvone)
- Steroids: 2/2 (cholesterol, testosterone)
- Tough achiral: 3/4 (norbornane embedding issue)

### v0.4.0 — Materials Science + Catalysts Expansion (2026-02-24)

**Materials Science Module** (`zynerji_chirality/materials/`):
- `ChiralMaterialDetector` — wraps core engine for material-level analysis
- Supports SMILES input (molecular materials) and raw adjacency matrices (crystals/metamaterials)
- Properties: circular dichroism (CD activity + sign), CISS score (spin selectivity), optical rotation, band gap shift
- Crystal utilities: CIF parser (pure Python, no pymatgen), perovskite ABX3 graph generator, polymer repeat graph
- Visualization: simulated CD spectrum, CISS polar diagram, material comparison bar charts
- Dashboard: "Materials" tab with SMILES input → full material analysis
- CLI: `scripts/run_materials.py` (--smiles, --cif, --perovskite, --compare)

**Catalysts Module** (`zynerji_chirality/catalysts/`):
- `EEPredictor` — predicts enantiomeric excess (ee%) from catalyst+substrate chirality fingerprints
  - Combined fingerprint: catalyst FP + substrate FP + interaction (product + difference) = 4*nbits
  - Subclasses `BaseChiralPredictor` (sklearn RF/GB)
- `LigandScreener` — ranks ligands by predicted ee (ML mode) or chirality score (heuristic mode)
- `CatalyticReactionAnalyzer` — extends `ReactionStereoAnalyzer` with catalyst chirality + face selectivity
- BINAP hydrogenation benchmark: 10 known Ru-BINAP catalyzed reactions with published ee values
- Dashboard: "Catalysts" tab with catalyst+substrate SMILES → predicted ee
- CLI: `scripts/run_catalysts.py` (--predict, --screen, --benchmark, --rank-ligands)

**Tests**: 24+ new tests (12+ materials, 12+ catalysts), total 158+
**No new dependencies** — all built on numpy/scipy/rdkit/sklearn

### v0.3.0 — ChEMBL Enantiomer Screening Pipeline (2026-02-22)

**Pipeline**: Download ChEMBL (~2.2M compounds) -> discover enantiomer pairs -> enrich with bioactivity -> fingerprint -> screen for chirality-sensitive drug targets.

**Discovery results** (Stage 1, complete):
- 235,434 enantiomer/diastereomer pairs (37,172 enantiomers + 198,262 diastereomers)
- 155,815 unique ChEMBL molecule IDs
- pairs.json: 470MB, stored at `/opt/chirality/chembl_work/pairs.json`

**Enrichment** (Stage 2, COMPLETE):
- 235,434 pairs processed, 83,801 with activity data, 68,127 with differential activity (>3x fold change on same target)
- enriched_pairs.json saved at `/opt/chirality/chembl_work/enriched_pairs.json`

**Fingerprinting** (Stage 3, IN PROGRESS as of 2026-02-23 15:21 ET):
- **Progress**: 11,000 / 155,815 molecules (7.1%), ~2 mol/s, 99.3% success rate (10,921 success, 79 fail)
- **ETA**: ~20 hours remaining (~Feb 24 ~11:00 ET)
- Running in tmux session "chirality" with `--resume` (skips completed Stage 1 + 2)
- Checkpoint: `fp_checkpoint.json` in chembl_work/
- DB: `chembl_screen.db` (10,965 molecules fingerprinted so far)

**Bugs fixed 2026-02-23** (Stage 3):
1. **Pickle error**: `batch_fingerprint.<locals>._compute_one` not picklable → moved to module-level `_compute_one_fp`
2. **99% failure rate**: Large peptides fail ETKDG embedding → added `useRandomCoords=True` fallback in `mol_graph.py`
3. **Slow fingerprinting**: MMFF optimization on random-coords conformers → skip MMFF when `used_random_coords=True`
4. **Serial detect bottleneck**: Combined fingerprint + detect into single parallel worker `_detect_and_fingerprint_one`
5. **Batch size**: Increased from 200 to 1000 for better parallelism

**Files changed**:
- `zynerji_chirality/chirality/fingerprint.py` — module-level `_compute_one_fp`, tuple args for Pool.map
- `zynerji_chirality/core/mol_graph.py` — `useRandomCoords=True` fallback, skip MMFF on fallback path
- `zynerji_chirality/chembl/fingerprinter.py` — combined `_detect_and_fingerprint_one` worker, batch_size=1000

**After Stage 3 completes:**
1. Stage 4: Screen — activity gaps, differential ranking, generate `screening_report.txt` and `screening_hits.json`
2. Review results on dashboard and in report files

**What the results mean:**
- **With activity**: Pairs where at least one enantiomer has bioactivity data (IC50, Ki, EC50) in ChEMBL
- **Differential**: Both enantiomers tested on same target with >3x potency difference — chirality matters for these drug-target interactions
- **Novel findings come from Stage 4**: Activity gaps (one tested, other not) + chirality fingerprint analysis to predict which untested enantiomers are worth investigating

**VM Deployment**: Running on 204.12.163.252 (user: ivhl)
- Dashboard: http://204.12.163.252/chirality/ (port 8082, nginx reverse proxy)
- Service: `chirality.service` (systemd, auto-restart)
- Pipeline data: `/opt/chirality/chembl_work/`
- DB: `/opt/chirality/chembl_work/chembl_screen.db`
- Logs: `journalctl -u chirality -f`
- Pipeline runs in tmux session "chirality": `tmux attach -t chirality`
- BiCameral services (port 8081) are NOT to be disturbed
- ZynerjiTrader services were stopped/disabled to free resources
- **Disk**: 8.7G total, 74% used (2.3G free). Cache cleanup after Stage 2 will free ~894MB.

**Pipeline stages** (scripts/chembl_pipeline.py):
1. **Discover**: Download ChEMBL SDF via chembl-downloader, filter chiral molecules, group by stereo-stripped SMILES, classify enantiomer/diastereomer pairs
2. **Enrich**: Fetch bioactivity (IC50, Ki, EC50) via chembl_webresource_client API — **incremental with checkpoint** (500 pairs/chunk)
3. **Fingerprint**: Chirality-aware spectral fingerprints with multiprocessing + batch DB inserts
4. **Screen**: Activity gaps (one tested, other not), differential activity (>3x fold-change), query similarity

**Dashboard** (FastAPI, dark theme, 15s auto-refresh):
- Enrichment progress bar in header (gradient, auto-updates)
- 4 tabs: Screening Hits, Enantiomer Pairs, Differential Activity, Search
- REST API: /health, /pairs, /hits, /enriched, /stats, /search
- **Performance**: JSON cache with mtime invalidation, /stats uses lightweight `enrich_progress.json` instead of loading full enriched_pairs.json (was 11.6s, now <40ms)
- No-cache headers via middleware + nginx (`Cache-Control: no-store`)

**Bugs fixed (2026-02-22)**:
- Enrichment was non-incremental (all-or-nothing for 155K molecules, ~8hr with no saves)
- Dashboard /stats loaded 470MB pairs.json + 64MB+ enriched_pairs.json on every 15s refresh (11.6s response → dashboard never updated)
- Browser aggressively cached dashboard HTML (added no-cache middleware + nginx headers)
- Added `_cleanup_chembl_cache()` to pipeline — removes ~/.data/chembl/ (~894MB) after Stage 2 completes

**Session 2026-02-23**:
- Enrichment completed 100% (83,801 with activity, 68,127 differential)
- Stage 3 fingerprinting in progress: 11K/155K (7%), ~2 mol/s, 99.3% success rate
- Fixed pickle bug, 99% failure rate (useRandomCoords fallback), slow MMFF, serial detect
- Commits pushed: ff35808 (incremental enrichment + dashboard perf), 0c75513 (ChEMBL cache cleanup)

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
  materials/             # v0.4.0 — Chiral materials science
    detector.py          # ChiralMaterialDetector (CD, CISS, rotation, adjacency input)
    properties.py        # CD activity, CISS score, optical rotation, band gap shift
    crystal.py           # CIF parser, perovskite graph, polymer repeat graph
    visualization.py     # CD spectrum, CISS diagram, material comparison plots
  catalysts/             # v0.4.0 — Chiral catalysts / enantioselectivity
    ee_predictor.py      # EEPredictor (ee% from catalyst+substrate fingerprints)
    ligand_screen.py     # LigandScreener (ML or heuristic ligand ranking)
    reaction_sim.py      # CatalyticReactionAnalyzer (face selectivity + catalyst chirality)
    benchmarks.py        # BINAP hydrogenation benchmark (10 known reactions)
  benchmarks/
    amino_acids.py       # 19 L/D amino acid pairs + glycine
    rs_pairs.py          # 12 R/S drug pairs
    known_molecules.py   # 16 achiral + 2 meso compounds
    expanded.py          # Sugars, natural products, steroids, tough achiral
    axial_chirality.py   # Biaryl atropisomer benchmarks
    planar_chirality.py  # Paracyclophane benchmarks
  chembl/
    download.py          # ChEMBLDownloader + ChiralMolecule dataclass
    pairs.py             # EnantiomerPairFinder (stereo-stripped grouping + CIP verification)
    activity.py          # ActivityEnricher (ChEMBL bioactivity API)
    fingerprinter.py     # PairFingerprinter (parallel compute + batch DB insert)
    screen.py            # EnantiomerScreen (activity gaps, differential, query)
  dashboard/
    app.py               # FastAPI dashboard (dark theme, 6 tabs, auto-refresh)
  db/
    store.py             # SQLite FingerprintStore + batch_add_fast + vectorized search
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
tests/                   # 158+ tests across 13 files
scripts/
  demo.py                # Quick demo with all features
  run_benchmarks.py      # Full benchmark suite (5 categories)
  run_materials.py       # Materials chirality CLI (SMILES, CIF, perovskite, compare)
  run_catalysts.py       # Catalyst screening CLI (predict ee, screen ligands, benchmark)
  ingest.py              # CLI for batch SMILES ingestion
  compare_fingerprints.py # ZynerjiChirality vs ECFP4 vs MACCS comparison
  train_activity_model.py # Demo activity model training
  chembl_pipeline.py     # ChEMBL enantiomer screening pipeline (4-stage CLI)
  run_dashboard.py       # Standalone dashboard launcher
deploy/
  chirality.service      # systemd service (port 8082, User=ivhl)
  nginx_chirality.conf   # nginx location block (/chirality/ -> 8082)
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
- **FingerprintStore**: SQLite-backed with `add()`, `get()`, `batch_add()`, `batch_add_fast()`, `search_similar()`
- **Batch inserts**: `batch_add_fast()` wraps all inserts in single transaction (10-50x speedup)
- **Vectorized search**: `_load_fingerprint_matrix()` caches numpy matrix for cosine k-NN
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
- scikit-learn (optional, `pip install zynerji-chirality[ml]`)
- chembl-downloader, chembl_webresource_client (optional, `pip install zynerji-chirality[chembl]`)
- fastapi, uvicorn (optional, `pip install zynerji-chirality[dashboard]`)
- All optional: `pip install zynerji-chirality[all]`

## Vast.ai GPU VM (Blackwell)

**GPU**: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB VRAM, SM 12.0)

**Connection:**
- SSH alias: `rtx6000` (configured in `~/.ssh/config`)
- Host: `38.79.155.162`, Port: `61938`
- User: `root`
- SSH key: `~/.ssh/id_ed25519`

```bash
# Connect
ssh rtx6000
# or explicit:
ssh -i ~/.ssh/id_ed25519 -p 61938 root@38.79.155.162

# Copy files TO VM
scp -i ~/.ssh/id_ed25519 -P 61938 local_file root@38.79.155.162:/opt/chirality/

# Copy files FROM VM
scp -i ~/.ssh/id_ed25519 -P 61938 root@38.79.155.162:/opt/chirality/chembl_work/chembl_screen.db .
```

**Data paths on VM:**
- Code: `/opt/chirality/` (ZynerjiChirality repo)
- Pipeline data: `/opt/chirality/chembl_work/`
- DB: `/opt/chirality/chembl_work/chembl_screen.db`
- CUDA module: `/opt/chirality/zynerji_chirality/cuda/`

**tmux sessions:**
- `gpu_bulk` — GPU fingerprinting of remaining molecules (cuda_retry.py)

**CUDA kernels deployed:**
- Distance geometry (triangle smoothing, refinement, chirality volume constraint, pairwise distances)
- Used for 3D conformer generation replacing RDKit ETKDG (~12.5x speedup: 5-7 mol/s vs 0.4 mol/s CPU)

**Important:** This is a rented VM — terminate after all chirality work is done to save budget.

## Known Limitations

1. **Glucose sign**: Equal R/S count in multi-center sugars → `_cip_sign` returns 0.0
2. **Axial chirality**: SMILES encoding non-standardized; relies on 3D geometry
3. **Planar chirality**: Pure geometry approach; no direct RDKit API
4. **ML models**: Require training data for real predictions (demo uses synthetic labels)
5. **Chirality transfer**: Framework only — needs experimental reaction data
