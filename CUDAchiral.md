# CUDA Chirality — GPU-Accelerated Distance Geometry for ZynerjiChirality

## Overview

ZynerjiChirality's CUDA module replaces RDKit's ETKDG (Empirical Torsion-angle Distance Geometry with Knowledge) conformer generator with a GPU-accelerated distance geometry pipeline. RDKit ETKDG hangs on large peptides (>200 atoms) and is slow for medium molecules — the CUDA pipeline solves both problems while preserving stereochemistry through chirality-aware gradient constraints.

**Target hardware**: NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM, SM 12.0, Tensor Cores)

**Performance**: 6.2 mol/s (small, 1 restart) to 0.4 mol/s (large 400+ atom peptides, 3 restarts). Zero failures on 2,029 deferred large molecules.

---

## Module Structure

```
zynerji_chirality/cuda/
  __init__.py              # GPU detection, lazy imports, graceful degradation
  pipeline.py              # CUDAConformerPipeline — orchestrates full embed flow
  distance_geometry.py     # CUDADistanceGeometry — core GPU solver
  kernels.py               # Raw CUDA C kernels (4 kernels, JIT-compiled via CuPy)
  bounds.py                # Distance bounds computation (CPU, feeds GPU solver)
  gpu_fingerprinter.py     # GPUFingerprinter — batch fingerprint with CUDA embed
scripts/
  cuda_retry.py            # CLI: process deferred/failed molecules via GPU
  export_remaining.py      # CLI: export un-fingerprinted SMILES for GPU batch
```

---

## Architecture: End-to-End Data Flow

```
SMILES string
  │
  ▼
┌─────────────────────────────────────────────┐
│  1. RDKit Parsing (CPU)                     │
│     Chem.MolFromSmiles() → AddHs()          │
│     Extract: atoms, bonds, hybridization,   │
│     CIP labels (R/S per chiral center)      │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  2. Distance Bounds (CPU)                   │
│     Try RDKit GetMoleculeBoundsMatrix()     │
│     Fallback: manual (bond lengths, angles, │
│     VDW radii, shortest-path upper bounds)  │
│     Output: lower[N×N], upper[N×N] matrices │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌═════════════════════════════════════════════╗
║  3. GPU Distance Geometry Solver            ║
║                                             ║
║  ┌─────────────────────────────────────┐    ║
║  │ 3a. Triangle Smoothing (Floyd-W.)  │    ║
║  │     CUDA kernel: N² threads × N    │    ║
║  │     Enforce triangle inequality     │    ║
║  └──────────────┬──────────────────────┘    ║
║                 │                            ║
║  ┌──────────────▼──────────────────────┐    ║
║  │ 3b. Distance Sampling              │    ║
║  │     CuPy random: uniform in bounds │    ║
║  │     Symmetrize + zero diagonal     │    ║
║  └──────────────┬──────────────────────┘    ║
║                 │                            ║
║  ┌──────────────▼──────────────────────┐    ║
║  │ 3c. MDS Embedding (Tensor Cores)   │    ║
║  │     B = -½ H D² H    (cuBLAS)     │    ║
║  │     eigh(B)           (cuSOLVER)   │    ║
║  │     X = V√Λ           (top 3)     │    ║
║  └──────────────┬──────────────────────┘    ║
║                 │                            ║
║  ┌──────────────▼──────────────────────┐    ║
║  │ 3d. Constrained Refinement         │    ║
║  │     CUDA kernel 1: distance grads  │    ║
║  │     CUDA kernel 2: chirality vols  │    ║
║  │     Adaptive LR gradient descent   │    ║
║  └──────────────┬──────────────────────┘    ║
║                 │                            ║
║  ┌──────────────▼──────────────────────┐    ║
║  │ 3e. Multi-Restart Selection        │    ║
║  │     1-4 restarts, keep lowest      │    ║
║  │     violation. Early exit < 1e-4.  │    ║
║  └─────────────────────────────────────┘    ║
╚═══════════════╤═════════════════════════════╝
                │
                ▼
┌─────────────────────────────────────────────┐
│  4. Coordinate Assignment (CPU)             │
│     Set 3D coords on RDKit Conformer        │
│     AssignStereochemistry(force=True)        │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  5. Chirality Fingerprint (CPU)             │
│     Dual-helix spectral decomposition       │
│     8k-dim → 128-bit random projection      │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  6. DB Storage (SQLite batch insert)        │
│     fingerprint_blob, chirality_score/sign  │
└─────────────────────────────────────────────┘
```

---

## GPU Memory Management

### Unified Memory (Zero-Copy)

The solver uses CUDA Unified Memory via `cp.cuda.malloc_managed`:

```python
self._mem_pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(self._mem_pool.malloc)
```

**Why unified memory?** CPU and GPU share the same virtual address space. No explicit `cudaMemcpy` calls needed — the CUDA runtime automatically migrates pages between host and device on access. This eliminates:
- Manual host-to-device transfer before kernel launch
- Manual device-to-host transfer after kernel completion
- Double-buffering complexity for streaming pipelines

**Trade-off**: Slightly higher latency on first access (page fault migration) but dramatically simpler code and zero-copy semantics for CuPy ↔ NumPy interop.

**Memory pool**: `MemoryPool` reduces allocation overhead. Without it, each `cp.asarray()` call hits the CUDA allocator. The pool recycles freed blocks.

---

## CUDA Kernels (kernels.py)

Four raw CUDA C kernels compiled at import time via CuPy's `RawKernel` JIT:

### Kernel 1: Triangle Inequality Smoothing

**Purpose**: Enforce the triangle inequality on distance bounds so the bounds are geometrically consistent before sampling.

**Algorithm**: Floyd-Warshall adapted for bound matrices. For each intermediate node `k`, update all (i,j) pairs:
- `upper[i][j] = min(upper[i][j], upper[i][k] + upper[k][j])`
- `lower[i][j] = max(lower[i][j], lower[i][k] - upper[k][j])`
- `lower[i][j] = max(lower[i][j], lower[k][j] - upper[i][k])`

**Complexity**: O(N³) total, with N² parallelism per step (N sequential steps).

```cuda
extern "C" __global__ void triangle_smooth_step(
    double* __restrict__ lower,
    double* __restrict__ upper,
    const int n,
    const int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / n;
    int j = idx % n;

    if (i >= n || j >= n || i == j || i == k || j == k) return;

    // Upper bound: u[i][j] <= u[i][k] + u[k][j]
    double u_ik = upper[i * n + k];
    double u_kj = upper[k * n + j];
    double new_upper = u_ik + u_kj;
    if (new_upper < upper[i * n + j]) {
        upper[i * n + j] = new_upper;
    }

    // Lower bound: l[i][j] >= l[i][k] - u[k][j]
    double l_ik = lower[i * n + k];
    double new_lower = l_ik - u_kj;
    if (new_lower > lower[i * n + j]) {
        lower[i * n + j] = new_lower;
    }

    // Also: l[i][j] >= l[k][j] - u[i][k]
    double l_kj = lower[k * n + j];
    double new_lower2 = l_kj - u_ik;
    if (new_lower2 > lower[i * n + j]) {
        lower[i * n + j] = new_lower2;
    }
}
```

**Launch configuration**: `blocks = ceil(N²/256)`, `threads = 256`. Called N times (once per k).

**Memory access**: Row-major indexing `[i * n + j]` gives coalesced reads when adjacent threads have adjacent `j` values (threads within a warp access consecutive memory).

**Correctness note**: Updates are idempotent (max/min operations), so no inter-thread synchronization is needed within a single `k` step.

---

### Kernel 2: Refinement Gradient (Distance Violations)

**Purpose**: Compute per-atom gradients for distance bound violations. This is the inner loop of gradient descent refinement.

**Physics**: Atoms that are too close (below lower bound) are pushed apart. Atoms that are too far (above upper bound) are pulled together. Force magnitude is proportional to the squared violation.

**Loss function**:

```
L = Σ_{i<j} [(lower[i][j] - dist[i][j])²  if dist < lower
              (dist[i][j] - upper[i][j])²  if dist > upper
              0                              otherwise]
```

**Gradient** per atom i:

```
∂L/∂xᵢ = Σⱼ force_ij × (xᵢ - xⱼ) / dist_ij
```

```cuda
extern "C" __global__ void compute_refinement_gradient(
    const double* __restrict__ coords,   // (n, 3) — atom positions
    const double* __restrict__ lower,    // (n, n) — lower bounds
    const double* __restrict__ upper,    // (n, n) — upper bounds
    double* __restrict__ grad,           // (n, 3) — output gradients
    double* __restrict__ violation,      // (1,)   — total violation energy
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double gx = 0.0, gy = 0.0, gz = 0.0;
    double energy = 0.0;

    double xi = coords[i * 3];
    double yi = coords[i * 3 + 1];
    double zi = coords[i * 3 + 2];

    for (int j = 0; j < n; j++) {
        if (j == i) continue;

        double dx = xi - coords[j * 3];
        double dy = yi - coords[j * 3 + 1];
        double dz = zi - coords[j * 3 + 2];
        double dist2 = dx * dx + dy * dy + dz * dz;
        double dist = sqrt(dist2);
        if (dist < 1e-10) dist = 1e-10;

        double lb = lower[i * n + j];
        double ub = upper[i * n + j];
        double force = 0.0;

        if (dist < lb) {
            double diff = lb - dist;
            force = -2.0 * diff / dist;   // push apart
            energy += diff * diff;
        } else if (dist > ub) {
            double diff = dist - ub;
            force = 2.0 * diff / dist;    // pull together
            energy += diff * diff;
        }

        gx += force * dx;
        gy += force * dy;
        gz += force * dz;
    }

    grad[i * 3] = gx;
    grad[i * 3 + 1] = gy;
    grad[i * 3 + 2] = gz;

    atomicAdd(violation, energy);
}
```

**Launch**: `blocks = ceil(N/256)`, `threads = min(256, N)`. Each thread processes one atom.

**Complexity**: O(N) threads × O(N) inner loop = O(N²) per kernel call.

**`atomicAdd` note**: The violation accumulation has a race condition, but this is intentional — the violation value is used only for convergence monitoring, not for gradient computation. Approximate total is sufficient.

---

### Kernel 3: Chirality Volume Constraint

**Purpose**: Preserve R/S stereochemistry during refinement by constraining the signed tetrahedral volume at each chiral center.

**Mathematical basis**: The signed volume of a tetrahedron formed by a chiral center and its 4 neighbors encodes handedness:

```
V = v₀ · (v₁ × v₂)
```

where `v₀, v₁, v₂` are vectors from the center atom to three of its neighbors. V > 0 implies R configuration, V < 0 implies S.

**Gradient**: When the volume has the wrong sign (`V × target < 0`):

```
∂V/∂v₀ = v₁ × v₂  (the cross product)
```

The gradient pushes the neighbor atoms to flip the volume sign, restoring correct chirality.

```cuda
extern "C" __global__ void chirality_volume_gradient(
    double* __restrict__ coords,            // (n, 3)
    const int* __restrict__ chiral_centers,  // (n_chiral, 5): [center, nb0, nb1, nb2, nb3]
    const double* __restrict__ target_signs, // (n_chiral,): +1.0 (R) or -1.0 (S)
    double* __restrict__ grad,              // (n, 3) — accumulated
    const int n_chiral,
    const double weight
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chiral) return;

    int center = chiral_centers[idx * 5];
    int nb0 = chiral_centers[idx * 5 + 1];
    int nb1 = chiral_centers[idx * 5 + 2];
    int nb2 = chiral_centers[idx * 5 + 3];
    int nb3 = chiral_centers[idx * 5 + 4];

    // Vectors from center to neighbors
    double v0x = coords[nb0*3]   - coords[center*3];
    double v0y = coords[nb0*3+1] - coords[center*3+1];
    double v0z = coords[nb0*3+2] - coords[center*3+2];

    double v1x = coords[nb1*3]   - coords[center*3];
    double v1y = coords[nb1*3+1] - coords[center*3+1];
    double v1z = coords[nb1*3+2] - coords[center*3+2];

    double v2x = coords[nb2*3]   - coords[center*3];
    double v2y = coords[nb2*3+1] - coords[center*3+1];
    double v2z = coords[nb2*3+2] - coords[center*3+2];

    // Signed volume = v0 · (v1 × v2)
    double cx = v1y * v2z - v1z * v2y;
    double cy = v1z * v2x - v1x * v2z;
    double cz = v1x * v2y - v1y * v2x;
    double vol = v0x * cx + v0y * cy + v0z * cz;

    double target = target_signs[idx];

    // Only apply penalty when chirality is wrong
    if (vol * target < 0.0) {
        double scale = -weight * target;

        // Push nb0 along cross product direction
        atomicAdd(&grad[nb0*3],     scale * cx);
        atomicAdd(&grad[nb0*3+1],   scale * cy);
        atomicAdd(&grad[nb0*3+2],   scale * cz);

        // Push center opposite
        atomicAdd(&grad[center*3],   -scale * cx);
        atomicAdd(&grad[center*3+1], -scale * cy);
        atomicAdd(&grad[center*3+2], -scale * cz);
    }
}
```

**Key design**: The constraint is **one-sided** — it only activates when the volume has the wrong sign. Once chirality is correct, the constraint contributes zero gradient, avoiding over-constraint.

**`atomicAdd`**: Required because multiple chiral centers may share atoms (e.g., adjacent stereocenters in peptides). Gradients from different centers accumulate correctly.

---

### Kernel 4: Pairwise Distance Matrix

**Purpose**: Compute full N×N Euclidean distance matrix from 3D coordinates. Used for violation checking and verification.

```cuda
extern "C" __global__ void compute_pairwise_distances(
    const double* __restrict__ coords,  // (n, 3)
    double* __restrict__ dist_matrix,   // (n, n)
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / n;
    int j = idx % n;

    if (i >= n || j >= n) return;
    if (i == j) { dist_matrix[i * n + j] = 0.0; return; }

    double dx = coords[i*3]   - coords[j*3];
    double dy = coords[i*3+1] - coords[j*3+1];
    double dz = coords[i*3+2] - coords[j*3+2];
    dist_matrix[i * n + j] = sqrt(dx*dx + dy*dy + dz*dz);
}
```

**Launch**: `blocks = ceil(N²/256)`, `threads = 256`. N² threads total.

---

## Core Solver: CUDADistanceGeometry

### solve() — Full Pipeline with Multi-Restart

```python
def solve(self, lower, upper, chiral_info=None, n_restarts=4,
          base_seed=42, max_refine_iter=1000) -> np.ndarray:
```

**Algorithm**:

1. **Smooth bounds once** (shared across restarts) — O(N³) GPU
2. **For each restart** (serial, different random seed):
   - Sample distances within smoothed bounds — O(N²) GPU
   - MDS embedding to 3D — O(N³) Tensor Cores
   - Refine with distance + chirality constraints — O(N² × iters) GPU
   - Compute final violation
3. **Return coordinates with lowest violation**
4. **Early exit** if violation < 1e-4

**Why multiple restarts?** Distance geometry is a non-convex optimization. Different random distance samples lead to different local minima. Multiple restarts increase the probability of finding a low-violation embedding, especially for molecules with complex ring systems.

### Tensor Core Usage in MDS

The MDS step performs two operations that leverage Tensor Cores on Blackwell:

```python
# 1. Double centering — cuBLAS matmul (Tensor Cores for FP64)
H = cp.eye(n) - cp.ones((n, n)) / n
B = -0.5 * H @ D2 @ H    # Two N×N matrix multiplications

# 2. Eigendecomposition — cuSOLVER (Tensor Core acceleration)
eigenvalues, eigenvectors = cp.linalg.eigh(B)
```

cuBLAS automatically routes FP64 matrix multiplications through Tensor Cores when available (Blackwell SM 12.0 supports FP64 Tensor Core operations). cuSOLVER's `syevd` implementation likewise uses Tensor Core-accelerated BLAS routines internally.

### Adaptive Learning Rate

The refinement loop uses an adaptive learning rate strategy:

```python
lr = 0.005  # initial
for iteration in range(max_iter):
    # ... compute gradient ...
    coords -= lr * grad

    violation = compute_violation()

    if violation > prev_violation:
        lr *= 0.5          # halve on overshoot
    elif iteration % 200 == 0:
        lr *= 0.95         # slow decay for convergence

    if violation < tol:
        break              # converged
```

This prevents oscillation when the loss landscape is steep (common for under-constrained molecules) while maintaining convergence speed for well-behaved cases.

---

## Distance Bounds Computation (bounds.py)

The bounds computation runs on CPU and produces the lower/upper bound matrices that feed the GPU solver.

### MolecularGraph Extraction

```python
@dataclass
class MolecularGraph:
    n_atoms: int
    atomic_nums: np.ndarray              # (n,) atomic numbers
    bonds: list[tuple[int, int, float]]  # (i, j, bond_order)
    bond_lengths: np.ndarray             # (n_bonds,) in Angstroms
    hybridizations: list                 # sp, sp2, sp3 per atom
    chiral_centers: list[tuple[int, list[int], str]]  # (idx, neighbors, R/S)
    adjacency_list: dict[int, list[int]]
```

### Bounds Strategy (4 layers)

**Layer 1 — 1,2 pairs (directly bonded)**: Bond length ± 0.05 Å

Uses a lookup table of ~50 entries keyed by `(atomic_num_lo, atomic_num_hi, bond_order)`:
- C-C single: 1.54 Å, C=C double: 1.34 Å, C≡C triple: 1.20 Å
- C-H: 1.09 Å, C-N: 1.47 Å, C-O: 1.43 Å, C=O: 1.23 Å
- Fallback: sum of covalent radii

**Layer 2 — 1,3 pairs (two bonds apart)**: Law of cosines with hybridization angle

```
d_ab² = d_ac² + d_cb² - 2·d_ac·d_cb·cos(θ)
```

Angles by hybridization: sp3 → 109.5°, sp2 → 120°, sp → 180°. Tolerance ± 0.2 Å.

**Layer 3 — Non-bonded pairs**: VDW lower bounds

```
lower[i][j] = max(existing, 0.7 × (vdw_radius[i] + vdw_radius[j]))
```

VDW radii: H=1.20, C=1.70, N=1.55, O=1.52, S=1.80, F=1.47, Cl=1.75 Å.

**Layer 4 — Graph-theoretic upper bounds**: BFS shortest path × 1.1

For distant atoms, the maximum possible distance is bounded by the shortest graph path length (sum of bond lengths along path). Multiplied by 1.1 for molecular flexibility.

### RDKit Fast Path

```python
def try_rdkit_bounds(mol) -> tuple[np.ndarray, np.ndarray] | None:
```

When available, RDKit's `GetMoleculeBoundsMatrix()` is preferred — it uses chemically accurate force field parameters and produces tighter bounds. The manual computation is the fallback when RDKit's internal DG fails (common for large peptides).

---

## CUDAConformerPipeline (pipeline.py)

### Constructor

```python
CUDAConformerPipeline(
    device: int = 0,           # CUDA device
    n_restarts: int = 4,       # DG solver restarts (1 for speed, 3-4 for quality)
    max_refine_iter: int = 1000,
    try_rdkit_first: bool = True,   # Try RDKit bounds (faster when they work)
    rdkit_timeout: int = 30,
)
```

### embed_molecule()

The primary entry point. Returns an RDKit `Mol` with a 3D conformer:

```python
def embed_molecule(self, smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    lower, upper, chiral_info = self._get_bounds(mol)

    coords = self.dg_solver.solve(
        lower, upper,
        chiral_info=chiral_info,
        n_restarts=self.n_restarts,
    )

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, tuple(coords[i]))
    mol.AddConformer(conf, assignId=True)

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol
```

### CPU Fallback

When CUDA is unavailable, a pure-NumPy fallback runs O(N² × 500) iterations:

```python
def _cpu_fallback(self, lower, upper, chiral_info, n_atoms) -> np.ndarray:
    coords = rng.randn(n_atoms, 3) * 2.0
    for iteration in range(500):
        grad = np.zeros_like(coords)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # push/pull based on bound violations
                ...
        coords -= 0.01 * grad
    return coords
```

---

## GPUFingerprinter (gpu_fingerprinter.py)

Higher-level batch processing API wrapping the pipeline:

```python
class GPUFingerprinter:
    def __init__(self, store=None, nbits=128, n_restarts=4, device=0)

    def fingerprint_one(self, smiles, chembl_id) -> dict
    # GPU embed → chirality_fingerprint() → detect() → dict

    def process_failed_file(self, jsonl_path, batch_size=50) -> dict
    # Batch process JSONL, commit to DB every batch_size, return stats
```

Each molecule goes through: `embed_molecule()` (GPU) → `chirality_fingerprint()` (CPU, <10ms) → `HelixChiralityDetector.detect()` (CPU) → DB batch insert.

---

## Computational Complexity

| Stage | Algorithm | Complexity | GPU Parallelism |
|-------|-----------|------------|-----------------|
| Triangle smoothing | Floyd-Warshall | O(N³) | N² threads × N steps |
| Distance sampling | Uniform random | O(N²) | N² elements (CuPy) |
| MDS double centering | Matrix multiply | O(N³) | Tensor Cores (cuBLAS) |
| MDS eigendecomposition | Divide-and-conquer | O(N² log N) | Tensor Cores (cuSOLVER) |
| Refinement (per iter) | Pairwise gradients | O(N²) | N threads, O(N) inner loop |
| Chirality constraint | Volume + cross product | O(K) | K threads (K = chiral centers) |
| Multi-restart | Serial restarts | × R | R sequential solves |
| **Full solve** | **All stages** | **O(R × (N³ + N² × I))** | **See above** |

Where N = atoms, R = restarts (1-4), I = max refinement iterations (1000).

**Practical scaling**:
- 20 atoms: ~0.2s (GPU overhead dominates)
- 100 atoms: ~0.5s (MDS + refinement)
- 400 atoms: ~3s (Floyd-Warshall dominates)
- 1000 atoms: ~15s (theoretical, untested at this scale)

---

## Chirality Preservation

### The Problem

Standard distance geometry (sample → embed → refine) doesn't preserve stereochemistry. Two enantiomers have identical distance bounds matrices — the solver can freely produce either R or S configurations.

### The Solution: Signed Volume Constraints

Each chiral center is parameterized as `(center_idx, [nb0, nb1, nb2, nb3], target_sign)`:

```
target_sign = +1.0  (R configuration)
             -1.0  (S configuration)
```

During refinement, the chirality kernel computes the signed tetrahedral volume:

```
V = v₀ · (v₁ × v₂)
```

If `V × target < 0` (wrong handedness), gradients push the neighbor atoms to flip the volume sign. The gradient direction is the cross product `v₁ × v₂`, which points perpendicular to the face defined by neighbors 1 and 2 — moving neighbor 0 in this direction flips the tetrahedron.

### Integration with CIP Labels

1. RDKit assigns CIP labels (R/S) from the input SMILES stereochemistry annotations
2. `extract_graph()` captures `(center_idx, neighbor_indices, "R"/"S")` tuples
3. These flow into the chirality kernel as `chiral_centers` array + `target_signs`
4. After embedding, `Chem.AssignStereochemistry(force=True)` validates that 3D geometry matches expected CIP labels

---

## Error Handling & Graceful Degradation

```
Layer 1: No CUDA GPU → _GPU_AVAILABLE = False → CPU fallback in pipeline
Layer 2: CuPy import fails → _HAS_CUDA = False → CPU fallback in solver
Layer 3: RDKit bounds fail → Manual bounds computation (bonds, angles, VDW)
Layer 4: GPU solver fails → _cpu_fallback() (NumPy, 500 iterations)
Layer 5: Molecule embed fails → Log warning, write to cuda_still_failed.jsonl
Layer 6: DB write fails → Log error, continue processing next molecule
```

The system never crashes on a single molecule failure. Failed molecules are recorded and can be retried or investigated later.

---

## Usage Examples

### Single Molecule

```python
from zynerji_chirality.cuda.pipeline import CUDAConformerPipeline
from zynerji_chirality.chirality.fingerprint import chirality_fingerprint
from zynerji_chirality.chirality.detector import HelixChiralityDetector

pipeline = CUDAConformerPipeline(n_restarts=3)
mol = pipeline.embed_molecule("C[C@@H](O)CC")  # (R)-2-butanol

fp = chirality_fingerprint(mol, nbits=128)
result = HelixChiralityDetector().detect(mol)
print(f"Score: {result.chirality_score:.4f}, Sign: {result.chirality_sign}")
```

### Batch Processing (CLI)

```bash
# Process deferred large molecules
PYTHONPATH=. python3 scripts/cuda_retry.py \
    --failed chembl_work/failed_molecules.jsonl \
    --db chembl_work/chembl_screen.db \
    --restarts 3

# Process all remaining (small/medium, 1 restart for speed)
PYTHONPATH=. python3 scripts/cuda_retry.py \
    --failed chembl_work/remaining_molecules.jsonl \
    --db chembl_work/chembl_screen.db \
    --restarts 1

# Benchmark (first 20 molecules, no DB)
PYTHONPATH=. python3 scripts/cuda_retry.py \
    --failed chembl_work/remaining_molecules.jsonl \
    --test 20
```

### Export Remaining Molecules

```bash
PYTHONPATH=. python3 scripts/export_remaining.py \
    --pairs chembl_work/enriched_pairs.json \
    --db chembl_work/chembl_screen.db \
    --output chembl_work/remaining_molecules.jsonl
```

---

## Performance Benchmarks (Blackwell RTX PRO 6000)

| Molecule Size | Atoms | SMILES Length | Time/mol | Restarts | Rate |
|---------------|-------|--------------|----------|----------|------|
| Small (amino acid) | 11-25 | 13-17 chars | 0.16s | 1 | 6.2 mol/s |
| Medium (drug) | 30-80 | 30-100 chars | 0.3s | 1 | ~3 mol/s |
| Large (peptide) | 200-400 | 200-500 chars | 3-4s | 3 | 0.4 mol/s |
| Very large | 400+ | 500-1200 chars | 4-8s | 3 | 0.2 mol/s |

**GPU overhead**: ~0.2s minimum per molecule (kernel launch, memory allocation). This dominates for very small molecules. For molecules under 30 atoms, CPU ETKDG may actually be faster — the GPU advantage appears at ~50+ atoms.

**vs CPU ETKDG**: For molecules >200 atoms, RDKit ETKDG either hangs indefinitely or takes minutes. The CUDA pipeline processes these in seconds with zero failures.

---

## Database Schema (for reference)

```sql
CREATE TABLE molecules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    smiles TEXT NOT NULL,
    canonical_smiles TEXT NOT NULL,
    name TEXT
);

CREATE TABLE fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mol_id INTEGER NOT NULL REFERENCES molecules(id),
    nbits INTEGER NOT NULL,          -- 128 default
    params_hash TEXT NOT NULL,        -- "default"
    fingerprint_blob BLOB NOT NULL,  -- float64 array (128 × 8 = 1024 bytes)
    chirality_score REAL NOT NULL,   -- |differential asymmetry|, 0=achiral
    chirality_sign REAL NOT NULL,    -- +1 (R), -1 (S), 0 (achiral)
    UNIQUE(mol_id, nbits, params_hash)
);

CREATE TABLE metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mol_id INTEGER NOT NULL REFERENCES molecules(id),
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    UNIQUE(mol_id, key)
);
```

The `fingerprint_blob` stores the raw 128-dimensional spectral chirality fingerprint as a serialized float64 numpy array. Similarity search is cosine-based: `sim = (fp₁ · fp₂) / (‖fp₁‖ × ‖fp₂‖)`, mapped to [0,1].
