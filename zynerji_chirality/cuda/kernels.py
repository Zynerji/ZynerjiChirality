"""CUDA kernel source strings for JIT compilation via CuPy RawKernel.

Kernels:
- triangle_smooth_step: One Floyd-Warshall step for bounds smoothing
- compute_refinement_gradient: Per-atom gradient from distance violations
- chirality_volume_gradient: Chirality constraint (signed tetrahedral volume)
- compute_pairwise_distances: Batched pairwise distance computation
"""

# Triangle inequality smoothing — one step of Floyd-Warshall
# Called N times (once per intermediate node k)
# Each thread handles one (i,j) pair
TRIANGLE_SMOOTH_KERNEL = r"""
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

    // Upper bound smoothing: upper[i][j] <= upper[i][k] + upper[k][j]
    double u_ik = upper[i * n + k];
    double u_kj = upper[k * n + j];
    double new_upper = u_ik + u_kj;
    if (new_upper < upper[i * n + j]) {
        upper[i * n + j] = new_upper;
    }

    // Lower bound smoothing: lower[i][j] >= lower[i][k] - upper[k][j]
    double l_ik = lower[i * n + k];
    double new_lower = l_ik - u_kj;
    if (new_lower > lower[i * n + j]) {
        lower[i * n + j] = new_lower;
    }

    // Also try: lower[i][j] >= lower[k][j] - upper[i][k]
    double l_kj = lower[k * n + j];
    double new_lower2 = l_kj - u_ik;
    if (new_lower2 > lower[i * n + j]) {
        lower[i * n + j] = new_lower2;
    }
}
"""

# Refinement gradient: per-atom force from distance bound violations
# Each thread computes the gradient for one atom
REFINEMENT_GRADIENT_KERNEL = r"""
extern "C" __global__ void compute_refinement_gradient(
    const double* __restrict__ coords,   // (n, 3)
    const double* __restrict__ lower,    // (n, n)
    const double* __restrict__ upper,    // (n, n)
    double* __restrict__ grad,           // (n, 3)
    double* __restrict__ violation,      // (1,) total violation energy
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
            // Below lower bound — push atoms apart
            double diff = lb - dist;
            force = -2.0 * diff / dist;
            energy += diff * diff;
        } else if (dist > ub) {
            // Above upper bound — pull atoms together
            double diff = dist - ub;
            force = 2.0 * diff / dist;
            energy += diff * diff;
        }

        gx += force * dx;
        gy += force * dy;
        gz += force * dz;
    }

    grad[i * 3] = gx;
    grad[i * 3 + 1] = gy;
    grad[i * 3 + 2] = gz;

    // Atomic add to total violation (approximate — race condition OK for monitoring)
    atomicAdd(violation, energy);
}
"""

# Chirality volume constraint gradient
# Maintains correct R/S stereochemistry via signed tetrahedral volume
CHIRALITY_VOLUME_KERNEL = r"""
extern "C" __global__ void chirality_volume_gradient(
    double* __restrict__ coords,            // (n, 3) atom positions
    const int* __restrict__ chiral_centers,  // (n_chiral, 5): [center, nb0, nb1, nb2, nb3]
    const double* __restrict__ target_signs, // (n_chiral,): +1 for R, -1 for S
    double* __restrict__ grad,              // (n, 3) output gradients (accumulated)
    const int n_chiral,
    const double weight                     // constraint weight
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chiral) return;

    int center = chiral_centers[idx * 5];
    int nb0 = chiral_centers[idx * 5 + 1];
    int nb1 = chiral_centers[idx * 5 + 2];
    int nb2 = chiral_centers[idx * 5 + 3];
    int nb3 = chiral_centers[idx * 5 + 4];

    // Vectors from center to neighbors
    double v0x = coords[nb0*3] - coords[center*3];
    double v0y = coords[nb0*3+1] - coords[center*3+1];
    double v0z = coords[nb0*3+2] - coords[center*3+2];

    double v1x = coords[nb1*3] - coords[center*3];
    double v1y = coords[nb1*3+1] - coords[center*3+1];
    double v1z = coords[nb1*3+2] - coords[center*3+2];

    double v2x = coords[nb2*3] - coords[center*3];
    double v2y = coords[nb2*3+1] - coords[center*3+1];
    double v2z = coords[nb2*3+2] - coords[center*3+2];

    // Signed volume = v0 . (v1 x v2)
    double cx = v1y * v2z - v1z * v2y;
    double cy = v1z * v2x - v1x * v2z;
    double cz = v1x * v2y - v1y * v2x;
    double vol = v0x * cx + v0y * cy + v0z * cz;

    // Target sign
    double target = target_signs[idx];

    // Penalty if volume has wrong sign
    if (vol * target < 0.0) {
        // Gradient of volume w.r.t. v0 = (v1 x v2) = (cx, cy, cz)
        double scale = -weight * target;

        // Push nb0 in direction of cross product
        atomicAdd(&grad[nb0*3],   scale * cx);
        atomicAdd(&grad[nb0*3+1], scale * cy);
        atomicAdd(&grad[nb0*3+2], scale * cz);

        // Push center opposite
        atomicAdd(&grad[center*3],   -scale * cx);
        atomicAdd(&grad[center*3+1], -scale * cy);
        atomicAdd(&grad[center*3+2], -scale * cz);
    }
}
"""

# Pairwise distance matrix computation (batched)
PAIRWISE_DISTANCES_KERNEL = r"""
extern "C" __global__ void compute_pairwise_distances(
    const double* __restrict__ coords,  // (n, 3)
    double* __restrict__ dist_matrix,   // (n, n)
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / n;
    int j = idx % n;

    if (i >= n || j >= n) return;
    if (i == j) {
        dist_matrix[i * n + j] = 0.0;
        return;
    }

    double dx = coords[i*3] - coords[j*3];
    double dy = coords[i*3+1] - coords[j*3+1];
    double dz = coords[i*3+2] - coords[j*3+2];
    dist_matrix[i * n + j] = sqrt(dx*dx + dy*dy + dz*dz);
}
"""
