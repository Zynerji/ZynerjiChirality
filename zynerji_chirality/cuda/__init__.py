"""CUDA-accelerated distance geometry for large peptide fingerprinting.

Replaces RDKit's ETKDG conformer generation with a GPU-accelerated pipeline:
- Distance bounds computation from molecular graph
- Triangle inequality smoothing (O(N^3) → GPU parallelized)
- Classical MDS embedding (cuSOLVER eigendecomposition → Tensor Cores)
- Coordinate refinement with chirality constraints (custom CUDA kernels)
- Zero-copy unified memory management

Requires: cupy-cuda12x, rdkit, numpy, scipy
Target: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM, CUDA 12.8)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_GPU_AVAILABLE = False
_GPU_INFO = {}

try:
    import cupy as cp

    device = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    _GPU_AVAILABLE = True
    _GPU_INFO = {
        "name": props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
        "total_memory_gb": props["totalGlobalMem"] / (1024**3),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "multiprocessors": props["multiProcessorCount"],
        "cuda_version": cp.cuda.runtime.runtimeGetVersion(),
    }
    logger.info("CUDA GPU detected: %s (%.1f GB, SM %s)",
                _GPU_INFO["name"], _GPU_INFO["total_memory_gb"],
                _GPU_INFO["compute_capability"])
except Exception as e:
    logger.info("No CUDA GPU available: %s (CPU fallback will be used)", e)


def gpu_available() -> bool:
    """Check if CUDA GPU is available."""
    return _GPU_AVAILABLE


def gpu_info() -> dict:
    """Get GPU device information."""
    return _GPU_INFO.copy()


# Lazy imports to avoid import errors when CUDA is not available
def get_pipeline():
    """Get CUDAConformerPipeline (lazy import)."""
    from zynerji_chirality.cuda.pipeline import CUDAConformerPipeline
    return CUDAConformerPipeline


def get_fingerprinter():
    """Get GPUFingerprinter (lazy import)."""
    from zynerji_chirality.cuda.gpu_fingerprinter import GPUFingerprinter
    return GPUFingerprinter
