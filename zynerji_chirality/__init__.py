"""ZynerjiChirality — Chirality detection via dual-helix spectral graph analysis."""

__version__ = "0.5.0"

# Core detection API
from zynerji_chirality.chirality.detector import (
    HelixChiralityDetector,
    ChiralityResult,
    EnantiomerComparison,
    EnsembleResult,
)

# Fingerprinting
from zynerji_chirality.chirality.fingerprint import (
    chirality_fingerprint,
    batch_fingerprint,
    fingerprint_similarity,
)

# Database
from zynerji_chirality.db.store import FingerprintStore

# Optional: Reactions (not needed for GPU pipeline)
try:
    from zynerji_chirality.reactions.reaction_graph import (
        ReactionStereoAnalyzer,
        ReactionStereoResult,
    )
    from zynerji_chirality.reactions.retro import RetroChiralityPlanner
except ImportError:
    pass

# Optional: Materials science
try:
    from zynerji_chirality.materials.detector import (
        ChiralMaterialDetector,
        MaterialChiralityResult,
    )
except ImportError:
    pass

# Optional: Catalysts
try:
    from zynerji_chirality.catalysts.ee_predictor import EEPredictor, EEPrediction
except ImportError:
    pass

# Optional: COD pipeline
try:
    from zynerji_chirality.cod.download import CODDownloader, CrystalStructure
except ImportError:
    pass

# Optional: ORD pipeline
try:
    from zynerji_chirality.ord.download import ORDDownloader, ORDReaction
except ImportError:
    pass

__all__ = [
    "HelixChiralityDetector",
    "ChiralityResult",
    "EnantiomerComparison",
    "EnsembleResult",
    "chirality_fingerprint",
    "batch_fingerprint",
    "fingerprint_similarity",
    "FingerprintStore",
]
