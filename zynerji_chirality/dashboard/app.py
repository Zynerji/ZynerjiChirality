"""FastAPI dashboard for ZynerjiChirality pipeline monitoring.

Provides real-time visibility into:
- Pipeline status and progress
- Discovered enantiomer pairs
- Screening hits and differential activity
- Fingerprint database statistics
- Molecule visualization
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from zynerji_chirality.db.store import FingerprintStore

logger = logging.getLogger(__name__)


def create_app(
    db_path: str = "chembl_screen.db",
    work_dir: str = "chembl_work",
    title: str = "ZynerjiChirality",
):
    """Create the FastAPI dashboard application.

    Parameters
    ----------
    db_path : str
        Path to the fingerprint SQLite database.
    work_dir : str
        Working directory with pipeline checkpoints/outputs.
    title : str
        Dashboard title.
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError:
        raise ImportError(
            "FastAPI required for dashboard. Install with: "
            "pip install zynerji-chirality[dashboard]"
        )

    from starlette.middleware.base import BaseHTTPMiddleware

    app = FastAPI(title=title)
    work = Path(work_dir)
    _start_time = time.time()

    class NoCacheMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            return response

    app.add_middleware(NoCacheMiddleware)

    # Cache for expensive JSON loads: {filename: (mtime, data)}
    _json_cache: dict[str, tuple[float, any]] = {}

    def _get_store():
        return FingerprintStore(db_path)

    def _load_json(filename: str) -> dict | list | None:
        path = work / filename
        if not path.exists():
            return None
        try:
            mtime = path.stat().st_mtime
            if filename in _json_cache and _json_cache[filename][0] == mtime:
                return _json_cache[filename][1]
            with open(path) as f:
                data = json.load(f)
            _json_cache[filename] = (mtime, data)
            return data
        except Exception:
            return None

    def _load_json_small(filename: str) -> dict | list | None:
        """Load small JSON files without caching (progress files, etc.)."""
        path = work / filename
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    @app.get("/health")
    def health():
        try:
            store = _get_store()
            count = store.count()
            store.close()
        except Exception:
            count = 0

        # Use lightweight progress files — avoid loading huge JSON on every health check
        enrich_progress = _load_json_small("enrich_progress.json")
        fp_ckpt = _load_json_small("fp_checkpoint.json")
        hits_data = _load_json_small("screening_hits.json")

        # Get pair count from progress or cached pairs
        pairs_enriched = 0
        pairs_total = 0
        if enrich_progress:
            pairs_enriched = enrich_progress.get("pairs_enriched", 0)
            pairs_total = enrich_progress.get("pairs_total", 0)

        # Only load pairs.json if we don't have a total from progress
        if pairs_total == 0:
            pairs_data = _load_json("pairs.json")
            pairs_total = len(pairs_data) if pairs_data else 0

        return {
            "status": "ok",
            "uptime_seconds": int(time.time() - _start_time),
            "db_path": db_path,
            "molecules_in_store": count,
            "pairs_discovered": pairs_total,
            "pairs_enriched": pairs_enriched,
            "screening_hits": len(hits_data) if hits_data else 0,
            "fingerprint_progress": fp_ckpt,
            "enrich_progress": enrich_progress,
        }

    @app.get("/pairs")
    def get_pairs(limit: int = 50, offset: int = 0, relationship: str = None):
        pairs_data = _load_json("pairs.json")
        if not pairs_data:
            return {"pairs": [], "total": 0}

        if relationship:
            pairs_data = [p for p in pairs_data if p.get("relationship") == relationship]

        total = len(pairs_data)
        page = pairs_data[offset:offset + limit]

        # Slim down for API response
        result = []
        for p in page:
            result.append({
                "connectivity": p.get("connectivity_smiles", ""),
                "mol_a_id": p.get("mol_a", {}).get("chembl_id", ""),
                "mol_a_smiles": p.get("mol_a", {}).get("canonical_smiles", ""),
                "mol_b_id": p.get("mol_b", {}).get("chembl_id", ""),
                "mol_b_smiles": p.get("mol_b", {}).get("canonical_smiles", ""),
                "relationship": p.get("relationship", ""),
                "n_centers_a": p.get("mol_a", {}).get("n_centers", 0),
                "n_centers_b": p.get("mol_b", {}).get("n_centers", 0),
            })

        return {"pairs": result, "total": total}

    @app.get("/hits")
    def get_hits(hit_type: str = None, limit: int = 50):
        hits_data = _load_json("screening_hits.json")
        if not hits_data:
            return {"hits": [], "total": 0}

        if hit_type:
            hits_data = [h for h in hits_data if h.get("hit_type") == hit_type]

        total = len(hits_data)
        return {"hits": hits_data[:limit], "total": total}

    @app.get("/enriched")
    def get_enriched(limit: int = 20, has_differential: bool = False):
        enriched_data = _load_json("enriched_pairs.json")
        if not enriched_data:
            return {"pairs": [], "total": 0}

        if has_differential:
            enriched_data = [
                e for e in enriched_data
                if e.get("differential_targets")
            ]

        total = len(enriched_data)
        result = []
        for e in enriched_data[:limit]:
            pair = e.get("pair", {})
            result.append({
                "mol_a_id": pair.get("mol_a", {}).get("chembl_id", ""),
                "mol_b_id": pair.get("mol_b", {}).get("chembl_id", ""),
                "mol_a_smiles": pair.get("mol_a", {}).get("canonical_smiles", ""),
                "mol_b_smiles": pair.get("mol_b", {}).get("canonical_smiles", ""),
                "relationship": pair.get("relationship", ""),
                "n_activities_a": len(e.get("activities_a", [])),
                "n_activities_b": len(e.get("activities_b", [])),
                "shared_targets": len(e.get("shared_targets", [])),
                "differential_targets": e.get("differential_targets", []),
                "gap_targets_a": len(e.get("gap_targets_a", [])),
                "gap_targets_b": len(e.get("gap_targets_b", [])),
            })

        return {"pairs": result, "total": total}

    @app.get("/stats")
    def get_stats():
        try:
            store = _get_store()
            count = store.count()
            store.close()
        except Exception:
            count = 0

        # Use cached pairs.json (only reloads when mtime changes)
        pairs_data = _load_json("pairs.json")
        hits_data = _load_json_small("screening_hits.json")

        # Use lightweight progress file for enrichment stats (not the huge enriched_pairs.json)
        enrich_progress = _load_json_small("enrich_progress.json")

        n_enantiomers = 0
        n_diastereomers = 0
        if pairs_data:
            for p in pairs_data:
                if p.get("relationship") == "enantiomer":
                    n_enantiomers += 1
                else:
                    n_diastereomers += 1

        # Get enrichment stats from progress file (fast) instead of parsing full JSON
        n_with_activity = 0
        n_with_differential = 0
        n_with_gaps = 0
        if enrich_progress:
            n_with_activity = enrich_progress.get("pairs_with_activity", 0)
            n_with_differential = enrich_progress.get("pairs_with_differential", 0)
            n_with_gaps = enrich_progress.get("pairs_with_gaps", 0)

        hit_types = {}
        if hits_data:
            for h in hits_data:
                t = h.get("hit_type", "unknown")
                hit_types[t] = hit_types.get(t, 0) + 1

        return {
            "molecules_fingerprinted": count,
            "total_pairs": len(pairs_data) if pairs_data else 0,
            "enantiomer_pairs": n_enantiomers,
            "diastereomer_pairs": n_diastereomers,
            "pairs_with_activity": n_with_activity,
            "pairs_with_differential": n_with_differential,
            "pairs_with_gaps": n_with_gaps,
            "total_hits": len(hits_data) if hits_data else 0,
            "hits_by_type": hit_types,
        }

    @app.get("/search")
    def search_molecule(smiles: str, k: int = 10, mode: str = "similar"):
        from zynerji_chirality.chirality.fingerprint import chirality_fingerprint

        try:
            fp = chirality_fingerprint(smiles)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to fingerprint: {e}"},
            )

        store = _get_store()
        results = store.search_similar(fp, k=k, mode=mode)
        store.close()

        return {
            "query": smiles,
            "mode": mode,
            "results": [
                {
                    "mol_id": r.mol_id,
                    "smiles": r.smiles,
                    "name": r.name,
                    "similarity": round(r.similarity, 4),
                    "chirality_score": round(r.chirality_score, 4),
                    "chirality_sign": r.chirality_sign,
                }
                for r in results
            ],
        }

    # ── Materials & Catalysts endpoints ──────────────────────────────────

    @app.post("/materials/analyze")
    def analyze_material(request_body: dict = None):
        """Analyze chirality of a material from SMILES."""
        if request_body is None:
            return JSONResponse(status_code=400, content={"error": "Request body required"})
        smiles = request_body.get("smiles", "")
        if not smiles:
            return JSONResponse(status_code=400, content={"error": "smiles field required"})

        try:
            from zynerji_chirality.materials.detector import ChiralMaterialDetector
            detector = ChiralMaterialDetector()
            result = detector.detect_material(smiles)
            return {
                "smiles": smiles,
                "is_chiral": result.is_chiral,
                "chirality_score": round(result.chirality_score, 6),
                "chirality_sign": result.chirality_sign,
                "cd_activity": round(result.cd_activity, 6),
                "cd_sign": result.cd_sign,
                "ciss_score": round(result.ciss_score, 4),
                "optical_rotation": result.optical_rotation,
                "material_class": result.material_class,
                "band_gap_shift": round(result.band_gap_shift, 6),
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    @app.post("/catalysts/predict-ee")
    def predict_ee(request_body: dict = None):
        """Predict enantiomeric excess for a catalyst-substrate pair."""
        if request_body is None:
            return JSONResponse(status_code=400, content={"error": "Request body required"})
        catalyst = request_body.get("catalyst", "")
        substrate = request_body.get("substrate", "")
        if not catalyst or not substrate:
            return JSONResponse(status_code=400, content={"error": "catalyst and substrate fields required"})

        try:
            from zynerji_chirality.catalysts.ee_predictor import EEPredictor
            from zynerji_chirality.catalysts.benchmarks import get_benchmark_data

            predictor = EEPredictor(nbits=64)
            predictor.fit(get_benchmark_data())
            pred = predictor.predict_ee(catalyst, substrate)
            return {
                "catalyst": catalyst,
                "substrate": substrate,
                "ee_percent": round(pred.ee_percent, 1),
                "major_enantiomer": pred.major_enantiomer,
                "confidence": round(pred.confidence, 3),
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return DASHBOARD_HTML

    return app


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ZynerjiChirality Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #0a0a0f;
    color: #e0e0e0;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
    padding: 16px;
}
h1 {
    color: #00d4ff;
    font-size: 20px;
    margin-bottom: 4px;
}
.subtitle { color: #888; font-size: 12px; margin-bottom: 16px; }
.grid { display: grid; gap: 12px; margin-bottom: 12px; }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.grid-2 { grid-template-columns: repeat(2, 1fr); }
.grid-1 { grid-template-columns: 1fr; }
.card {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 14px;
}
.card-title {
    color: #00d4ff;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    border-bottom: 1px solid #1e1e2e;
    padding-bottom: 6px;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
}
.stat-label { color: #888; }
.stat-value { color: #e0e0e0; font-weight: bold; }
.val-green { color: #00ff88; }
.val-red { color: #ff4444; }
.val-cyan { color: #00d4ff; }
.val-orange { color: #ffaa00; }
.val-purple { color: #c084fc; }
.badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: bold;
}
.badge-enantiomer { background: #00ff8833; color: #00ff88; }
.badge-diastereomer { background: #ffaa0033; color: #ffaa00; }
.badge-gap { background: #ff444433; color: #ff4444; }
.badge-differential { background: #c084fc33; color: #c084fc; }
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}
th {
    color: #00d4ff;
    text-align: left;
    padding: 6px 8px;
    border-bottom: 1px solid #1e1e2e;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
td {
    padding: 5px 8px;
    border-bottom: 1px solid #0d0d14;
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
tr:hover td { background: #1a1a28; }
.bar-container {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0;
}
.bar-label { width: 120px; color: #888; font-size: 11px; }
.bar-track {
    flex: 1;
    height: 16px;
    background: #1e1e2e;
    border-radius: 4px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
}
.bar-value { width: 60px; text-align: right; font-size: 11px; }
.search-box {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
}
.search-box input {
    flex: 1;
    background: #1e1e2e;
    border: 1px solid #2e2e3e;
    color: #e0e0e0;
    padding: 8px 12px;
    border-radius: 6px;
    font-family: inherit;
    font-size: 13px;
}
.search-box input:focus { outline: none; border-color: #00d4ff; }
.search-box button {
    background: #00d4ff22;
    border: 1px solid #00d4ff44;
    color: #00d4ff;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-family: inherit;
}
.search-box button:hover { background: #00d4ff33; }
.search-box select {
    background: #1e1e2e;
    border: 1px solid #2e2e3e;
    color: #e0e0e0;
    padding: 8px;
    border-radius: 6px;
    font-family: inherit;
}
.tab-bar {
    display: flex;
    gap: 4px;
    margin-bottom: 12px;
}
.tab {
    padding: 6px 14px;
    border-radius: 6px 6px 0 0;
    cursor: pointer;
    color: #888;
    border: 1px solid transparent;
    background: transparent;
    font-family: inherit;
    font-size: 12px;
}
.tab.active {
    color: #00d4ff;
    background: #12121a;
    border-color: #1e1e2e;
    border-bottom-color: #12121a;
}
.tab:hover { color: #e0e0e0; }
.tab-content { display: none; }
.tab-content.active { display: block; }
#refresh-indicator {
    position: fixed;
    top: 8px;
    right: 12px;
    color: #444;
    font-size: 10px;
}
.mol-pair {
    display: flex;
    gap: 8px;
    align-items: center;
    font-size: 11px;
}
.mol-pair .arrow { color: #00d4ff; font-size: 14px; }
.fold-change {
    font-weight: bold;
    padding: 1px 4px;
    border-radius: 2px;
}
.fold-high { background: #ff444433; color: #ff4444; }
.fold-med { background: #ffaa0033; color: #ffaa00; }
.fold-low { background: #00ff8833; color: #888; }
</style>
</head>
<body>
<h1>ZynerjiChirality</h1>
<div class="subtitle">Enantiomer Screening Pipeline &mdash; <span id="status">loading...</span></div>
<div id="enrich-bar-wrap" style="margin:10px 0;display:none">
    <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px">
        <span style="color:#ffaa00">Enrichment</span>
        <span id="enrich-pct" style="color:#ffaa00">0%</span>
    </div>
    <div style="height:10px;background:#1e1e2e;border-radius:5px;overflow:hidden">
        <div id="enrich-fill" style="height:100%;width:0%;background:linear-gradient(90deg,#ffaa00,#00d4ff);border-radius:5px;transition:width 0.5s"></div>
    </div>
    <div id="enrich-detail" style="color:#666;font-size:10px;margin-top:2px"></div>
</div>
<div id="refresh-indicator"></div>

<!-- Stats Cards -->
<div class="grid grid-3" id="stats-grid">
    <div class="card">
        <div class="card-title">Pipeline</div>
        <div id="pipeline-stats">Loading...</div>
    </div>
    <div class="card">
        <div class="card-title">Discovery</div>
        <div id="discovery-stats">Loading...</div>
    </div>
    <div class="card">
        <div class="card-title">Screening</div>
        <div id="screening-stats">Loading...</div>
    </div>
</div>

<!-- Progress Bars -->
<div class="grid grid-1">
    <div class="card">
        <div class="card-title">Breakdown</div>
        <div id="bars-container"></div>
    </div>
</div>

<!-- Tabs -->
<div class="tab-bar">
    <button class="tab active" onclick="switchTab('hits')">Screening Hits</button>
    <button class="tab" onclick="switchTab('pairs')">Enantiomer Pairs</button>
    <button class="tab" onclick="switchTab('differential')">Differential Activity</button>
    <button class="tab" onclick="switchTab('search')">Search</button>
    <button class="tab" onclick="switchTab('materials')">Materials</button>
    <button class="tab" onclick="switchTab('catalysts')">Catalysts</button>
</div>

<!-- Tab Content: Hits -->
<div class="tab-content active" id="tab-hits">
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Type</th>
                    <th>Priority</th>
                    <th>Mol A</th>
                    <th>Mol B</th>
                    <th>Relationship</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody id="hits-table"></tbody>
        </table>
    </div>
</div>

<!-- Tab Content: Pairs -->
<div class="tab-content" id="tab-pairs">
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Mol A</th>
                    <th>SMILES A</th>
                    <th>Mol B</th>
                    <th>SMILES B</th>
                    <th>Type</th>
                    <th>Centers</th>
                </tr>
            </thead>
            <tbody id="pairs-table"></tbody>
        </table>
    </div>
</div>

<!-- Tab Content: Differential -->
<div class="tab-content" id="tab-differential">
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Mol A</th>
                    <th>Mol B</th>
                    <th>Target</th>
                    <th>Type</th>
                    <th>Value A</th>
                    <th>Value B</th>
                    <th>Fold Change</th>
                </tr>
            </thead>
            <tbody id="diff-table"></tbody>
        </table>
    </div>
</div>

<!-- Tab Content: Search -->
<div class="tab-content" id="tab-search">
    <div class="search-box">
        <input type="text" id="search-smiles" placeholder="Enter SMILES (e.g. N[C@@H](C)C(=O)O)" />
        <select id="search-mode">
            <option value="similar">Similar</option>
            <option value="enantiomer">Enantiomer</option>
        </select>
        <button onclick="doSearch()">Search</button>
    </div>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Name</th>
                    <th>SMILES</th>
                    <th>Similarity</th>
                    <th>Chirality Score</th>
                    <th>Sign</th>
                </tr>
            </thead>
            <tbody id="search-results"></tbody>
        </table>
    </div>
</div>

<!-- Tab Content: Materials -->
<div class="tab-content" id="tab-materials">
    <div class="search-box">
        <input type="text" id="mat-smiles" placeholder="Enter SMILES (e.g. N[C@@H](C)C(=O)O)" />
        <button onclick="doMaterialAnalysis()">Analyze Material</button>
    </div>
    <div class="card" id="mat-results" style="display:none">
        <div class="card-title">Material Chirality Analysis</div>
        <div id="mat-results-body"></div>
    </div>
</div>

<!-- Tab Content: Catalysts -->
<div class="tab-content" id="tab-catalysts">
    <div class="search-box">
        <input type="text" id="cat-catalyst" placeholder="Catalyst SMILES" style="flex:1" />
        <input type="text" id="cat-substrate" placeholder="Substrate SMILES" style="flex:1" />
        <button onclick="doCatalystPredict()">Predict ee</button>
    </div>
    <div class="card" id="cat-results" style="display:none">
        <div class="card-title">Enantioselectivity Prediction</div>
        <div id="cat-results-body"></div>
    </div>
</div>

<script>
const BASE = window.location.pathname.replace(/\\/$/, '');
let refreshTimer;

function switchTab(name) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    event.target.classList.add('active');
}

function statRow(label, value, cls = '') {
    return `<div class="stat-row"><span class="stat-label">${label}</span><span class="stat-value ${cls}">${value}</span></div>`;
}

function bar(label, value, max, color) {
    const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
    return `<div class="bar-container">
        <span class="bar-label">${label}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
        <span class="bar-value">${value.toLocaleString()}</span>
    </div>`;
}

function foldBadge(fc) {
    if (fc >= 10) return `<span class="fold-change fold-high">${fc.toFixed(1)}x</span>`;
    if (fc >= 3) return `<span class="fold-change fold-med">${fc.toFixed(1)}x</span>`;
    return `<span class="fold-change fold-low">${fc.toFixed(1)}x</span>`;
}

function typeBadge(type) {
    if (type === 'enantiomer') return '<span class="badge badge-enantiomer">ENT</span>';
    if (type === 'diastereomer') return '<span class="badge badge-diastereomer">DIA</span>';
    if (type === 'activity_gap') return '<span class="badge badge-gap">GAP</span>';
    if (type === 'differential') return '<span class="badge badge-differential">DIFF</span>';
    return type;
}

async function fetchJSON(endpoint) {
    try {
        const r = await fetch(BASE + endpoint);
        if (!r.ok) return null;
        return await r.json();
    } catch(e) { return null; }
}

async function refresh() {
    const now = new Date().toLocaleTimeString();
    document.getElementById('refresh-indicator').textContent = 'Last refresh: ' + now;

    const [health, stats, hits, pairs, enriched] = await Promise.all([
        fetchJSON('/health'),
        fetchJSON('/stats'),
        fetchJSON('/hits?limit=50'),
        fetchJSON('/pairs?limit=50'),
        fetchJSON('/enriched?has_differential=true&limit=50'),
    ]);

    if (health) {
        document.getElementById('status').textContent = health.status;
        const fp = health.fingerprint_progress;
        const ep = health.enrich_progress;
        let fpInfo = '';
        if (fp) {
            fpInfo = statRow('FP Progress', `${fp.n_success || 0} done, ${fp.n_fail || 0} fail`, 'val-cyan');
        }
        let enrichInfo = '';
        if (ep && ep.pairs_total > 0) {
            enrichInfo = statRow('Enrichment', `${(ep.pairs_enriched || 0).toLocaleString()} / ${(ep.pairs_total || 0).toLocaleString()} (${ep.percent_complete || 0}%)`, 'val-orange') +
                statRow('With Activity', (ep.pairs_with_activity || 0).toLocaleString(), 'val-green') +
                statRow('Differential', (ep.pairs_with_differential || 0).toLocaleString(), 'val-purple');
            const wrap = document.getElementById('enrich-bar-wrap');
            wrap.style.display = 'block';
            document.getElementById('enrich-fill').style.width = (ep.percent_complete || 0) + '%';
            document.getElementById('enrich-pct').textContent = (ep.percent_complete || 0) + '%';
            document.getElementById('enrich-detail').textContent =
                (ep.pairs_enriched || 0).toLocaleString() + ' / ' + (ep.pairs_total || 0).toLocaleString() +
                ' pairs | ' + (ep.pairs_with_activity || 0) + ' with activity | ' +
                (ep.pairs_with_differential || 0) + ' differential';
        }
        document.getElementById('pipeline-stats').innerHTML =
            statRow('Status', health.status, 'val-green') +
            statRow('Uptime', Math.floor(health.uptime_seconds / 60) + ' min') +
            statRow('DB Molecules', health.molecules_in_store.toLocaleString(), 'val-cyan') +
            enrichInfo +
            fpInfo;
    }

    if (stats) {
        document.getElementById('discovery-stats').innerHTML =
            statRow('Total Pairs', stats.total_pairs.toLocaleString(), 'val-cyan') +
            statRow('Enantiomers', stats.enantiomer_pairs.toLocaleString(), 'val-green') +
            statRow('Diastereomers', stats.diastereomer_pairs.toLocaleString(), 'val-orange') +
            statRow('Fingerprinted', stats.molecules_fingerprinted.toLocaleString(), 'val-purple');

        document.getElementById('screening-stats').innerHTML =
            statRow('Total Hits', stats.total_hits.toLocaleString(), 'val-cyan') +
            statRow('With Activity', stats.pairs_with_activity.toLocaleString(), 'val-green') +
            statRow('Differential', stats.pairs_with_differential.toLocaleString(), 'val-orange') +
            statRow('Activity Gaps', stats.pairs_with_gaps.toLocaleString(), 'val-red');

        const maxBar = Math.max(stats.total_pairs || 1, 1);
        document.getElementById('bars-container').innerHTML =
            bar('Enantiomers', stats.enantiomer_pairs, maxBar, '#00ff88') +
            bar('Diastereomers', stats.diastereomer_pairs, maxBar, '#ffaa00') +
            bar('With Activity', stats.pairs_with_activity, maxBar, '#00d4ff') +
            bar('Differential', stats.pairs_with_differential, maxBar, '#c084fc') +
            bar('Activity Gaps', stats.pairs_with_gaps, maxBar, '#ff4444') +
            bar('Fingerprinted', stats.molecules_fingerprinted, maxBar * 2, '#00d4ff');
    }

    if (hits && hits.hits) {
        document.getElementById('hits-table').innerHTML = hits.hits.map(h =>
            `<tr>
                <td>${typeBadge(h.hit_type)}</td>
                <td class="val-cyan">${h.priority_score.toFixed(1)}</td>
                <td>${h.mol_a_chembl_id || ''}</td>
                <td>${h.mol_b_chembl_id || ''}</td>
                <td>${typeBadge(h.relationship)}</td>
                <td title="${h.description}">${h.description.substring(0, 80)}...</td>
            </tr>`
        ).join('');
    }

    if (pairs && pairs.pairs) {
        document.getElementById('pairs-table').innerHTML = pairs.pairs.map(p =>
            `<tr>
                <td>${p.mol_a_id}</td>
                <td title="${p.mol_a_smiles}">${p.mol_a_smiles.substring(0, 30)}</td>
                <td>${p.mol_b_id}</td>
                <td title="${p.mol_b_smiles}">${p.mol_b_smiles.substring(0, 30)}</td>
                <td>${typeBadge(p.relationship)}</td>
                <td>${p.n_centers_a}</td>
            </tr>`
        ).join('');
    }

    if (enriched && enriched.pairs) {
        let diffRows = '';
        for (const e of enriched.pairs) {
            for (const dt of (e.differential_targets || [])) {
                diffRows += `<tr>
                    <td>${e.mol_a_id}</td>
                    <td>${e.mol_b_id}</td>
                    <td title="${dt.target_id}">${dt.target_name || dt.target_id}</td>
                    <td>${dt.type}</td>
                    <td>${typeof dt.mean_a === 'number' ? dt.mean_a.toFixed(2) : '-'}</td>
                    <td>${typeof dt.mean_b === 'number' ? dt.mean_b.toFixed(2) : '-'}</td>
                    <td>${foldBadge(dt.fold_change || 0)}</td>
                </tr>`;
            }
        }
        document.getElementById('diff-table').innerHTML = diffRows || '<tr><td colspan="7" style="color:#888">No differential activity found yet</td></tr>';
    }
}

async function doSearch() {
    const smiles = document.getElementById('search-smiles').value.trim();
    const mode = document.getElementById('search-mode').value;
    if (!smiles) return;

    const data = await fetchJSON('/search?smiles=' + encodeURIComponent(smiles) + '&mode=' + mode + '&k=20');
    if (!data || !data.results) {
        document.getElementById('search-results').innerHTML =
            '<tr><td colspan="6" style="color:#ff4444">Search failed</td></tr>';
        return;
    }

    document.getElementById('search-results').innerHTML = data.results.map((r, i) =>
        `<tr>
            <td>${i + 1}</td>
            <td>${r.name || '-'}</td>
            <td title="${r.smiles}">${r.smiles.substring(0, 40)}</td>
            <td class="${r.similarity > 0.8 ? 'val-green' : r.similarity > 0.6 ? 'val-orange' : ''}">${r.similarity.toFixed(4)}</td>
            <td>${r.chirality_score.toFixed(4)}</td>
            <td class="${r.chirality_sign > 0 ? 'val-green' : r.chirality_sign < 0 ? 'val-red' : ''}">${r.chirality_sign > 0 ? 'R' : r.chirality_sign < 0 ? 'S' : '-'}</td>
        </tr>`
    ).join('') || '<tr><td colspan="6" style="color:#888">No results</td></tr>';
}

document.getElementById('search-smiles').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') doSearch();
});

async function doMaterialAnalysis() {
    const smiles = document.getElementById('mat-smiles').value.trim();
    if (!smiles) return;
    const card = document.getElementById('mat-results');
    const body = document.getElementById('mat-results-body');
    body.innerHTML = '<div style="color:#888">Analyzing...</div>';
    card.style.display = 'block';

    try {
        const r = await fetch(BASE + '/materials/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({smiles})
        });
        const data = await r.json();
        if (data.error) {
            body.innerHTML = `<div style="color:#ff4444">Error: ${data.error}</div>`;
            return;
        }
        const chiral = data.is_chiral ? '<span class="val-green">CHIRAL</span>' : '<span class="val-red">ACHIRAL</span>';
        body.innerHTML =
            statRow('Status', chiral) +
            statRow('Chirality Score', data.chirality_score.toFixed(6), 'val-cyan') +
            statRow('Material Class', data.material_class) +
            '<div style="margin:8px 0;border-top:1px solid #1e1e2e;padding-top:8px"><span style="color:#00d4ff;font-size:11px">CIRCULAR DICHROISM</span></div>' +
            statRow('CD Activity', data.cd_activity.toFixed(6), data.cd_sign === 'positive' ? 'val-green' : data.cd_sign === 'negative' ? 'val-red' : '') +
            statRow('CD Sign', data.cd_sign) +
            '<div style="margin:8px 0;border-top:1px solid #1e1e2e;padding-top:8px"><span style="color:#00d4ff;font-size:11px">SPIN SELECTIVITY</span></div>' +
            statRow('CISS Score', data.ciss_score.toFixed(4), data.ciss_score > 0.5 ? 'val-green' : '') +
            statRow('Optical Rotation', data.optical_rotation) +
            statRow('Band Gap Shift', data.band_gap_shift.toFixed(6));
    } catch(e) {
        body.innerHTML = `<div style="color:#ff4444">Error: ${e.message}</div>`;
    }
}

async function doCatalystPredict() {
    const catalyst = document.getElementById('cat-catalyst').value.trim();
    const substrate = document.getElementById('cat-substrate').value.trim();
    if (!catalyst || !substrate) return;
    const card = document.getElementById('cat-results');
    const body = document.getElementById('cat-results-body');
    body.innerHTML = '<div style="color:#888">Predicting (training model on benchmark data)...</div>';
    card.style.display = 'block';

    try {
        const r = await fetch(BASE + '/catalysts/predict-ee', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({catalyst, substrate})
        });
        const data = await r.json();
        if (data.error) {
            body.innerHTML = `<div style="color:#ff4444">Error: ${data.error}</div>`;
            return;
        }
        const eeColor = data.ee_percent > 90 ? 'val-green' : data.ee_percent > 50 ? 'val-orange' : 'val-red';
        body.innerHTML =
            statRow('Predicted ee', data.ee_percent.toFixed(1) + '%', eeColor) +
            statRow('Major Enantiomer', data.major_enantiomer, data.major_enantiomer === 'R' ? 'val-green' : 'val-red') +
            statRow('Confidence', data.confidence.toFixed(3)) +
            '<div style="margin:8px 0;border-top:1px solid #1e1e2e;padding-top:8px;color:#666;font-size:10px">Model trained on BINAP hydrogenation benchmark (10 reactions)</div>';
    } catch(e) {
        body.innerHTML = `<div style="color:#ff4444">Error: ${e.message}</div>`;
    }
}

document.getElementById('mat-smiles').addEventListener('keydown', (e) => { if (e.key === 'Enter') doMaterialAnalysis(); });
document.getElementById('cat-substrate').addEventListener('keydown', (e) => { if (e.key === 'Enter') doCatalystPredict(); });

refresh();
refreshTimer = setInterval(refresh, 15000);
</script>
</body>
</html>
"""
