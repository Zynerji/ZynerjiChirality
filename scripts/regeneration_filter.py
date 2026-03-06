#!/usr/bin/env python3
"""Filter screening_hits.json for cellular regeneration targets.

Produces regeneration_full_analysis.json with the same schema as
longevity_full_analysis.json for use with elixir_ml.py.

Usage:
  python scripts/regeneration_filter.py [--input chembl_work/screening_hits.json]
                                         [--output chembl_work/regeneration_full_analysis.json]
                                         [--min-fc 3.0]
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

from zynerji_chirality.chirality.fingerprint import (
    chirality_differential_fingerprint,
    fingerprint_similarity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("regen_filter")

# Regeneration pathway keyword mapping
PATHWAY_KEYWORDS = {
    "Wnt": ["wnt", "frizzled", "beta-catenin", "beta catenin", "dishevelled",
            "gsk-3", "gsk3", "axin"],
    "Notch": ["notch", "gamma-secretase", "gamma secretase", "presenilin", "nicastrin"],
    "Hedgehog": ["hedgehog", "smoothened", "patched", " shh ", "gli1", "gli2"],
    "BMP_TGF": ["bone morphogenetic", "bmp receptor", "smad",
                "tgf-beta", "tgf beta", "transforming growth factor",
                "tgfbr", "alk5", "alk-5"],
    "FGF": ["fibroblast growth factor", "fgfr", "fgf receptor"],
    "VEGF": ["vascular endothelial growth factor", "vegfr", "vegf receptor", "kdr"],
    "EGFR": ["epidermal growth factor", "egfr", "erbb", "her2", "her-2"],
    "Bcl2": ["bcl-2", "bcl2", "bcl-xl", "bclxl", "mcl-1", "mcl1"],
    "mitophagy": ["pink1", "parkin", "mitophagy", "mitochondrial"],
    "autophagy": ["autophagy", "lc3", "beclin", " atg ", "ulk1"],
    "PARP": [" parp", "poly(adp", "poly-adp"],
    "telomerase": ["telomerase", " tert "],
    "stem_cell": ["stem cell", "oct4", "sox2", "nanog", "klf4"],
    "DNA_repair": ["dna repair", "base excision", "nucleotide excision", "brca"],
}


def parse_targets(description: str) -> list[dict]:
    """Parse target entries from a screening hit description.

    Returns list of {name, chembl_id, fc} dicts.
    """
    targets = []
    # Each target line: "  TargetName (CHEMBLXXXX): 123.4x differential"
    pattern = re.compile(
        r"^\s+(.+?)\s+\(CHEMBL(\d+)\):\s+([\d.eE+]+)x\s+differential",
        re.MULTILINE,
    )
    for m in pattern.finditer(description):
        try:
            fc = float(m.group(3))
        except ValueError:
            continue
        targets.append({
            "name": m.group(1).strip(),
            "chembl_id": f"CHEMBL{m.group(2)}",
            "fc": fc,
        })
    return targets


def match_pathway(target_name: str) -> str | None:
    """Match a target name to a regeneration pathway. Returns tag or None."""
    name_lower = target_name.lower()
    for pathway, keywords in PATHWAY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return pathway
    return None


def compute_descriptors(smiles: str) -> dict | None:
    """Compute RDKit molecular descriptors from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return {
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "tpsa": Descriptors.TPSA(mol),
            "rotb": Descriptors.NumRotatableBonds(mol),
            "n_rings": Descriptors.RingCount(mol),
            "n_heavy": mol.GetNumHeavyAtoms(),
            "qed": Descriptors.qed(mol),
            "n_centers": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            "lip_violations": sum([
                Descriptors.MolWt(mol) > 500,
                Descriptors.MolLogP(mol) > 5,
                Descriptors.NumHAcceptors(mol) > 10,
                Descriptors.NumHDonors(mol) > 5,
            ]),
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Filter screening hits for regeneration targets")
    parser.add_argument("--input", default="chembl_work/screening_hits.json",
                        help="Path to screening_hits.json")
    parser.add_argument("--output", default="chembl_work/regeneration_full_analysis.json",
                        help="Output JSON path")
    parser.add_argument("--min-fc", type=float, default=3.0,
                        help="Minimum fold-change to include")
    args = parser.parse_args()

    t0 = time.time()

    # Load screening hits
    log.info("Loading %s...", args.input)
    with open(args.input) as f:
        hits = json.load(f)
    log.info("Loaded %d screening hits", len(hits))

    # Phase 1: Filter hits by regeneration keywords
    log.info("Filtering for regeneration targets...")
    matched_pairs = []  # (hit, target_info, pathway_tag)

    for hit in hits:
        targets = parse_targets(hit["description"])
        for t in targets:
            if t["fc"] < args.min_fc:
                continue
            pathway = match_pathway(t["name"])
            if pathway is not None:
                matched_pairs.append((hit, t, pathway))

    log.info("Found %d target-pair matches across pathways", len(matched_pairs))
    pathway_counts = Counter(p for _, _, p in matched_pairs)
    for pw, c in sorted(pathway_counts.items(), key=lambda x: -x[1]):
        log.info("  %s: %d", pw, c)

    # Phase 2: Deduplicate (same mol pair + same target → keep highest FC)
    seen = {}  # (id_a, id_b, target_chembl_id) → best entry
    for hit, target, pathway in matched_pairs:
        key = (hit["mol_a_chembl_id"], hit["mol_b_chembl_id"], target["chembl_id"])
        if key not in seen or target["fc"] > seen[key][1]["fc"]:
            seen[key] = (hit, target, pathway)

    log.info("After dedup: %d unique (pair, target) entries", len(seen))

    # Phase 3: Compute chirality fingerprints + descriptors
    log.info("Computing chirality fingerprints and descriptors...")
    results = []
    failed = 0

    for i, (key, (hit, target, pathway)) in enumerate(seen.items()):
        smi_a = hit["mol_a_smiles"]
        smi_b = hit["mol_b_smiles"]

        # Chirality fingerprints
        try:
            fp_a = chirality_differential_fingerprint(smi_a)
            fp_b = chirality_differential_fingerprint(smi_b)
        except Exception as e:
            failed += 1
            continue

        # Divergence
        try:
            sim = fingerprint_similarity(smi_a, smi_b)
            div = 1.0 - sim
        except Exception:
            div = float(np.linalg.norm(fp_a - fp_b))

        # Chirality scores and signs
        score_a = float(np.linalg.norm(fp_a))
        score_b = float(np.linalg.norm(fp_b))
        sign_a = float(np.sign(fp_a.sum())) if score_a > 0.001 else 0.0
        sign_b = float(np.sign(fp_b.sum())) if score_b > 0.001 else 0.0

        # Molecular descriptors (use mol_a)
        desc = compute_descriptors(smi_a)
        if desc is None:
            failed += 1
            continue

        results.append({
            "target": target["name"],
            "target_chembl_id": target["chembl_id"],
            "tag": pathway,
            "fc": target["fc"],
            "div": round(div, 6),
            "score_a": round(score_a, 6),
            "score_b": round(score_b, 6),
            "sign_a": sign_a,
            "sign_b": sign_b,
            "id_a": hit["mol_a_chembl_id"],
            "id_b": hit["mol_b_chembl_id"],
            "smi_a": smi_a,
            "smi_b": smi_b,
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in desc.items()},
            "n_pc_fps": 1,
        })

        if (i + 1) % 100 == 0:
            log.info("  Processed %d / %d pairs...", i + 1, len(seen))

    log.info("Completed: %d valid pairs, %d failed", len(results), failed)

    # Summary
    pathway_final = Counter(r["tag"] for r in results)
    log.info("Final pathway distribution:")
    for pw, c in sorted(pathway_final.items(), key=lambda x: -x[1]):
        log.info("  %s: %d", pw, c)

    # HIGH candidates
    n_high = sum(1 for r in results if r["div"] > 0.7 and r["fc"] > 10)
    n_golden = sum(1 for r in results
                   if r["div"] > 0.7 and r["fc"] > 10
                   and r["lip_violations"] <= 1 and r["qed"] > 0.3)
    log.info("HIGH candidates (div>0.7, FC>10x): %d", n_high)
    log.info("Golden candidates (HIGH + drug-like): %d", n_golden)

    # Sort by FC descending
    results.sort(key=lambda r: r["fc"], reverse=True)

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved %d pairs to %s", len(results), args.output)

    elapsed = time.time() - t0
    log.info("Done in %.1fs", elapsed)


if __name__ == "__main__":
    main()
