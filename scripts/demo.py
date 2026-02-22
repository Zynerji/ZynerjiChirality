#!/usr/bin/env python3
"""Quick demonstration of ZynerjiChirality.

Shows chirality detection on L/D-alanine and a few drug molecules,
plus new features: meso detection, per-center analysis, ensemble,
fingerprint database, and retrosynthetic planning.
"""

from __future__ import annotations

import sys


def main():
    from zynerji_chirality.chirality.detector import HelixChiralityDetector
    from zynerji_chirality.chirality.fingerprint import (
        chirality_fingerprint,
        fingerprint_similarity,
    )

    print("ZynerjiChirality — Dual-Helix Spectral Chirality Detection")
    print("=" * 60)

    detector = HelixChiralityDetector()

    # --- 1. L-Alanine vs D-Alanine ---
    print("\n1. L-Alanine vs D-Alanine (enantiomers)")
    print("-" * 40)

    l_ala = "N[C@@H](C)C(=O)O"  # L-Alanine (S)
    d_ala = "N[C@H](C)C(=O)O"   # D-Alanine (R)

    comp = detector.compare_enantiomers(l_ala, d_ala)
    print(f"  L-Ala: score={comp.result_a.chirality_score:.4f}, "
          f"sign={comp.result_a.chirality_sign:+.0f}, "
          f"chiral={comp.result_a.is_chiral}")
    print(f"  D-Ala: score={comp.result_b.chirality_score:.4f}, "
          f"sign={comp.result_b.chirality_sign:+.0f}, "
          f"chiral={comp.result_b.is_chiral}")
    print(f"  Signs opposite: {comp.signs_opposite}")
    print(f"  Are enantiomers: {comp.are_enantiomers}")

    # --- 2. Glycine (achiral control) ---
    print("\n2. Glycine (achiral control)")
    print("-" * 40)

    glycine = "NCC(=O)O"
    gly_result = detector.detect(glycine)
    print(f"  Score: {gly_result.chirality_score:.6f}")
    print(f"  Is chiral: {gly_result.is_chiral}")

    # --- 3. Meso compound detection ---
    print("\n3. Meso Compound Detection")
    print("-" * 40)

    meso_tartaric = "O[C@H](C(=O)O)[C@@H](O)C(=O)O"
    meso_result = detector.detect(meso_tartaric)
    print(f"  meso-Tartaric acid: score={meso_result.chirality_score:.6f}, "
          f"chiral={meso_result.is_chiral}")

    # --- 4. Per-center analysis ---
    print("\n4. Per-Center Chirality Analysis (Isoleucine)")
    print("-" * 40)

    isoleucine = "N[C@@H]([C@@H](C)CC)C(=O)O"
    per_center = detector.detect_per_center(isoleucine)
    for idx, result in per_center.items():
        sign_str = {1.0: "R", -1.0: "S", 0.0: "-"}.get(result.chirality_sign, "?")
        print(f"  Center {idx}: score={result.chirality_score:.4f}, sign={sign_str}")

    # --- 5. Conformer ensemble ---
    print("\n5. Conformer Ensemble (L-Alanine, 5 conformers)")
    print("-" * 40)

    ensemble = detector.detect_ensemble(l_ala, n_conformers=5)
    print(f"  Mean score: {ensemble.mean_score:.4f}")
    print(f"  Std: {ensemble.std_score:.4f}")
    print(f"  Range: [{ensemble.min_score:.4f}, {ensemble.max_score:.4f}]")
    print(f"  Consensus chiral: {ensemble.consensus_is_chiral}")

    # --- 6. Fingerprint similarity ---
    print("\n6. Fingerprint Similarity")
    print("-" * 40)

    fp_l = chirality_fingerprint(l_ala)
    fp_d = chirality_fingerprint(d_ala)
    fp_gly = chirality_fingerprint(glycine)

    sim_ld = fingerprint_similarity(fp_l, fp_d)
    sim_lg = fingerprint_similarity(fp_l, fp_gly)

    print(f"  L-Ala vs D-Ala similarity: {sim_ld:.4f}")
    print(f"  L-Ala vs Glycine similarity: {sim_lg:.4f}")

    # --- 7. Fingerprint Database ---
    print("\n7. Fingerprint Database")
    print("-" * 40)

    from zynerji_chirality.db.store import FingerprintStore

    with FingerprintStore(":memory:") as store:
        store.add(l_ala, fp_l, comp.result_a.chirality_score, comp.result_a.chirality_sign, name="L-Ala")
        store.add(d_ala, fp_d, comp.result_b.chirality_score, comp.result_b.chirality_sign, name="D-Ala")
        store.add(glycine, fp_gly, gly_result.chirality_score, gly_result.chirality_sign, name="Glycine")
        print(f"  Stored {store.count()} molecules")

        results = store.search_similar(fp_l, k=3)
        print(f"  Top match for L-Ala query: {results[0].name} (sim={results[0].similarity:.4f})")

    # --- 8. Retrosynthetic Planning ---
    print("\n8. Retrosynthetic Chirality Planning (L-Alanine)")
    print("-" * 40)

    from zynerji_chirality.reactions.retro import RetroChiralityPlanner

    planner = RetroChiralityPlanner()
    suggestions = planner.plan(l_ala)
    for sug in suggestions:
        print(f"  Center {sug.atom_idx} ({sug.cip_label}): {sug.environment}")
        for strat in sug.strategies[:2]:
            print(f"    - {strat}")

    # --- 9. R/S Classification ---
    print("\n9. R/S Classification")
    print("-" * 40)

    for name, smiles in [
        ("L-Alanine", l_ala),
        ("D-Alanine", d_ala),
        ("Glycine", glycine),
    ]:
        cls = detector.classify_rs(smiles)
        print(f"  {name}: {cls}")

    print("\n" + "=" * 60)
    print("Demo complete.")

    # Optional: save visualization
    if "--plot" in sys.argv:
        from zynerji_chirality.viz import plot_enantiomer_pair

        fig = plot_enantiomer_pair(l_ala, d_ala, "L-Alanine", "D-Alanine", detector)
        fig.savefig("chirality_demo.png", dpi=150)
        print("Saved chirality_demo.png")


if __name__ == "__main__":
    main()
