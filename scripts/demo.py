#!/usr/bin/env python3
"""Quick demonstration of ZynerjiChirality.

Shows chirality detection on L/D-alanine and a few drug molecules.
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

    # --- 3. Ibuprofen R vs S ---
    print("\n3. Ibuprofen R vs S")
    print("-" * 40)

    ibu_r = "CC(C)Cc1ccc([C@H](C)C(=O)O)cc1"
    ibu_s = "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1"

    comp_ibu = detector.compare_enantiomers(ibu_r, ibu_s)
    print(f"  R-Ibuprofen: score={comp_ibu.result_a.chirality_score:.4f}, "
          f"sign={comp_ibu.result_a.chirality_sign:+.0f}")
    print(f"  S-Ibuprofen: score={comp_ibu.result_b.chirality_score:.4f}, "
          f"sign={comp_ibu.result_b.chirality_sign:+.0f}")
    print(f"  Are enantiomers: {comp_ibu.are_enantiomers}")

    # --- 4. Fingerprint similarity ---
    print("\n4. Fingerprint Similarity")
    print("-" * 40)

    fp_l = chirality_fingerprint(l_ala)
    fp_d = chirality_fingerprint(d_ala)
    fp_gly = chirality_fingerprint(glycine)

    sim_ld = fingerprint_similarity(fp_l, fp_d)
    sim_lg = fingerprint_similarity(fp_l, fp_gly)

    print(f"  L-Ala vs D-Ala similarity: {sim_ld:.4f}")
    print(f"  L-Ala vs Glycine similarity: {sim_lg:.4f}")
    print(f"  (Enantiomers should be similar but not identical)")

    # --- 5. Classification ---
    print("\n5. R/S Classification")
    print("-" * 40)

    for name, smiles in [
        ("L-Alanine", l_ala),
        ("D-Alanine", d_ala),
        ("Glycine", glycine),
    ]:
        cls = detector.classify_rs(smiles)
        print(f"  {name}: {cls}")

    # --- 6. Achiral molecules ---
    print("\n6. Achiral Controls")
    print("-" * 40)

    achiral = [
        ("Methane", "C"),
        ("Benzene", "c1ccccc1"),
        ("Ethanol", "CCO"),
    ]

    for name, smiles in achiral:
        try:
            res = detector.detect(smiles)
            print(f"  {name}: score={res.chirality_score:.6f}, chiral={res.is_chiral}")
        except Exception as e:
            print(f"  {name}: error — {e}")

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
