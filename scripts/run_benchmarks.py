#!/usr/bin/env python3
"""Full benchmark suite for ZynerjiChirality.

Runs chirality detection on:
1. L/D amino acid pairs (19 chiral + glycine achiral)
2. R/S drug pairs (12+ pairs)
3. Achiral negative controls
4. Meso compounds
"""

from __future__ import annotations

import sys
import time
import traceback

from zynerji_chirality.chirality.detector import HelixChiralityDetector
from zynerji_chirality.benchmarks.amino_acids import (
    get_all_pairs as get_amino_pairs,
    get_achiral as get_amino_achiral,
)
from zynerji_chirality.benchmarks.rs_pairs import get_all_pairs as get_drug_pairs
from zynerji_chirality.benchmarks.known_molecules import (
    get_achiral,
    get_meso,
)
from zynerji_chirality.benchmarks.expanded import (
    get_sugar_pairs,
    get_natural_product_pairs,
    get_steroid_molecules,
    get_tough_achiral,
)


def run_amino_acid_benchmark(detector: HelixChiralityDetector) -> dict:
    """Benchmark on L/D amino acid pairs."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: L/D Amino Acid Pairs")
    print("=" * 70)

    pairs = get_amino_pairs()
    n_detected = 0
    n_correct_sign = 0
    n_opposite = 0
    n_total = len(pairs)
    results = []

    for name, l_smiles, d_smiles in pairs:
        try:
            comp = detector.compare_enantiomers(l_smiles, d_smiles)
            l_res = comp.result_a
            d_res = comp.result_b

            l_chiral = l_res.is_chiral
            d_chiral = d_res.is_chiral
            detected = l_chiral and d_chiral
            opposite = comp.signs_opposite

            if detected:
                n_detected += 1
            if opposite:
                n_opposite += 1

            # For amino acids, L = S config, D = R config
            l_correct = l_res.chirality_sign < 0 if l_chiral else False
            d_correct = d_res.chirality_sign > 0 if d_chiral else False
            if l_correct and d_correct:
                n_correct_sign += 1

            status = "OK" if detected and opposite else "FAIL"
            print(
                f"  {status:4s} {name:15s} | "
                f"L: score={l_res.chirality_score:.4f} sign={l_res.chirality_sign:+.0f} | "
                f"D: score={d_res.chirality_score:.4f} sign={d_res.chirality_sign:+.0f} | "
                f"opposite={opposite}"
            )
            results.append((name, l_res, d_res, comp))
        except Exception as e:
            print(f"  ERR  {name:15s} | {e}")
            traceback.print_exc()

    # Glycine (achiral)
    achiral_list = get_amino_achiral()
    glycine_correct = False
    for name, smiles in achiral_list:
        try:
            res = detector.detect(smiles)
            glycine_correct = not res.is_chiral
            status = "OK" if glycine_correct else "FAIL"
            print(f"  {status:4s} {name:15s} | score={res.chirality_score:.4f} (achiral={not res.is_chiral})")
        except Exception as e:
            print(f"  ERR  {name:15s} | {e}")

    print(f"\n  Detection: {n_detected}/{n_total} chiral pairs detected")
    print(f"  Opposite signs: {n_opposite}/{n_total}")
    print(f"  Correct R/S: {n_correct_sign}/{n_total}")
    print(f"  Glycine achiral: {glycine_correct}")

    return {
        "n_detected": n_detected,
        "n_total": n_total,
        "n_opposite": n_opposite,
        "n_correct_sign": n_correct_sign,
        "glycine_correct": glycine_correct,
    }


def run_drug_benchmark(detector: HelixChiralityDetector) -> dict:
    """Benchmark on R/S drug pairs."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: R/S Drug Molecule Pairs")
    print("=" * 70)

    pairs = get_drug_pairs()
    n_detected = 0
    n_opposite = 0
    n_total = len(pairs)

    for name, r_smiles, s_smiles, note in pairs:
        try:
            comp = detector.compare_enantiomers(r_smiles, s_smiles)
            r_res = comp.result_a
            s_res = comp.result_b

            detected = r_res.is_chiral and s_res.is_chiral
            opposite = comp.signs_opposite

            if detected:
                n_detected += 1
            if opposite:
                n_opposite += 1

            status = "OK" if detected else "FAIL"
            print(
                f"  {status:4s} {name:18s} | "
                f"R: score={r_res.chirality_score:.4f} sign={r_res.chirality_sign:+.0f} | "
                f"S: score={s_res.chirality_score:.4f} sign={s_res.chirality_sign:+.0f} | "
                f"opp={opposite} | {note}"
            )
        except Exception as e:
            print(f"  ERR  {name:18s} | {e}")
            traceback.print_exc()

    print(f"\n  Detection: {n_detected}/{n_total} chiral pairs detected")
    print(f"  Opposite signs: {n_opposite}/{n_total}")

    return {
        "n_detected": n_detected,
        "n_total": n_total,
        "n_opposite": n_opposite,
    }


def run_achiral_benchmark(detector: HelixChiralityDetector) -> dict:
    """Benchmark on achiral negative controls."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Achiral Negative Controls")
    print("=" * 70)

    molecules = get_achiral()
    n_correct = 0
    n_total = len(molecules)

    for name, smiles in molecules:
        try:
            res = detector.detect(smiles)
            correct = not res.is_chiral
            if correct:
                n_correct += 1
            status = "OK" if correct else "FAIL"
            print(f"  {status:4s} {name:18s} | score={res.chirality_score:.6f} achiral={not res.is_chiral}")
        except Exception as e:
            print(f"  ERR  {name:18s} | {e}")

    print(f"\n  Achiral rejection: {n_correct}/{n_total}")

    return {"n_correct": n_correct, "n_total": n_total}


def run_meso_benchmark(detector: HelixChiralityDetector) -> dict:
    """Benchmark on meso compounds (have chiral centers but achiral overall)."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Meso Compounds")
    print("=" * 70)

    molecules = get_meso()
    n_correct = 0
    n_total = len(molecules)

    for name, smiles in molecules:
        try:
            res = detector.detect(smiles)
            # Meso compounds may still show nonzero score due to local chirality
            # but should have lower scores than true chiral molecules
            print(f"  INFO {name:25s} | score={res.chirality_score:.6f} chiral={res.is_chiral}")
            if not res.is_chiral:
                n_correct += 1
        except Exception as e:
            print(f"  ERR  {name:25s} | {e}")

    print(f"\n  Meso correctly achiral: {n_correct}/{n_total}")

    return {"n_correct": n_correct, "n_total": n_total}


def run_expanded_benchmark(detector: HelixChiralityDetector) -> dict:
    """Benchmark on expanded molecules: sugars, natural products, steroids."""
    print("\n" + "=" * 70)
    print("BENCHMARK 5: Expanded (Sugars, Natural Products, Steroids)")
    print("=" * 70)

    results = {"sugars": {}, "natural_products": {}, "steroids": {}, "tough_achiral": {}}

    # Sugar D/L pairs
    print("\n  --- Sugar D/L Pairs ---")
    sugar_pairs = get_sugar_pairs()
    n_detected = 0
    n_opposite = 0
    for name, d_smiles, l_smiles in sugar_pairs:
        try:
            comp = detector.compare_enantiomers(d_smiles, l_smiles)
            detected = comp.result_a.is_chiral and comp.result_b.is_chiral
            opposite = comp.signs_opposite
            if detected:
                n_detected += 1
            if opposite:
                n_opposite += 1
            status = "OK" if detected and opposite else "FAIL"
            print(
                f"  {status:4s} {name:12s} | "
                f"D: score={comp.result_a.chirality_score:.4f} sign={comp.result_a.chirality_sign:+.0f} | "
                f"L: score={comp.result_b.chirality_score:.4f} sign={comp.result_b.chirality_sign:+.0f} | "
                f"opposite={opposite}"
            )
        except Exception as e:
            print(f"  ERR  {name:12s} | {e}")
    results["sugars"] = {"n_detected": n_detected, "n_total": len(sugar_pairs), "n_opposite": n_opposite}

    # Natural product pairs
    print("\n  --- Natural Product Pairs ---")
    np_pairs = get_natural_product_pairs()
    n_detected = 0
    n_opposite = 0
    for name, r_smiles, s_smiles, note in np_pairs:
        try:
            comp = detector.compare_enantiomers(r_smiles, s_smiles)
            detected = comp.result_a.is_chiral and comp.result_b.is_chiral
            opposite = comp.signs_opposite
            if detected:
                n_detected += 1
            if opposite:
                n_opposite += 1
            status = "OK" if detected and opposite else "FAIL"
            print(
                f"  {status:4s} {name:12s} | "
                f"R: score={comp.result_a.chirality_score:.4f} sign={comp.result_a.chirality_sign:+.0f} | "
                f"S: score={comp.result_b.chirality_score:.4f} sign={comp.result_b.chirality_sign:+.0f} | "
                f"opp={opposite} | {note[:40]}"
            )
        except Exception as e:
            print(f"  ERR  {name:12s} | {e}")
    results["natural_products"] = {"n_detected": n_detected, "n_total": len(np_pairs), "n_opposite": n_opposite}

    # Steroids (single molecule chirality)
    print("\n  --- Steroids ---")
    steroids = get_steroid_molecules()
    n_chiral = 0
    for name, smiles, note in steroids:
        try:
            res = detector.detect(smiles)
            if res.is_chiral:
                n_chiral += 1
            status = "OK" if res.is_chiral else "FAIL"
            print(f"  {status:4s} {name:15s} | score={res.chirality_score:.4f} chiral={res.is_chiral} | {note[:40]}")
        except Exception as e:
            print(f"  ERR  {name:15s} | {e}")
    results["steroids"] = {"n_chiral": n_chiral, "n_total": len(steroids)}

    # Tough achiral controls
    print("\n  --- Tough Achiral Controls ---")
    tough = get_tough_achiral()
    n_correct = 0
    for name, smiles in tough:
        try:
            res = detector.detect(smiles)
            correct = not res.is_chiral
            if correct:
                n_correct += 1
            status = "OK" if correct else "FAIL"
            print(f"  {status:4s} {name:15s} | score={res.chirality_score:.6f} achiral={not res.is_chiral}")
        except Exception as e:
            print(f"  ERR  {name:15s} | {e}")
    results["tough_achiral"] = {"n_correct": n_correct, "n_total": len(tough)}

    print(f"\n  Sugars: {results['sugars']['n_detected']}/{results['sugars']['n_total']} detected, "
          f"{results['sugars']['n_opposite']}/{results['sugars']['n_total']} opposite signs")
    print(f"  Natural products: {results['natural_products']['n_detected']}/{results['natural_products']['n_total']} detected, "
          f"{results['natural_products']['n_opposite']}/{results['natural_products']['n_total']} opposite signs")
    print(f"  Steroids: {results['steroids']['n_chiral']}/{results['steroids']['n_total']} detected chiral")
    print(f"  Tough achiral: {results['tough_achiral']['n_correct']}/{results['tough_achiral']['n_total']} rejected")

    return results


def main():
    print("ZynerjiChirality Benchmark Suite")
    print("Dual-Helix Spectral Chirality Detection")
    print("=" * 70)

    detector = HelixChiralityDetector()
    print(f"Params: omega={detector.params.omega}, k={detector.params.k}, "
          f"alpha={detector.params.alpha}, threshold={detector.threshold}")

    t0 = time.time()

    amino_results = run_amino_acid_benchmark(detector)
    drug_results = run_drug_benchmark(detector)
    achiral_results = run_achiral_benchmark(detector)
    meso_results = run_meso_benchmark(detector)
    expanded_results = run_expanded_benchmark(detector)

    elapsed = time.time() - t0

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Amino acids - Detection: {amino_results['n_detected']}/{amino_results['n_total']}, "
          f"Opposite signs: {amino_results['n_opposite']}/{amino_results['n_total']}, "
          f"Glycine achiral: {amino_results['glycine_correct']}")
    print(f"  Drug pairs  - Detection: {drug_results['n_detected']}/{drug_results['n_total']}, "
          f"Opposite signs: {drug_results['n_opposite']}/{drug_results['n_total']}")
    print(f"  Achiral     - Rejection: {achiral_results['n_correct']}/{achiral_results['n_total']}")
    print(f"  Meso        - Rejection: {meso_results['n_correct']}/{meso_results['n_total']}")
    sr = expanded_results["sugars"]
    npr = expanded_results["natural_products"]
    ster = expanded_results["steroids"]
    ta = expanded_results["tough_achiral"]
    print(f"  Sugars      - Detection: {sr['n_detected']}/{sr['n_total']}, "
          f"Opposite: {sr['n_opposite']}/{sr['n_total']}")
    print(f"  Nat. prods  - Detection: {npr['n_detected']}/{npr['n_total']}, "
          f"Opposite: {npr['n_opposite']}/{npr['n_total']}")
    print(f"  Steroids    - Chiral: {ster['n_chiral']}/{ster['n_total']}")
    print(f"  Tough achir - Rejection: {ta['n_correct']}/{ta['n_total']}")
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
