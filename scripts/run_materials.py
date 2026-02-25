#!/usr/bin/env python3
"""Materials chirality analysis CLI.

Usage:
    python run_materials.py --smiles "N[C@@H](C)C(=O)O"
    python run_materials.py --cif path/to/structure.cif
    python run_materials.py --perovskite Cs Pb I
    python run_materials.py --compare "N[C@@H](C)C(=O)O" "N[C@H](C)C(=O)O"
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Analyze chirality of materials (molecules, crystals, polymers)"
    )
    parser.add_argument("--smiles", type=str, help="SMILES string to analyze")
    parser.add_argument("--cif", type=str, help="CIF file path for crystal structure")
    parser.add_argument("--perovskite", nargs=3, metavar=("A", "B", "X"),
                        help="Generate and analyze perovskite ABX3 (e.g., Cs Pb I)")
    parser.add_argument("--compare", nargs=2, metavar=("SMILES1", "SMILES2"),
                        help="Compare two materials")
    parser.add_argument("--lattice", type=float, default=3.9,
                        help="Lattice parameter for perovskite (default: 3.9 A)")
    parser.add_argument("--cutoff", type=float, default=3.0,
                        help="Distance cutoff for CIF adjacency (default: 3.0 A)")

    args = parser.parse_args()

    if not any([args.smiles, args.cif, args.perovskite, args.compare]):
        parser.print_help()
        sys.exit(1)

    from zynerji_chirality.materials.detector import ChiralMaterialDetector
    detector = ChiralMaterialDetector()

    if args.smiles:
        _analyze_smiles(detector, args.smiles)

    elif args.cif:
        _analyze_cif(detector, args.cif, args.cutoff)

    elif args.perovskite:
        _analyze_perovskite(detector, args.perovskite, args.lattice)

    elif args.compare:
        _compare_materials(detector, args.compare[0], args.compare[1])


def _analyze_smiles(detector, smiles):
    """Analyze a molecule as a material."""
    print(f"\n  Material Analysis: {smiles}")
    print("=" * 60)

    result = detector.detect_material(smiles)
    _print_result(result)

    # CD spectrum details
    cd = detector.predict_cd_spectrum(smiles)
    print(f"\n  CD Spectrum ({len(cd['peaks'])} modes):")
    for peak in cd["peaks"][:5]:
        sign = "+" if peak["cd_contribution"] > 0 else "-"
        print(f"    Mode {peak['mode_index']}: {sign}{abs(peak['cd_contribution']):.4f}")


def _analyze_cif(detector, cif_path, cutoff):
    """Analyze a crystal structure from CIF file."""
    from zynerji_chirality.materials.crystal import cif_to_adjacency

    print(f"\n  Crystal Analysis: {cif_path}")
    print("=" * 60)

    adj, labels = cif_to_adjacency(cif_path, cutoff=cutoff)
    print(f"  Atoms: {len(labels)}")
    print(f"  Elements: {', '.join(sorted(set(labels)))}")
    print(f"  Edges: {adj.nnz // 2}")
    print(f"  Cutoff: {cutoff} A")

    result = detector.detect_from_adjacency(adj, atom_labels=labels)
    _print_result(result)


def _analyze_perovskite(detector, elements, lattice):
    """Generate and analyze a perovskite structure."""
    from zynerji_chirality.materials.crystal import perovskite_graph

    A, B, X = elements
    print(f"\n  Perovskite Analysis: {A}{B}{X}3 (a={lattice} A)")
    print("=" * 60)

    adj, labels = perovskite_graph(A, B, X, a=lattice)
    print(f"  Unit cell: {' '.join(labels)}")
    print(f"  Edges: {adj.nnz // 2}")

    result = detector.detect_from_adjacency(adj, atom_labels=labels)
    _print_result(result)


def _compare_materials(detector, smiles1, smiles2):
    """Compare two materials side by side."""
    print(f"\n  Material Comparison")
    print("=" * 60)

    r1 = detector.detect_material(smiles1)
    r2 = detector.detect_material(smiles2)

    print(f"\n  {'Property':<25} {'Material 1':<20} {'Material 2':<20}")
    print(f"  {'-'*25} {'-'*20} {'-'*20}")
    print(f"  {'SMILES':<25} {smiles1[:18]:<20} {smiles2[:18]:<20}")
    print(f"  {'Chiral?':<25} {str(r1.is_chiral):<20} {str(r2.is_chiral):<20}")
    print(f"  {'Chirality Score':<25} {r1.chirality_score:<20.4f} {r2.chirality_score:<20.4f}")
    print(f"  {'CD Activity':<25} {r1.cd_activity:<20.4f} {r2.cd_activity:<20.4f}")
    print(f"  {'CD Sign':<25} {r1.cd_sign:<20} {r2.cd_sign:<20}")
    print(f"  {'CISS Score':<25} {r1.ciss_score:<20.3f} {r2.ciss_score:<20.3f}")
    print(f"  {'Optical Rotation':<25} {r1.optical_rotation:<20} {r2.optical_rotation:<20}")
    print(f"  {'Band Gap Shift':<25} {r1.band_gap_shift:<20.4f} {r2.band_gap_shift:<20.4f}")


def _print_result(result):
    """Print a MaterialChiralityResult."""
    label = "CHIRAL" if result.is_chiral else "ACHIRAL"
    print(f"\n  Status:          {label}")
    print(f"  Chirality Score: {result.chirality_score:.6f}")
    print(f"  Chirality Sign:  {result.chirality_sign:+.1f}")
    print(f"  Material Class:  {result.material_class}")
    print(f"\n  Circular Dichroism:")
    print(f"    Activity:      {result.cd_activity:.6f}")
    print(f"    Sign:          {result.cd_sign}")
    print(f"\n  Spin Selectivity:")
    print(f"    CISS Score:    {result.ciss_score:.4f}")
    print(f"\n  Optical Rotation: {result.optical_rotation}")
    print(f"  Band Gap Shift:   {result.band_gap_shift:.6f}")


if __name__ == "__main__":
    main()
