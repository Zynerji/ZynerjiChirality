#!/usr/bin/env python3
"""Catalyst screening and ee prediction CLI.

Usage:
    python run_catalysts.py --predict "catalyst_smiles" "substrate_smiles"
    python run_catalysts.py --screen "substrate_smiles" --ligands ligands.txt
    python run_catalysts.py --benchmark
    python run_catalysts.py --rank-ligands "SMILES1" "SMILES2" "SMILES3"
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Chiral catalyst screening and ee prediction"
    )
    parser.add_argument("--predict", nargs=2, metavar=("CATALYST", "SUBSTRATE"),
                        help="Predict ee for catalyst-substrate pair")
    parser.add_argument("--screen", type=str, metavar="SUBSTRATE",
                        help="Screen ligands for a substrate")
    parser.add_argument("--ligands", type=str,
                        help="File with ligand SMILES (one per line)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run BINAP hydrogenation benchmark")
    parser.add_argument("--rank-ligands", nargs="+", metavar="SMILES",
                        help="Rank ligands by chirality profile")

    args = parser.parse_args()

    if not any([args.predict, args.screen, args.benchmark, args.rank_ligands]):
        parser.print_help()
        sys.exit(1)

    if args.predict:
        _predict_ee(args.predict[0], args.predict[1])

    elif args.screen:
        _screen_ligands(args.screen, args.ligands)

    elif args.benchmark:
        _run_benchmark()

    elif args.rank_ligands:
        _rank_ligands(args.rank_ligands)


def _predict_ee(catalyst_smi, substrate_smi):
    """Predict ee for a single catalyst-substrate pair."""
    from zynerji_chirality.catalysts.ee_predictor import EEPredictor
    from zynerji_chirality.catalysts.benchmarks import get_benchmark_data

    print(f"\n  EE Prediction")
    print("=" * 60)
    print(f"  Catalyst:  {catalyst_smi}")
    print(f"  Substrate: {substrate_smi}")

    # Train on benchmark data
    predictor = EEPredictor(nbits=64)
    data = get_benchmark_data()

    print(f"\n  Training on {len(data)} BINAP benchmark reactions...")
    try:
        predictor.fit(data)
    except Exception as e:
        print(f"  Training failed: {e}")
        print("  Using heuristic mode instead.")
        # Fallback: just show chirality analysis
        from zynerji_chirality.chirality.detector import HelixChiralityDetector
        det = HelixChiralityDetector()
        cat_result = det.detect(catalyst_smi)
        sub_result = det.detect(substrate_smi)
        print(f"\n  Catalyst chirality: {'CHIRAL' if cat_result.is_chiral else 'ACHIRAL'} "
              f"(score={cat_result.chirality_score:.4f})")
        print(f"  Substrate chirality: {'CHIRAL' if sub_result.is_chiral else 'ACHIRAL'} "
              f"(score={sub_result.chirality_score:.4f})")
        return

    pred = predictor.predict_ee(catalyst_smi, substrate_smi)
    print(f"\n  Predicted ee:         {pred.ee_percent:.1f}%")
    print(f"  Major enantiomer:     {pred.major_enantiomer}")
    print(f"  Confidence:           {pred.confidence:.3f}")


def _screen_ligands(substrate_smi, ligands_file):
    """Screen candidate ligands for a substrate."""
    from zynerji_chirality.catalysts.ligand_screen import LigandScreener

    if ligands_file:
        with open(ligands_file) as f:
            ligands = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        print("  No ligands file provided. Using example ligands.")
        ligands = [
            "N[C@@H](C)C(=O)O",            # L-alanine
            "N[C@H](C)C(=O)O",             # D-alanine
            "N[C@@H](CC1=CC=CC=C1)C(=O)O", # L-phenylalanine
            "NCC(=O)O",                      # glycine (achiral)
        ]

    print(f"\n  Ligand Screening")
    print("=" * 60)
    print(f"  Substrate: {substrate_smi}")
    print(f"  Ligands:   {len(ligands)}")

    screener = LigandScreener()
    results = screener.screen(substrate_smi, ligands)

    print(f"\n  {'Rank':<6} {'Predicted ee':<15} {'FP Distance':<15} {'Ligand SMILES'}")
    print(f"  {'-'*6} {'-'*15} {'-'*15} {'-'*30}")
    for r in results:
        print(f"  {r.rank:<6} {r.predicted_ee:<15.2f} {r.fingerprint_distance:<15.4f} {r.ligand_smiles[:30]}")


def _run_benchmark():
    """Run BINAP hydrogenation benchmark."""
    from zynerji_chirality.catalysts.ee_predictor import EEPredictor
    from zynerji_chirality.catalysts.benchmarks import run_binap_benchmark, get_benchmark_data

    print(f"\n  BINAP Hydrogenation Benchmark")
    print("=" * 60)

    data = get_benchmark_data()
    print(f"  Benchmark reactions: {len(data)}")

    # Train predictor on the benchmark data itself (to verify pipeline)
    predictor = EEPredictor(nbits=64)
    try:
        predictor.fit(data)
    except Exception as e:
        print(f"  Training failed: {e}")
        return

    result = run_binap_benchmark(predictor)

    print(f"\n  Results:")
    print(f"    Reactions tested:          {result['n_reactions']}")
    print(f"    Successfully predicted:    {result.get('n_predicted', 0)}")
    print(f"    Correct major enantiomer:  {result.get('correct_major_enantiomer', 0)}")
    if "mae" in result:
        print(f"    MAE:                       {result['mae']:.1f}%")
    if "rmse" in result:
        print(f"    RMSE:                      {result['rmse']:.1f}%")
    if "correlation" in result:
        print(f"    Correlation:               {result['correlation']:.3f}")

    print(f"\n  Per-reaction results:")
    print(f"  {'Type':<40} {'Actual':<10} {'Predicted':<12} {'Major'}")
    print(f"  {'-'*40} {'-'*10} {'-'*12} {'-'*10}")
    for p in result["predictions"]:
        if "predicted_ee" in p:
            match = "OK" if p.get("actual_major") == p.get("predicted_major") else "MISS"
            print(f"  {p['reaction_type']:<40} {p['actual_ee']:<10.1f} "
                  f"{p['predicted_ee']:<12.1f} {match}")
        else:
            print(f"  {p['reaction_type']:<40} {p['actual_ee']:<10.1f} {'ERROR':<12} -")


def _rank_ligands(smiles_list):
    """Rank ligands by chirality profile."""
    from zynerji_chirality.catalysts.ligand_screen import rank_by_chirality_profile

    print(f"\n  Ligand Chirality Ranking")
    print("=" * 60)

    ranked = rank_by_chirality_profile(smiles_list)

    print(f"\n  {'Rank':<6} {'Score':<15} {'SMILES'}")
    print(f"  {'-'*6} {'-'*15} {'-'*30}")
    for i, (smi, score) in enumerate(ranked):
        print(f"  {i+1:<6} {score:<15.6f} {smi[:40]}")


if __name__ == "__main__":
    main()
