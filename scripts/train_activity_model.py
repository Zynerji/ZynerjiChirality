#!/usr/bin/env python3
"""Demo training of a ChiralActivityPredictor on the 12 drug pairs.

Uses synthetic activity labels (R-active=1, S-active=0, or vice versa)
since real activity data requires experimental measurements.
"""

from __future__ import annotations

import numpy as np

from zynerji_chirality.ml.activity_model import ChiralActivityPredictor
from zynerji_chirality.benchmarks.rs_pairs import get_all_pairs


def main():
    print("Training ChiralActivityPredictor on Drug Pairs (Synthetic Labels)")
    print("=" * 70)

    pairs = get_all_pairs()

    # Build training data with synthetic labels
    # For each pair: R-form gets activity based on pharmacological note
    smiles_list = []
    activities = []

    for name, r_smiles, s_smiles, note in pairs:
        smiles_list.append(r_smiles)
        smiles_list.append(s_smiles)

        # Synthetic labels: assign based on known pharmacology
        if "R=active" in note or "R-active" in note:
            activities.extend([1.0, 0.2])
        elif "S=active" in note or "S-active" in note:
            activities.extend([0.2, 1.0])
        else:
            # Unknown: assign moderate difference
            activities.extend([0.7, 0.5])

    activities = np.array(activities)

    print(f"Training set: {len(smiles_list)} molecules ({len(pairs)} pairs)")
    print(f"Activity range: [{activities.min():.1f}, {activities.max():.1f}]")

    # Train
    predictor = ChiralActivityPredictor(n_estimators=50)

    print("\nFeaturizing molecules...")
    predictor.fit(smiles_list, activities)
    print("Model trained successfully.")

    # Predict on training set (demo — not a real evaluation)
    print("\nTraining set predictions:")
    for i, (name, r_smiles, s_smiles, note) in enumerate(pairs[:5]):
        r_pred = predictor.predict(r_smiles)
        s_pred = predictor.predict(s_smiles)
        print(f"  {name:15s} | R={r_pred.prediction:.3f} S={s_pred.prediction:.3f} "
              f"| diff={abs(r_pred.prediction - s_pred.prediction):.3f}")

    print(f"\n  (Showing first 5 of {len(pairs)} pairs)")
    print("\nNote: This uses synthetic labels for demonstration.")
    print("Real activity prediction requires experimental data.")


if __name__ == "__main__":
    main()
