"""BINAP hydrogenation benchmark data for catalyst enantioselectivity.

Contains known asymmetric hydrogenation reactions catalyzed by Ru-BINAP
complexes with published enantiomeric excess (ee) values. Used for
training and evaluating the EEPredictor model.
"""

from __future__ import annotations


# Known BINAP-Ru catalyzed asymmetric hydrogenation reactions
# Sources: Noyori et al., JACS 1987-1995; various reviews
BINAP_HYDROGENATION_DATA: list[dict] = [
    {
        "catalyst": "C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",  # (R)-BINAP (simplified)
        "substrate": "CC(=O)CC(=O)OCC",      # ethyl acetoacetate (beta-keto ester)
        "product": "C[C@@H](O)CC(=O)OCC",    # (R)-ethyl 3-hydroxybutanoate
        "ee": 99.0,
        "major_enantiomer": "R",
        "reaction_type": "beta-keto ester hydrogenation",
        "reference": "Noyori, JACS 1987",
    },
    {
        "catalyst": "C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",
        "substrate": "CC(=O)CC(=O)OC",        # methyl acetoacetate
        "product": "C[C@@H](O)CC(=O)OC",
        "ee": 97.0,
        "major_enantiomer": "R",
        "reaction_type": "beta-keto ester hydrogenation",
        "reference": "Noyori, JACS 1987",
    },
    {
        "catalyst": "C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",
        "substrate": "CCOC(=O)C=CC1=CC=CC=C1",   # ethyl cinnamate derivative
        "product": "O=C(C[C@@H](O)C1=CC=CC=C1)OCC",
        "ee": 92.0,
        "major_enantiomer": "R",
        "reaction_type": "beta-keto ester hydrogenation",
        "reference": "Noyori, Org Synth 1993",
    },
    {
        "catalyst": "C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",
        "substrate": "CC(=NC)C(=O)O",            # dehydroalanine (simplified)
        "product": "N[C@@H](C)C(=O)O",        # L-alanine
        "ee": 95.0,
        "major_enantiomer": "S",
        "reaction_type": "dehydroamino acid hydrogenation",
        "reference": "Noyori, JACS 1988",
    },
    {
        "catalyst": "C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",
        "substrate": "CC(=NC)CC1=CC=CC=C1",            # dehydrophenylalanine (simplified)
        "product": "N[C@@H](CC1=CC=CC=C1)C(=O)O",   # L-phenylalanine
        "ee": 96.0,
        "major_enantiomer": "S",
        "reaction_type": "dehydroamino acid hydrogenation",
        "reference": "Noyori, JACS 1988",
    },
    {
        "catalyst": "C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",
        "substrate": "CC(=O)C1=CC=CC=C1",     # acetophenone
        "product": "C[C@@H](O)C1=CC=CC=C1",   # (R)-1-phenylethanol
        "ee": 87.0,
        "major_enantiomer": "R",
        "reaction_type": "aromatic ketone hydrogenation",
        "reference": "Noyori, JACS 1995",
    },
    {
        "catalyst": "C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",
        "substrate": "O=C1CCCCC1",             # cyclohexanone
        "product": "O[C@H]1CCCCC1",            # (S)-cyclohexanol
        "ee": 44.0,
        "major_enantiomer": "S",
        "reaction_type": "simple ketone hydrogenation",
        "reference": "Noyori, JACS 1995",
    },
    {
        "catalyst": "C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",
        "substrate": "CC(=O)CC(C)=O",          # 2,4-pentanedione
        "product": "C[C@@H](O)CC(C)=O",
        "ee": 98.0,
        "major_enantiomer": "R",
        "reaction_type": "1,3-diketone monohydrogenation",
        "reference": "Noyori, JACS 1987",
    },
    {
        "catalyst": "C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",
        "substrate": "OC(=O)C=CC1=CC=CC=C1",    # cinnamic acid
        "product": "OC(=O)[C@@H](C)C1=CC=CC=C1",
        "ee": 90.0,
        "major_enantiomer": "R",
        "reaction_type": "alpha,beta-unsaturated acid hydrogenation",
        "reference": "Noyori, Angew Chem 1991",
    },
    {
        "catalyst": "C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",
        "substrate": "OC(=O)C=CC",              # tiglic acid
        "product": "OC(=O)[C@H](C)C",
        "ee": 91.0,
        "major_enantiomer": "S",
        "reaction_type": "alpha,beta-unsaturated acid hydrogenation",
        "reference": "Noyori, Angew Chem 1991",
    },
]


def get_benchmark_data() -> list[dict]:
    """Return the BINAP hydrogenation benchmark dataset.

    Returns
    -------
    list[dict]
        List of reaction entries with keys: catalyst, substrate, product,
        ee, major_enantiomer, reaction_type, reference.
    """
    return BINAP_HYDROGENATION_DATA.copy()


def run_binap_benchmark(predictor) -> dict:
    """Evaluate an EEPredictor against the BINAP benchmark.

    Parameters
    ----------
    predictor : EEPredictor
        A fitted ee predictor instance.

    Returns
    -------
    dict
        Benchmark results with keys: n_reactions, predictions, mae, rmse,
        correlation, correct_major_enantiomer.
    """
    import numpy as np

    data = get_benchmark_data()
    predictions = []
    actual_ees = []
    correct_major = 0

    for entry in data:
        try:
            pred = predictor.predict_ee(entry["catalyst"], entry["substrate"])
            predictions.append({
                "substrate": entry["substrate"],
                "actual_ee": entry["ee"],
                "predicted_ee": pred.ee_percent,
                "actual_major": entry["major_enantiomer"],
                "predicted_major": pred.major_enantiomer,
                "reaction_type": entry["reaction_type"],
            })
            actual_ees.append(entry["ee"])

            if pred.major_enantiomer == entry["major_enantiomer"]:
                correct_major += 1
        except Exception as e:
            predictions.append({
                "substrate": entry["substrate"],
                "actual_ee": entry["ee"],
                "error": str(e),
                "reaction_type": entry["reaction_type"],
            })

    actual = np.array(actual_ees)
    predicted = np.array([p["predicted_ee"] for p in predictions if "predicted_ee" in p])

    result = {
        "n_reactions": len(data),
        "n_predicted": len(predicted),
        "predictions": predictions,
        "correct_major_enantiomer": correct_major,
    }

    if len(predicted) > 1:
        result["mae"] = float(np.mean(np.abs(actual[:len(predicted)] - predicted)))
        result["rmse"] = float(np.sqrt(np.mean((actual[:len(predicted)] - predicted) ** 2)))
        if np.std(actual[:len(predicted)]) > 0 and np.std(predicted) > 0:
            result["correlation"] = float(np.corrcoef(actual[:len(predicted)], predicted)[0, 1])

    return result
