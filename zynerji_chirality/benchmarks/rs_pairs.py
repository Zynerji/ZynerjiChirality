"""R/S drug molecule pairs for chirality benchmarking.

Known chiral drugs where R and S forms have different pharmacological activity.
"""

from __future__ import annotations

# (name, R-SMILES, S-SMILES, note)
RS_DRUG_PAIRS: list[tuple[str, str, str, str]] = [
    (
        "Thalidomide",
        "O=C1CC[C@H](N1)C(=O)c2ccccc2N1C(=O)CCC1=O",
        "O=C1CC[C@@H](N1)C(=O)c2ccccc2N1C(=O)CCC1=O",
        "R=sedative, S=teratogenic",
    ),
    (
        "Ibuprofen",
        "CC(C)Cc1ccc([C@H](C)C(=O)O)cc1",
        "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1",
        "S=active anti-inflammatory, R=inactive",
    ),
    (
        "Naproxen",
        "COc1ccc2cc([C@H](C)C(=O)O)ccc2c1",
        "COc1ccc2cc([C@@H](C)C(=O)O)ccc2c1",
        "S=anti-inflammatory, R=liver toxin",
    ),
    (
        "Omeprazole",
        "COc1ccc2[nH]c([S@](=O)Cc3ncc(C)c(OC)c3C)nc2c1",
        "COc1ccc2[nH]c([S@@](=O)Cc3ncc(C)c(OC)c3C)nc2c1",
        "S=esomeprazole (more active PPI)",
    ),
    (
        "Ketamine",
        "O=C1CCCC[C@]1(NC)c1ccccc1Cl",
        "O=C1CCCC[C@@]1(NC)c1ccccc1Cl",
        "S=stronger anesthetic, R=analgesic",
    ),
    (
        "Citalopram",
        "N#Cc1ccc2c(c1)[C@](CCCCN1CCC1)(c1ccc(F)cc1)CO2",
        "N#Cc1ccc2c(c1)[C@@](CCCCN1CCC1)(c1ccc(F)cc1)CO2",
        "S=escitalopram (active SSRI)",
    ),
    (
        "Propranolol",
        "CC(C)NC[C@@H](O)COc1cccc2ccccc12",
        "CC(C)NC[C@H](O)COc1cccc2ccccc12",
        "S=beta-blocker, R=less active",
    ),
    (
        "Warfarin",
        "OC(CC(=O)c1ccccc1)[C@H]1CC(=O)c2ccccc2O1",
        "OC(CC(=O)c1ccccc1)[C@@H]1CC(=O)c2ccccc2O1",
        "S=5x more potent anticoagulant",
    ),
    (
        "Methylphenidate",
        "OC(=O)[C@@H]([C@H]1CCCCN1)c1ccccc1",
        "OC(=O)[C@H]([C@@H]1CCCCN1)c1ccccc1",
        "d-threo=Ritalin (active), l-threo=less active",
    ),
    (
        "Amphetamine",
        "C[C@H](N)Cc1ccccc1",
        "C[C@@H](N)Cc1ccccc1",
        "D=dextroamphetamine (stronger CNS), L=levamphetamine",
    ),
    (
        "Methadone",
        "CC[C@@](C(=O)CC)(c1ccccc1)CC(C)N(C)C",
        "CC[C@](C(=O)CC)(c1ccccc1)CC(C)N(C)C",
        "R=active opioid analgesic, S=less active",
    ),
    (
        "Penicillamine",
        "C[C@](N)(CS)C(=O)O",
        "C[C@@](N)(CS)C(=O)O",
        "D=antirheumatic, L=toxic",
    ),
]


def get_all_pairs() -> list[tuple[str, str, str, str]]:
    """Return all R/S drug pairs: [(name, R_smiles, S_smiles, note)]."""
    return list(RS_DRUG_PAIRS)
