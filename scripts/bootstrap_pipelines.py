#!/usr/bin/env python3
"""Bootstrap COD + ORD pipelines with test/seed data.

Creates minimal structures.json and reactions.json for validating
the pipeline stages work before running with full external data.

Usage:
    PYTHONPATH=/opt/chirality python3 scripts/bootstrap_pipelines.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def bootstrap_cod(work_dir: str = "cod_work"):
    """Create seed crystal structures for COD pipeline testing."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    structures_path = Path(work_dir) / "structures.json"

    if structures_path.exists():
        with open(structures_path) as f:
            existing = json.load(f)
        print(f"COD: structures.json already exists ({len(existing)} structures)")
        return

    # Seed data: common chiral crystals with known parameters
    structures = [
        {
            "cod_id": "9008460", "formula": "Si O2",
            "space_group": "P 31 2 1", "sg_number": 152,
            "a": 4.914, "b": 4.914, "c": 5.405,
            "alpha": 90.0, "beta": 90.0, "gamma": 120.0,
            "cif_url": "https://www.crystallography.net/cod/9008460.cif",
        },
        {
            "cod_id": "2300290", "formula": "C6 H12 O6",
            "space_group": "P 21 21 21", "sg_number": 19,
            "a": 7.782, "b": 9.740, "c": 10.520,
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
            "cif_url": "https://www.crystallography.net/cod/2300290.cif",
        },
        {
            "cod_id": "2100147", "formula": "C3 H7 N O2",
            "space_group": "P 21", "sg_number": 4,
            "a": 5.325, "b": 5.940, "c": 5.997,
            "alpha": 90.0, "beta": 93.0, "gamma": 90.0,
            "cif_url": "https://www.crystallography.net/cod/2100147.cif",
        },
        {
            "cod_id": "1538445", "formula": "Ca Ti O3",
            "space_group": "P 21 21 21", "sg_number": 19,
            "a": 5.381, "b": 7.645, "c": 5.443,
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
            "cif_url": "https://www.crystallography.net/cod/1538445.cif",
        },
        {
            "cod_id": "7200361", "formula": "Zn S",
            "space_group": "P 63 m c", "sg_number": 186,
            "a": 3.820, "b": 3.820, "c": 6.260,
            "alpha": 90.0, "beta": 90.0, "gamma": 120.0,
            "cif_url": "https://www.crystallography.net/cod/7200361.cif",
        },
    ]

    with open(structures_path, "w") as f:
        json.dump(structures, f, indent=2)
    print(f"COD: Created structures.json with {len(structures)} seed structures")


def bootstrap_ord(work_dir: str = "ord_work"):
    """Create seed reactions for ORD pipeline testing."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    reactions_path = Path(work_dir) / "reactions.json"

    if reactions_path.exists():
        with open(reactions_path) as f:
            existing = json.load(f)
        print(f"ORD: reactions.json already exists ({len(existing)} reactions)")
        return

    # Seed data: known asymmetric reactions with ee values
    # Based on BINAP benchmark + additional known reactions
    reactions = [
        {
            "reaction_id": "ord-seed-001",
            "reaction_smiles": "CC(=O)CC(=O)OCC>>C[C@@H](O)CC(=O)OCC",
            "catalyst_smiles": "C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",
            "substrate_smiles": "CC(=O)CC(=O)OCC",
            "product_smiles": "C[C@@H](O)CC(=O)OCC",
            "ee_percent": 99.0,
            "yield_percent": 95.0,
            "temperature": "25 C",
            "solvent": "methanol",
            "source_doi": "10.1021/ja00240a030",
        },
        {
            "reaction_id": "ord-seed-002",
            "reaction_smiles": "CC(=O)CC(=O)OC>>C[C@@H](O)CC(=O)OC",
            "catalyst_smiles": "C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",
            "substrate_smiles": "CC(=O)CC(=O)OC",
            "product_smiles": "C[C@@H](O)CC(=O)OC",
            "ee_percent": 97.0,
            "yield_percent": 92.0,
            "temperature": "25 C",
            "solvent": "methanol",
            "source_doi": "10.1021/ja00240a030",
        },
        {
            "reaction_id": "ord-seed-003",
            "reaction_smiles": "CC(=O)C1=CC=CC=C1>>C[C@@H](O)C1=CC=CC=C1",
            "catalyst_smiles": "C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",
            "substrate_smiles": "CC(=O)C1=CC=CC=C1",
            "product_smiles": "C[C@@H](O)C1=CC=CC=C1",
            "ee_percent": 87.0,
            "yield_percent": 90.0,
            "temperature": "50 C",
            "solvent": "ethanol",
            "source_doi": "10.1021/ja00296a035",
        },
        {
            "reaction_id": "ord-seed-004",
            "reaction_smiles": "CC(=NC)C(=O)O>>N[C@@H](C)C(=O)O",
            "catalyst_smiles": "C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",
            "substrate_smiles": "CC(=NC)C(=O)O",
            "product_smiles": "N[C@@H](C)C(=O)O",
            "ee_percent": 95.0,
            "yield_percent": 88.0,
            "temperature": "40 C",
            "solvent": "THF",
            "source_doi": "10.1021/ja00223a028",
        },
        {
            "reaction_id": "ord-seed-005",
            "reaction_smiles": "CC(=O)CC(C)=O>>C[C@@H](O)CC(C)=O",
            "catalyst_smiles": "C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",
            "substrate_smiles": "CC(=O)CC(C)=O",
            "product_smiles": "C[C@@H](O)CC(C)=O",
            "ee_percent": 98.0,
            "yield_percent": 93.0,
            "temperature": "25 C",
            "solvent": "methanol",
            "source_doi": "10.1021/ja00240a030",
        },
        {
            "reaction_id": "ord-seed-006",
            "reaction_smiles": "O=C1CCCCC1>>O[C@H]1CCCCC1",
            "catalyst_smiles": "C1=CC=C(C2=CC=CC3=CC=CC=C32)C4=C1C=CC5=CC=CC=C54",
            "substrate_smiles": "O=C1CCCCC1",
            "product_smiles": "O[C@H]1CCCCC1",
            "ee_percent": 44.0,
            "yield_percent": 85.0,
            "temperature": "50 C",
            "solvent": "isopropanol",
            "source_doi": "10.1021/ja00296a035",
        },
        {
            "reaction_id": "ord-seed-007",
            "reaction_smiles": "",
            "catalyst_smiles": "N[C@@H](CC1=CC=CC=C1)C(=O)O",
            "substrate_smiles": "CC(=O)CC(=O)OCC",
            "product_smiles": "C[C@@H](O)CC(=O)OCC",
            "ee_percent": 72.0,
            "yield_percent": 80.0,
            "temperature": "60 C",
            "solvent": "toluene",
            "source_doi": "",
        },
        {
            "reaction_id": "ord-seed-008",
            "reaction_smiles": "",
            "catalyst_smiles": "N[C@@H](CC(C)C)C(=O)O",
            "substrate_smiles": "CC(=O)C1=CC=CC=C1",
            "product_smiles": "C[C@@H](O)C1=CC=CC=C1",
            "ee_percent": 65.0,
            "yield_percent": 75.0,
            "temperature": "40 C",
            "solvent": "DCM",
            "source_doi": "",
        },
        {
            "reaction_id": "ord-seed-009",
            "reaction_smiles": "",
            "catalyst_smiles": "N[C@@H](CO)C(=O)O",
            "substrate_smiles": "CC(=O)CC(=O)OC",
            "product_smiles": None,
            "ee_percent": 55.0,
            "yield_percent": 70.0,
            "temperature": "25 C",
            "solvent": "water",
            "source_doi": "",
        },
        {
            "reaction_id": "ord-seed-010",
            "reaction_smiles": "",
            "catalyst_smiles": "N[C@@H](CS)C(=O)O",
            "substrate_smiles": "CC(=O)CC(=O)OCC",
            "product_smiles": None,
            "ee_percent": 60.0,
            "yield_percent": 72.0,
            "temperature": "30 C",
            "solvent": "ethanol",
            "source_doi": "",
        },
    ]

    with open(reactions_path, "w") as f:
        json.dump(reactions, f, indent=2)
    print(f"ORD: Created reactions.json with {len(reactions)} seed reactions")


if __name__ == "__main__":
    bootstrap_cod()
    bootstrap_ord()
    print("\nBootstrap complete. Run pipelines with --resume to use seed data.")
