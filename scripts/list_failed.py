#!/usr/bin/env python3
"""List molecules that have no fingerprint in the DB."""
import sqlite3
import sys

db = sys.argv[1] if len(sys.argv) > 1 else "chembl_work/chembl_screen.db"
conn = sqlite3.connect(db)
rows = conn.execute(
    "SELECT m.id, m.name, LENGTH(m.smiles), m.smiles "
    "FROM molecules m "
    "WHERE m.id NOT IN (SELECT mol_id FROM fingerprints) "
    "ORDER BY LENGTH(m.smiles)"
).fetchall()
print(f"Failed molecules: {len(rows)}")
for mid, name, slen, smi in rows:
    print(f"  id={mid} name={name} len={slen} smiles={smi[:120]}")
conn.close()
