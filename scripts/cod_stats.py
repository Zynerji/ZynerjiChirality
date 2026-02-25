#!/usr/bin/env python3
"""Extract comprehensive statistics from COD pipeline results."""
import json
import sqlite3
from collections import Counter

with open("cod_work/enriched_structures.json") as f:
    data = json.load(f)
valid = [d for d in data if not d.get("error")]

print("=== Top 20 CD Activity Structures ===")
top_cd = sorted(valid, key=lambda d: d["cd_activity"], reverse=True)[:20]
for d in top_cd:
    s = d["structure"]
    print(f"  COD {s['cod_id']:>8s} {s['formula'][:35]:35s} SG={s['sg_number']:3d}"
          f" CD={d['cd_activity']:.4f} CISS={d['ciss_score']:.4f} atoms={d['n_atoms']}")

print("\n=== Non-chiral in Sohncke groups ===")
non_chiral = [d for d in valid if not d["is_chiral"]]
print(f"  Total: {len(non_chiral)} / {len(valid)} ({100*len(non_chiral)/len(valid):.2f}%)")
sg_dist = Counter(d["structure"]["sg_number"] for d in non_chiral)
for sg, ct in sg_dist.most_common(10):
    print(f"  SG {sg}: {ct}")

print("\n=== High CISS but Low CD (CISS>0.99, CD<0.2) ===")
mismatches = [d for d in valid if d["ciss_score"] > 0.99 and d["cd_activity"] < 0.2]
print(f"  Total mismatches: {len(mismatches)}")
for d in sorted(mismatches, key=lambda d: d["cd_activity"])[:10]:
    s = d["structure"]
    print(f"  COD {s['cod_id']} {s['formula'][:30]} CD={d['cd_activity']:.4f} CISS={d['ciss_score']:.4f}")

print("\n=== Element frequency in high-CD (>1.5) ===")
elems = Counter()
high_cd = [d for d in valid if d["cd_activity"] > 1.5]
for d in high_cd:
    for label in d["atom_labels"]:
        elems[label] += 1
total_atoms = sum(elems.values())
for elem, ct in elems.most_common(15):
    print(f"  {elem}: {ct} ({100*ct/total_atoms:.1f}%)")
print(f"  Structures with CD>1.5: {len(high_cd)}")

# Element frequency in low-CD
print("\n=== Element frequency in low-CD (<0.2) ===")
elems_low = Counter()
low_cd = [d for d in valid if d["cd_activity"] < 0.2]
for d in low_cd:
    for label in d["atom_labels"]:
        elems_low[label] += 1
total_low = sum(elems_low.values())
for elem, ct in elems_low.most_common(15):
    print(f"  {elem}: {ct} ({100*ct/total_low:.1f}%)")
print(f"  Structures with CD<0.2: {len(low_cd)}")

print("\n=== Screening hits ===")
with open("cod_work/screening_hits.json") as f:
    hits = json.load(f)
hit_types = Counter(h["hit_type"] for h in hits)
for t, ct in sorted(hit_types.items()):
    print(f"  {t}: {ct}")
print(f"  Total hits: {len(hits)}")

print("\n=== DB stats ===")
db = sqlite3.connect("cod_work/cod_screen.db")
mol_count = db.execute("SELECT COUNT(*) FROM molecules").fetchone()[0]
fp_count = db.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0]
db.close()
print(f"  Molecules: {mol_count}, Fingerprints: {fp_count}")

# Band gap shift analysis
print("\n=== Band gap shift by crystal system ===")
systems = {
    "Triclinic": range(1, 3), "Monoclinic": range(3, 16),
    "Orthorhombic": range(16, 75), "Tetragonal": range(75, 143),
    "Trigonal": range(143, 168), "Hexagonal": range(168, 195),
    "Cubic": range(195, 231),
}
import statistics
for name, sg_range in systems.items():
    vals = [d["band_gap_shift"] for d in valid if d["structure"]["sg_number"] in sg_range]
    if vals:
        print(f"  {name}: n={len(vals)}, mean={statistics.mean(vals):.6f}, stdev={statistics.stdev(vals):.6f}")
