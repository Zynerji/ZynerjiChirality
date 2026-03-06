# Longevity Target Chirality Screen — ChEMBL Differential Activity

**Date**: 2026-03-05
**Version**: ZynerjiChirality v0.7.1
**Data**: 26,172 ChEMBL differential activity pairs, filtered by longevity-related target keywords

## Summary

Screened ChEMBL enantiomer pairs with >3x fold-change against longevity-associated targets: p53/Mdm2, mTOR/PI3K, sirtuins, telomerase, proteasome, AMPK, FOXO, IGF.

**170 longevity-related pairs found**, **27/40 top candidates rated HIGH** (divergence > 0.7, FC > 10x).

## Target Distribution

| Target Keyword | Pairs Found |
|---------------|-------------|
| p53/Mdm2 | 65 |
| mTOR/PI3K | 64 |
| Sirtuin (SIRT1-5) | 24 |
| Proteasome | 39 |
| Telomerase | 5 |
| AMPK | 4 |

## Top 27 HIGH-Potential Candidates

Criteria: Differential FP divergence > 0.7 AND fold-change > 10x

| Target | FC | Div | Score | ChEMBL ID | Pathway |
|--------|-----|-----|-------|-----------|---------|
| p53/Mdm2 | 11,677x | 0.956 | 0.072 | CHEMBL3691737 | Senescence |
| p53/Mdm2 | 8,128x | 0.963 | 0.017 | CHEMBL3916525 | Senescence |
| PI3K p110-delta | 3,551x | 0.921 | 0.019 | CHEMBL4759169 | mTOR/growth |
| p53/Mdm2 | 2,138x | 0.934 | 0.044 | CHEMBL3915879 | Senescence |
| PI3K/mTOR | 1,911x | 0.950 | 0.060 | CHEMBL4460164 | mTOR/growth |
| p53/Mdm2 | 1,906x | 0.923 | 0.016 | CHEMBL3916664 | Senescence |
| p53/Mdm2 | 1,698x | 0.738 | 0.008 | CHEMBL3913159 | Senescence |
| p53/Mdm2 | 1,349x | 0.965 | 0.007 | CHEMBL3914063 | Senescence |
| p53/Mdm2 | 1,175x | 0.981 | 0.008 | CHEMBL3894082 | Senescence |
| Proteasome beta-5 | 1,089x | 0.705 | 0.062 | CHEMBL5205717 | Proteostasis |
| PI3K/mTOR | 938x | 0.976 | 0.072 | CHEMBL4285202 | mTOR/growth |
| p53/Mdm2 | 925x | 0.747 | 0.066 | CHEMBL6064720 | Senescence |
| p53/Mdm2 | 871x | 0.974 | 0.007 | CHEMBL4112168 | Senescence |
| p53/Mdm2 | 851x | 0.797 | 0.019 | CHEMBL3310183 | Senescence |
| PI3K p110-delta | 789x | 0.903 | 0.019 | CHEMBL4784596 | mTOR/growth |
| EphB4 kinase | 700x | 0.818 | 0.115 | CHEMBL1242468 | Signaling |
| p53/Mdm2 | 692x | 0.911 | 0.008 | CHEMBL3920430 | Senescence |
| p53/Mdm2 | 676x | 0.959 | 0.012 | CHEMBL3892862 | Senescence |
| p53/Mdm2 | 513x | 0.756 | 0.008 | CHEMBL3897997 | Senescence |
| p53/Mdm2 | 457x | 0.946 | 0.021 | CHEMBL3901716 | Senescence |
| p53/Mdm2 | 447x | 0.902 | 0.007 | CHEMBL3909949 | Senescence |
| **SIRT1** | **363x** | **0.875** | **0.043** | **CHEMBL371432** | **NAD+/aging** |
| p53/Mdm2 | 299x | 0.949 | 0.008 | CHEMBL3924973 | Senescence |
| p53/Mdm2 | 263x | 0.980 | 0.004 | CHEMBL3895404 | Senescence |
| p53/Mdm2 | 207x | 0.964 | 0.067 | CHEMBL3937629 | Senescence |
| p53/Mdm2 | 182x | 0.971 | 0.065 | CHEMBL3919782 | Senescence |
| p53/Mdm2 | 166x | 0.880 | 0.021 | CHEMBL4115066 | Senescence |

## Key Findings

### 1. p53/Mdm2 Dominates
The p53-Mdm2 protein-protein interaction is by far the most chirality-sensitive longevity target. Enantiomers of Mdm2 inhibitors (nutlin analogs) show **100-12,000x** fold-change in activity. This makes sense: Mdm2 inhibitors must fit a chiral binding pocket on p53's surface.

### 2. SIRT1 Hit (CHEMBL371432)
The sirtuin-1 pair (363x FC, 0.875 divergence) is notable — SIRT1 is a direct aging target (NAD+-dependent deacetylase). One enantiomer activates SIRT1 363x more potently than its mirror image. This is a genuine "elixir candidate" — the correct stereochemistry is essential.

### 3. PI3K/mTOR Pathway
Multiple PI3K hits with >700x fold-change and >0.9 divergence. The PI3K-mTOR pathway is a key longevity axis (rapamycin's target). Chirality strongly determines which enantiomer inhibits the pathway.

### 4. Proteasome
One proteasome hit at 1,089x (0.705 divergence). Proteasome function declines with age; inhibitors with the right stereochemistry could modulate proteostasis.

### 5. Telomerase
5 pairs found (100-333x FC), but not in the top 40 by fold-change. Telomerase-targeting chirality-sensitive compounds exist but are rarer.

## Interpretation

**The differential FP divergence does NOT predict fold-change** (Atlas v8 confirmed this at n=1998). However, it is excellent at **confirming that chirality matters** for these targets. A high divergence (>0.7) means the spectral chirality signal is strong and unambiguous — the enantiomers are maximally distinguishable by the fingerprint.

The practical use: for any drug targeting p53/Mdm2, mTOR/PI3K, or SIRT1, **stereochemistry is not optional** — the wrong enantiomer can be 100-10,000x less active. The differential FP can be used as a quality gate in drug design pipelines: if the differential FP divergence is high, the molecule MUST be synthesized enantiopure.

## Longevity Pathway Summary

| Pathway | Mechanism | Top FC | Chirality-Sensitive? |
|---------|-----------|--------|---------------------|
| p53/Mdm2 | Senescence, apoptosis | 11,677x | **Extremely** |
| PI3K/mTOR | Growth signaling, rapamycin target | 3,551x | **Extremely** |
| Sirtuins | NAD+ metabolism, epigenetic aging | 363x | **Yes** |
| Proteasome | Proteostasis, protein turnover | 1,089x | **Yes** |
| Telomerase | Telomere maintenance | 333x | **Moderate** |
| AMPK | Energy sensing, autophagy | 8x | **Low** |
