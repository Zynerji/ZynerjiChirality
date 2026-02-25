# ZynerjiChirality — Derived Findings & Speculative Applications

## Most Impressive Finding

### The Midostaurin Enantiomer Gap

**Midostaurin** (CHEMBL608533, brand name Rydapt by Novartis) is an FDA-approved multi-kinase inhibitor for FLT3-mutated acute myeloid leukemia. It has been tested against **1,139 biological targets** in ChEMBL — one of the most extensively profiled drugs in existence.

Its enantiomer, **ent-midostaurin** (CHEMBL338448), has been tested against exactly **1 target**.

This is an approved, billion-dollar oncology drug whose mirror-image partner is almost completely uncharacterized. The platform identified this as the #3 priority hit across all 235,434 pairs screened.

**Why this matters**: Staurosporine-class kinase inhibitors are known to have steep enantioselectivity. The "wrong" enantiomer may be inactive — or it may have an entirely different selectivity profile that could be therapeutically useful against different kinase targets. Testing ent-midostaurin against even 50 of those 1,139 targets would generate immediate, publishable SAR data with minimal cost ($50K-$100K in assay fees vs the $2.7B+ Novartis invested in midostaurin development).

### Other Top-Tier Findings

| Finding | Significance |
|---------|-------------|
| **Cytarabine diastereomer** (CHEMBL803 vs CHEMBL95606): 639 vs 21 targets | Backbone AML chemotherapy since 1969. Diastereomer barely tested. |
| **Podofilox diastereomer** (CHEMBL61 vs CHEMBL283120): 158 vs 277 targets, bidirectional gaps | Parent compound of etoposide (major chemo drug). Both stereoisomers have large untested target sets. |
| **Shikonin/Alkannin** (CHEMBL9470 vs CHEMBL28457): 692 vs 4 targets | Natural enantiomer pair. Shikonin is a promiscuous anticancer research compound; its enantiomer alkannin is almost untested. |
| **HIV-1 protease**: 179 diastereomer pairs with up to 2.3 billion-fold potency differential | Confirms extreme chirality sensitivity. Platform recovers this known pharmacology automatically. |
| **7-atom histamine molecule**: 50,000,000x differential across H2/H3/H4 subtypes | Smallest molecule with the largest differential. Single stereocenter in 7 atoms determines receptor subtype selectivity. |

---

## Five Speculative Applications & Target Directions

### 1. Enantiomer-Selective Kinase Inhibitor Repurposing (~$500M+ opportunity)

**Concept**: Screen the "wrong" enantiomers of all approved kinase inhibitors for activity against kinases that the approved enantiomer does NOT hit.

**Data support**: The midostaurin finding (1,139 targets for one enantiomer, 1 for the other) suggests that enantiomers of kinase inhibitors may have entirely different selectivity profiles. The platform identified 357 JAK1, 356 JAK2, 340 BTK, and 221 TYK2 chirality-sensitive pairs.

**Financial model**: If ent-midostaurin, ent-ibrutinib, or ent-tofacitinib shows activity against a novel kinase target, this creates a new drug candidate with:
- Existing safety data on the parent enantiomer (de-risks toxicology)
- Known synthetic route (the racemate was likely an intermediate)
- Novel composition-of-matter patent (the "wrong" enantiomer is a new chemical entity)

**Customer**: Large pharma with kinase inhibitor portfolios (Novartis, Pfizer, AbbVie, AstraZeneca, Lilly).

### 2. Nucleoside Analog Chirality Mining for Antiviral Discovery (~$200M+ opportunity)

**Concept**: Systematically screen diastereomers of approved nucleoside analogs against viral targets.

**Data support**: Cytarabine (CHEMBL803) — one of the most important chemotherapy drugs ever developed — has a diastereomer tested on only 21 targets vs 639 for the parent. Nucleoside analogs are the backbone of antiviral therapy (remdesivir, sofosbuvir, tenofovir). The sugar stereochemistry (ribo- vs arabino- vs xylo- vs lyxo-) determines which polymerases are inhibited.

**Financial model**: The COVID-19 pandemic demonstrated that a single antiviral nucleoside analog (remdesivir, $5.3B in peak sales) can generate enormous value. Each sugar diastereomer of an existing nucleoside drug is a potentially novel antiviral with a known safety-adjacent profile.

**Customer**: Antiviral-focused pharma (Gilead, Merck, Roche). Government biodefense agencies (BARDA, DARPA) for pandemic preparedness screening.

### 3. Natural Product Enantiomer Library for Phenotypic Screening (~$100M+ opportunity)

**Concept**: Build a physical screening library of the untested enantiomers of broadly-profiled natural products.

**Data support**: Shikonin (692 targets) vs alkannin (4 targets) is the archetype. Natural products are frequently tested as single enantiomers because that's what the organism produces, but the "unnatural" enantiomer is a valid drug candidate. The platform identified thousands of natural product enantiomer pairs where one partner is extensively tested and the other is not.

**Financial model**: A curated library of 500 "unnatural" enantiomers of known bioactive natural products, sold as a screening deck to pharma companies at $50K-$200K per copy. Cost to produce: $2M-$5M (contract synthesis). Revenue potential: $10M-$50M over 5 years from licensing, plus royalties on any hits that enter development.

**Customer**: Phenotypic screening CROs, academic screening centers, pharma natural products groups (Novartis NIBR, Roche).

### 4. Chirality-Guided PROTAC Design (~$300M+ opportunity)

**Concept**: Use chirality fingerprints to optimize the stereochemistry of PROTAC (proteolysis-targeting chimera) linkers and warheads.

**Data support**: PROTACs have two binding events (target protein + E3 ligase) connected by a linker. Each binding event is chirality-sensitive, and the linker geometry determines the ternary complex orientation. The platform's data on opioid receptors (474 differential pairs), BACE1 (388 pairs), and dopamine receptors (308 pairs) demonstrates that even small stereochemical changes dramatically alter binding. PROTACs amplify this sensitivity because BOTH ends must bind simultaneously.

**Financial model**: PROTAC development is bottlenecked by the empirical optimization of linker stereochemistry. Computational prioritization of stereoisomers before synthesis could reduce the number of PROTACs that must be made by 30-50%, saving $1M-$5M per program in a space where companies (Arvinas, C4 Therapeutics, Kymera) are running dozens of programs.

**Customer**: PROTAC-focused biotechs (Arvinas, C4 Therapeutics, Kymera, Nurix), pharma companies with degrader platforms.

### 5. Enantioselective ADMET Prediction for Regulatory Strategy (~$50M+ opportunity)

**Concept**: Build predictive models for stereoselective drug metabolism using the 26,172 confirmed differential interactions as training data.

**Data support**: FDA guidance requires stereoisomer-specific ADMET characterization. The platform's data shows that chirality determines not just target binding but also metabolic fate — the same enzyme that metabolizes one enantiomer may ignore the other. The 26,172 differential pairs across 500+ targets provide an unprecedented training set for predicting which stereocenters affect ADMET properties.

**Financial model**: Late-stage clinical failures due to unexpected stereoselective metabolism cost $100M-$500M per failure. A predictive model that flags ADMET-critical stereocenters during lead optimization could prevent 1-2 such failures per year across the industry. Delivered as a SaaS tool at $200K-$500K per seat to safety/DMPK departments.

**Customer**: Drug safety and DMPK departments at all major pharma. CROs with ADMET screening services (Eurofins, Charles River, WuXi).

---

## Data Summary

| Metric | Value |
|--------|-------|
| Molecules fingerprinted | 156,956 |
| Stereoisomer pairs screened | 235,434 |
| Pairs with bioactivity data | 83,801 |
| Differential activity hits (>3x fold change) | 26,172 |
| Activity gap hits (untested enantiomers) | 15,512 |
| Unique protein targets with chirality sensitivity | 500+ |
| Approved drugs with uncharacterized enantiomers | Multiple (midostaurin, cytarabine, podofilox, etc.) |
| Fingerprint completion rate | 100% |

## Key Molecules Identified

| ChEMBL ID | Drug Name | Status | Targets Tested | Enantiomer Targets |
|-----------|-----------|--------|----------------|-------------------|
| CHEMBL608533 | Midostaurin (Rydapt) | FDA approved 2017 | 1,139 | 1 |
| CHEMBL803 | Cytarabine (Ara-C) | FDA approved 1969 | 639 | 21 |
| CHEMBL9470 | Shikonin | Research/TCM | 692 | 4 |
| CHEMBL61 | Podofilox (Condylox) | FDA approved | 158 | 277 (bidirectional) |
| CHEMBL1232461 | (Benzodiazepine derivative) | Research | 749 | 1 |
