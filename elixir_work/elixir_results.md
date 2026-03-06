# ElixirML — Longevity Chirality ML Results

**Dataset**: 170 enantiomer pairs across 7 longevity pathways

## Model 1: Fold-Change Regressor

Predicts log10(fold_change) from differential FP pairs + pathway context.

| Metric | LOGO CV | 5-Fold CV |
|--------|---------|----------|
| Spearman rho | -0.199 (p=9.2e-03) | 0.360 (p=1.5e-06) |
| MAE | 0.816 | 0.617 |
| R² | -0.401 | 0.175 |

**Baselines:**
- Mean predictor MAE: 0.699
- Divergence-only: rho=0.267, MAE=0.634

**Per-pathway (LOGO):**

| Pathway | N | Spearman rho | MAE |
|---------|---|-------------|-----|
| AMPK | 4 | 0.400 | 1.065 |
| PI3K | 5 | -0.700 | 1.106 |
| SIRT | 5 | -0.200 | 0.773 |
| mTOR | 59 | -0.014 | 0.698 |
| p53 | 60 | -0.197 | 0.951 |
| proteasome | 35 | 0.120 | 0.721 |
| telomerase | 2 | 1.000 | 0.751 |

## Model 2: HIGH Classifier

Binary: HIGH (divergence > 0.7 AND FC > 10x) vs not-HIGH.

| Metric | LOGO CV | Stratified 5-Fold CV |
|--------|---------|---------------------|
| ROC-AUC | 0.833 | 0.919 |
| Precision | 0.623 | 0.740 |
| Recall | 0.524 | 0.857 |
| F1 | 0.569 | 0.794 |

## Model 3: Global Model

Global-style model (pair chirality FP + target one-hot) trained on longevity data, LOO-CV.

- Mode: global_style_loo
- Valid predictions: 170
- Spearman rho: 0.549 (p=8.7e-15)
- MAE: 0.553
- R²: 0.335

## SHAP Feature Importance

### Regressor — Top 10 features (mean |SHAP|)

| Feature | Importance |
|---------|------------|
| fp_absdiff_76 | 0.1083 |
| QED | 0.0978 |
| FracCSP3 | 0.0899 |
| fp_absdiff_113 | 0.0707 |
| fp_absdiff_96 | 0.0513 |
| fp_b_57 | 0.0428 |
| fp_prod_72 | 0.0394 |
| fp_absdiff_91 | 0.0360 |
| fp_absdiff_19 | 0.0353 |
| fp_prod_12 | 0.0340 |

### Classifier — Top 10 features (mean |SHAP|)

| Feature | Importance |
|---------|------------|
| divergence | 5.1436 |
| fp_prod_14 | 1.0260 |
| fp_absdiff_91 | 0.5588 |
| fp_prod_38 | 0.3989 |
| fp_prod_26 | 0.2009 |
| fp_prod_23 | 0.1761 |
| fp_b_67 | 0.1730 |
| fp_prod_106 | 0.1716 |
| fp_prod_103 | 0.1676 |
| fp_prod_100 | 0.1657 |

## Golden Candidates (36)

HIGH + drug-like (Lipinski violations <= 1, QED > 0.3).

| # | ID_A | Pathway | Actual FC | Pred log₁₀FC | HIGH Prob | Global log₁₀FC | QED | Div |
|---|------|---------|-----------|-------------|-----------|---------------|-----|-----|
| 1 | CHEMBL4759169 | PI3K | 3551.1x | 3.50 | 1.00 | 3.48 | 0.47 | 0.92 |
| 2 | CHEMBL4460164 | mTOR | 1910.8x | 3.26 | 1.00 | 3.18 | 0.73 | 0.95 |
| 3 | CHEMBL3913159 | p53 | 1698.2x | 3.21 | 1.00 | 3.20 | 0.35 | 0.74 |
| 4 | CHEMBL3914063 | p53 | 1349.0x | 3.11 | 1.00 | 3.06 | 0.35 | 0.97 |
| 5 | CHEMBL3894082 | p53 | 1174.9x | 3.07 | 1.00 | 3.05 | 0.34 | 0.98 |
| 6 | CHEMBL5205717 | proteasome | 1089.2x | 3.01 | 1.00 | 2.99 | 0.69 | 0.71 |
| 7 | CHEMBL4285202 | mTOR | 937.5x | 2.97 | 1.00 | 2.97 | 0.75 | 0.98 |
| 8 | CHEMBL4112168 | p53 | 871.0x | 2.93 | 1.00 | 2.94 | 0.34 | 0.97 |
| 9 | CHEMBL4784596 | PI3K | 789.1x | 2.93 | 1.00 | 2.89 | 0.47 | 0.90 |
| 10 | CHEMBL1242468 | mTOR | 700.0x | 2.84 | 1.00 | 2.79 | 0.67 | 0.82 |
| 11 | CHEMBL3897997 | p53 | 512.9x | 2.72 | 1.00 | 2.71 | 0.38 | 0.76 |
| 12 | CHEMBL3909949 | p53 | 446.7x | 2.63 | 1.00 | 2.62 | 0.50 | 0.90 |
| 13 | CHEMBL371432 | SIRT | 363.1x | 2.55 | 1.00 | 2.55 | 0.76 | 0.87 |
| 14 | CHEMBL3924973 | p53 | 298.5x | 2.49 | 1.00 | 2.49 | 0.38 | 0.95 |
| 15 | CHEMBL4594206 | mTOR | 87.5x | 1.93 | 1.00 | 1.90 | 0.84 | 0.98 |
| 16 | CHEMBL4562100 | mTOR | 76.2x | 1.87 | 1.00 | 1.85 | 0.84 | 0.96 |
| 17 | CHEMBL5933575 | mTOR | 64.6x | 1.79 | 1.00 | 1.79 | 0.75 | 0.82 |
| 18 | CHEMBL5188029 | proteasome | 61.3x | 1.78 | 1.00 | 1.78 | 0.71 | 0.74 |
| 19 | CHEMBL4438609 | mTOR | 61.3x | 1.76 | 1.00 | 1.78 | 0.80 | 0.88 |
| 20 | CHEMBL5763596 | mTOR | 55.0x | 1.74 | 1.00 | 1.75 | 0.37 | 0.97 |
| 21 | CHEMBL4288210 | mTOR | 49.8x | 1.69 | 1.00 | 1.70 | 0.79 | 0.86 |
| 22 | CHEMBL5200546 | proteasome | 43.8x | 1.63 | 1.00 | 1.62 | 0.71 | 0.82 |
| 23 | CHEMBL5747666 | mTOR | 40.0x | 1.59 | 1.00 | 1.56 | 0.54 | 0.76 |
| 24 | CHEMBL1774364 | mTOR | 39.5x | 1.60 | 1.00 | 1.56 | 0.65 | 0.93 |
| 25 | CHEMBL3769991 | SIRT | 37.2x | 1.56 | 1.00 | 1.58 | 0.32 | 0.83 |
| 26 | CHEMBL6037298 | proteasome | 36.4x | 1.56 | 1.00 | 1.56 | 0.48 | 0.77 |
| 27 | CHEMBL3653613 | mTOR | 30.9x | 1.51 | 1.00 | 1.45 | 0.41 | 0.95 |
| 28 | CHEMBL575989 | mTOR | 28.3x | 1.45 | 1.00 | 1.44 | 0.42 | 0.85 |
| 29 | CHEMBL575345 | mTOR | 27.5x | 1.43 | 1.00 | 1.43 | 0.59 | 0.88 |
| 30 | CHEMBL1242748 | mTOR | 27.4x | 1.43 | 1.00 | 1.44 | 0.75 | 0.90 |
| 31 | CHEMBL1774382 | mTOR | 18.9x | 1.28 | 1.00 | 1.25 | 0.61 | 0.77 |
| 32 | CHEMBL1774378 | mTOR | 18.1x | 1.26 | 1.00 | 1.26 | 0.69 | 0.95 |
| 33 | CHEMBL3645819 | mTOR | 15.5x | 1.17 | 1.00 | 1.23 | 0.57 | 0.94 |
| 34 | CHEMBL1088971 | mTOR | 13.8x | 1.16 | 1.00 | 1.15 | 0.45 | 0.72 |
| 35 | CHEMBL4457600 | mTOR | 12.9x | 1.09 | 1.00 | 1.11 | 0.79 | 0.94 |
| 36 | CHEMBL5416576 | p53 | 11.8x | 1.06 | 1.00 | 1.10 | 0.51 | 0.95 |

## Scientific Context

- Atlas v8 showed standalone FP divergence cannot predict FC (rho=-0.008, n=1998)
- This pipeline tests whether **FP + pathway/target context** rescues prediction
- Leave-one-pathway-out CV prevents pathway information leakage
- No fake augmentation — each pair produces exactly 1 feature vector (N=170)
