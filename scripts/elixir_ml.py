#!/usr/bin/env python3
"""ElixirML — Longevity Chirality ML Pipeline.

Scientifically valid pipeline for predicting chirality-driven fold-change
in longevity-related drug targets. Three models:
  1. FC Regressor — predict log10(fold_change) from FP pairs + pathway context
  2. HIGH Classifier — binary classify HIGH (div>0.7 AND FC>10x) vs not
  3. Global Model — evaluate pre-trained 138K-record ChEMBL model on longevity pairs
     (falls back to LOO-CV global-style model if PKL not on disk)

Usage:
  python scripts/elixir_ml.py [--data chembl_work/longevity_full_analysis.json]
                               [--global-model targets_work/global_model.pkl]
                               [--output-dir elixir_work]
                               [--no-shap]
                               [--cv-folds 5]
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import (
    cross_val_predict, cross_val_score,
    LeaveOneGroupOut, StratifiedKFold, KFold,
)
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    roc_auc_score, precision_recall_fscore_support, f1_score,
)
from sklearn.dummy import DummyRegressor

from rdkit import Chem
from rdkit.Chem import Descriptors

# Suppress RDKit warnings
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

# ZynerjiChirality imports
from zynerji_chirality.chirality.fingerprint import (
    chirality_differential_fingerprint,
    chirality_fingerprint,
    fingerprint_similarity,
)
from zynerji_chirality.ml.features import _rdkit_descriptors, descriptor_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("elixir_ml")

# Pathway canonical order (for one-hot encoding)
# Auto-detected from data in main(); set here as default for longevity
PATHWAYS = ["p53", "mTOR", "PI3K", "SIRT", "proteasome", "telomerase", "AMPK"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(json_path: str) -> pd.DataFrame:
    """Load longevity pairs from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    log.info("Loaded %d pairs from %s", len(df), json_path)
    log.info("Pathways: %s", dict(df['tag'].value_counts()))
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def pathway_onehot(tag: str) -> np.ndarray:
    """One-hot encode pathway tag (7d)."""
    vec = np.zeros(len(PATHWAYS), dtype=np.float64)
    if tag in PATHWAYS:
        vec[PATHWAYS.index(tag)] = 1.0
    return vec


def compute_pair_features(row: dict) -> np.ndarray | None:
    """Compute feature vector for one enantiomer pair.

    Returns 532d vector:
      - Differential FP pair: [fp_a(128), fp_b(128), product(128), |diff|(128)] = 512d
      - Molecular descriptors: 10d
      - Pathway one-hot: 7d
      - Divergence: 1d
      - Chiral centers: 1d
      - QED: 1d
    """
    try:
        fp_a = chirality_differential_fingerprint(row['smi_a'])
        fp_b = chirality_differential_fingerprint(row['smi_b'])
    except Exception as e:
        log.warning("FP failed for %s / %s: %s", row['id_a'], row['id_b'], e)
        return None

    product = fp_a * fp_b
    abs_diff = np.abs(fp_a - fp_b)
    fp_pair = np.concatenate([fp_a, fp_b, product, abs_diff])  # 512d

    # Molecular descriptors from data (already computed, use them directly)
    descriptors = np.array([
        row['mw'], row['logp'], row['tpsa'],
        row['hbd'], row['hba'], row['rotb'],
        0.0,  # n_aromatic_rings not in data, use 0
        row['n_rings'], 0.0,  # frac_csp3 not in data
        row['n_heavy'],
    ], dtype=np.float64)

    # If we have the SMILES, compute the missing descriptors properly
    mol = Chem.MolFromSmiles(row['smi_a'])
    if mol is not None:
        try:
            descriptors[6] = float(Descriptors.NumAromaticRings(mol))
            descriptors[8] = float(Descriptors.FractionCSP3(mol))
        except Exception:
            pass

    pw = pathway_onehot(row['tag'])         # 7d
    div = np.array([row['div']])            # 1d
    centers = np.array([row['n_centers']])   # 1d
    qed = np.array([row['qed']])            # 1d

    return np.concatenate([fp_pair, descriptors, pw, div, centers, qed])


def compute_all_features(df: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    """Compute features for all pairs. Returns (X, valid_indices)."""
    features = []
    valid_idx = []
    for i, row in df.iterrows():
        feat = compute_pair_features(row.to_dict())
        if feat is not None:
            features.append(feat)
            valid_idx.append(i)
        if (len(valid_idx)) % 50 == 0 and len(valid_idx) > 0:
            log.info("  Fingerprinted %d / %d pairs...", len(valid_idx), len(df))

    X = np.array(features)
    log.info("Feature matrix: %s (%.1f KB)", X.shape, X.nbytes / 1024)
    return X, valid_idx


def feature_names() -> list[str]:
    """Return ordered feature names for SHAP interpretability."""
    names = []
    for prefix in ['fp_a', 'fp_b', 'fp_prod', 'fp_absdiff']:
        names.extend([f"{prefix}_{i}" for i in range(128)])
    names.extend(["MW", "LogP", "TPSA", "HBD", "HBA", "RotB",
                   "ArRings", "Rings", "FracCSP3", "HeavyAtoms"])
    names.extend([f"pw_{p}" for p in PATHWAYS])
    names.extend(["divergence", "n_chiral_centers", "QED"])
    return names


# ---------------------------------------------------------------------------
# Model 1: FC Regressor
# ---------------------------------------------------------------------------

def train_fc_regressor(X, y, groups, cv_folds=5):
    """Train fold-change regressor with cross-validation.

    Returns (model, metrics_dict, cv_predictions).
    """
    log.info("=== Model 1: FC Regressor ===")
    model = HistGradientBoostingRegressor(
        max_iter=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=5, random_state=42,
    )

    # Leave-one-pathway-out CV predictions
    logo = LeaveOneGroupOut()
    y_pred_logo = cross_val_predict(model, X, y, cv=logo, groups=groups)
    rho_logo, p_logo = spearmanr(y, y_pred_logo)
    mae_logo = mean_absolute_error(y, y_pred_logo)
    r2_logo = r2_score(y, y_pred_logo)

    # K-fold CV predictions
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    y_pred_kf = cross_val_predict(model, X, y, cv=kf)
    rho_kf, p_kf = spearmanr(y, y_pred_kf)
    mae_kf = mean_absolute_error(y, y_pred_kf)
    r2_kf = r2_score(y, y_pred_kf)

    # Baselines
    dummy_mean = DummyRegressor(strategy='mean')
    y_pred_dummy = cross_val_predict(dummy_mean, X, y, cv=kf)
    mae_baseline = mean_absolute_error(y, y_pred_dummy)

    # Divergence-only baseline (512 FP + 10 descriptors + len(PATHWAYS) = divergence index)
    div_idx = 512 + 10 + len(PATHWAYS)
    X_div = X[:, div_idx:div_idx+1]
    div_model = HistGradientBoostingRegressor(
        max_iter=50, max_depth=2, random_state=42,
    )
    y_pred_div = cross_val_predict(div_model, X_div, y, cv=kf)
    rho_div, _ = spearmanr(y, y_pred_div)
    mae_div = mean_absolute_error(y, y_pred_div)

    # Per-pathway performance (LOGO)
    pathway_metrics = {}
    unique_groups = np.unique(groups)
    for g in unique_groups:
        mask = groups == g
        if mask.sum() < 2:
            continue
        rho_g, _ = spearmanr(y[mask], y_pred_logo[mask])
        mae_g = mean_absolute_error(y[mask], y_pred_logo[mask])
        pathway_metrics[g] = {"n": int(mask.sum()), "rho": rho_g, "mae": mae_g}

    # Fit final model on all data
    model.fit(X, y)

    metrics = {
        "logo": {"rho": rho_logo, "p": p_logo, "mae": mae_logo, "r2": r2_logo},
        "kfold": {"rho": rho_kf, "p": p_kf, "mae": mae_kf, "r2": r2_kf},
        "baseline_mean_mae": mae_baseline,
        "baseline_div_only": {"rho": rho_div, "mae": mae_div},
        "per_pathway": pathway_metrics,
    }

    log.info("  LOGO: rho=%.3f (p=%.1e), MAE=%.3f, R2=%.3f",
             rho_logo, p_logo, mae_logo, r2_logo)
    log.info("  KFold: rho=%.3f (p=%.1e), MAE=%.3f, R2=%.3f",
             rho_kf, p_kf, mae_kf, r2_kf)
    log.info("  Baselines — mean MAE=%.3f, div-only rho=%.3f MAE=%.3f",
             mae_baseline, rho_div, mae_div)
    for pw, m in pathway_metrics.items():
        log.info("    %s (n=%d): rho=%.3f, MAE=%.3f", pw, m['n'], m['rho'], m['mae'])

    return model, metrics, y_pred_logo


# ---------------------------------------------------------------------------
# Model 2: HIGH Classifier
# ---------------------------------------------------------------------------

def train_high_classifier(X, y_clf, groups, cv_folds=5):
    """Train HIGH binary classifier with cross-validation.

    Returns (model, metrics_dict, cv_probabilities).
    """
    log.info("=== Model 2: HIGH Classifier ===")
    n_pos = y_clf.sum()
    n_neg = len(y_clf) - n_pos
    log.info("  Class balance: %d HIGH, %d not-HIGH (%.1f%%)", n_pos, n_neg, 100*n_pos/len(y_clf))

    model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=5, random_state=42,
    )

    # LOGO CV
    logo = LeaveOneGroupOut()
    try:
        logo_proba = cross_val_predict(model, X, y_clf, cv=logo, groups=groups, method='predict_proba')
        logo_pred = (logo_proba[:, 1] >= 0.5).astype(int)
        auc_logo = roc_auc_score(y_clf, logo_proba[:, 1])
        prec_logo, rec_logo, f1_logo, _ = precision_recall_fscore_support(
            y_clf, logo_pred, average='binary', zero_division=0)
    except Exception as e:
        log.warning("  LOGO CV failed (likely single-class fold): %s", e)
        auc_logo = prec_logo = rec_logo = f1_logo = float('nan')
        logo_proba = None

    # Stratified K-fold CV
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    kf_proba = cross_val_predict(model, X, y_clf, cv=skf, method='predict_proba')
    kf_pred = (kf_proba[:, 1] >= 0.5).astype(int)
    auc_kf = roc_auc_score(y_clf, kf_proba[:, 1])
    prec_kf, rec_kf, f1_kf, _ = precision_recall_fscore_support(
        y_clf, kf_pred, average='binary', zero_division=0)

    # Fit final model
    model.fit(X, y_clf)

    metrics = {
        "logo": {"auc": auc_logo, "precision": prec_logo, "recall": rec_logo, "f1": f1_logo},
        "kfold": {"auc": auc_kf, "precision": prec_kf, "recall": rec_kf, "f1": f1_kf},
    }

    log.info("  LOGO: AUC=%.3f, P=%.3f, R=%.3f, F1=%.3f",
             auc_logo, prec_logo, rec_logo, f1_logo)
    log.info("  KFold: AUC=%.3f, P=%.3f, R=%.3f, F1=%.3f",
             auc_kf, prec_kf, rec_kf, f1_kf)

    return model, metrics, kf_proba[:, 1]


# ---------------------------------------------------------------------------
# Model 3: Global Model Evaluation
# ---------------------------------------------------------------------------

def evaluate_global_model(model_path: str, df: pd.DataFrame):
    """Evaluate pre-trained global model on longevity pairs.

    If the PKL exists, loads the 138K-record GlobalChiralityPredictor.
    If not, trains a global-style model (pair chirality_fingerprint + target one-hot)
    on the 170 longevity pairs using LOO-CV.

    Returns (metrics_dict, predictions_array, mode_str).
    """
    log.info("=== Model 3: Global Model Evaluation ===")

    y_true = np.log10(df['fc'].values)

    if os.path.exists(model_path):
        return _eval_pretrained_global(model_path, df, y_true)
    else:
        log.info("  PKL not found at %s — training global-style model on longevity data", model_path)
        return _train_global_style(df, y_true)


def _eval_pretrained_global(model_path: str, df: pd.DataFrame, y_true: np.ndarray):
    """Load and evaluate the pre-trained GlobalChiralityPredictor."""
    from zynerji_chirality.targets.global_model import GlobalChiralityPredictor

    gm = GlobalChiralityPredictor()
    gm.load(model_path)
    log.info("  Loaded global model: %d targets", gm._n_targets)

    preds = []
    for _, row in df.iterrows():
        # Extract target ChEMBL ID from description if available
        # The longevity data doesn't have target_chembl_id, use empty string
        # (global model will use zero-vector for unknown targets)
        try:
            pred = gm.predict_for_target(row['smi_a'], row['smi_b'], "")
            preds.append(np.log10(max(pred.predicted_fold_change, 1.01)))
        except Exception:
            preds.append(np.nan)

    preds = np.array(preds)
    valid = ~np.isnan(preds)
    if valid.sum() < 5:
        log.warning("  Too few valid predictions (%d)", valid.sum())
        return {"mode": "pretrained", "n_valid": int(valid.sum())}, preds, "pretrained"

    rho, p = spearmanr(y_true[valid], preds[valid])
    mae = mean_absolute_error(y_true[valid], preds[valid])
    r2 = r2_score(y_true[valid], preds[valid])

    metrics = {
        "mode": "pretrained",
        "n_valid": int(valid.sum()),
        "rho": rho, "p": p, "mae": mae, "r2": r2,
    }
    log.info("  Pretrained global: rho=%.3f, MAE=%.3f, R2=%.3f (%d/%d valid)",
             rho, mae, r2, valid.sum(), len(df))
    return metrics, preds, "pretrained"


def _train_global_style(df: pd.DataFrame, y_true: np.ndarray):
    """Train a global-style model on longevity pairs using LOO-CV.

    Uses chirality_fingerprint (not differential) + target name one-hot,
    matching the GlobalChiralityPredictor architecture.
    """
    # Build features: [fp_a(128), fp_b(128), product(128), |diff|(128), target_onehot]
    targets = sorted(df['target'].unique())
    target_map = {t: i for i, t in enumerate(targets)}
    n_targets = len(targets)
    log.info("  Targets in longevity data: %d", n_targets)

    X_list = []
    valid_idx = []
    for i, row in df.iterrows():
        try:
            fp_a = chirality_fingerprint(row['smi_a'])
            fp_b = chirality_fingerprint(row['smi_b'])
        except Exception:
            continue
        product = fp_a * fp_b
        diff = np.abs(fp_a - fp_b)
        pair_fp = np.concatenate([fp_a, fp_b, product, diff])  # 512d

        target_enc = np.zeros(n_targets, dtype=np.float64)
        target_enc[target_map[row['target']]] = 1.0

        X_list.append(np.concatenate([pair_fp, target_enc]))
        valid_idx.append(i)

    X = np.array(X_list)
    y = y_true[valid_idx]
    log.info("  Global-style features: %s", X.shape)

    model = HistGradientBoostingRegressor(
        max_iter=150, max_depth=4, learning_rate=0.05,
        min_samples_leaf=3, random_state=42,
    )

    # LOO-CV (small dataset, full leave-one-out is tractable)
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo)

    rho, p = spearmanr(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Fit final model
    model.fit(X, y)

    # Generate predictions for all rows (including any that failed FP)
    all_preds = np.full(len(df), np.nan)
    all_preds[valid_idx] = model.predict(X)

    metrics = {
        "mode": "global_style_loo",
        "n_valid": len(valid_idx),
        "n_targets": n_targets,
        "rho": rho, "p": p, "mae": mae, "r2": r2,
    }
    log.info("  Global-style LOO: rho=%.3f (p=%.1e), MAE=%.3f, R2=%.3f",
             rho, p, mae, r2)
    return metrics, all_preds, "global_style_loo"


# ---------------------------------------------------------------------------
# SHAP Analysis
# ---------------------------------------------------------------------------

def run_shap_analysis(model, X, output_dir, name, feat_names):
    """Run SHAP TreeExplainer and save summary plot."""
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — skipping SHAP analysis")
        return None

    log.info("Running SHAP for %s...", name)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feat_names,
                      show=False, max_display=20)
    plt.tight_layout()
    path = os.path.join(output_dir, f"shap_summary_{name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info("  Saved %s", path)

    # Bar plot (mean |SHAP|)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feat_names,
                      plot_type='bar', show=False, max_display=20)
    plt.tight_layout()
    path_bar = os.path.join(output_dir, f"shap_bar_{name}.png")
    plt.savefig(path_bar, dpi=150, bbox_inches='tight')
    plt.close()
    log.info("  Saved %s", path_bar)

    # Top features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    top_features = [(feat_names[i], float(mean_abs[i])) for i in top_idx]
    log.info("  Top SHAP features (%s):", name)
    for fname, val in top_features:
        log.info("    %s: %.4f", fname, val)

    return {"top_features": top_features, "shap_values": shap_values}


def get_shap_top3(shap_values, feat_names, idx):
    """Get top-3 SHAP features for a single sample."""
    if shap_values is None:
        return []
    sv = shap_values[idx]
    top = np.argsort(np.abs(sv))[::-1][:3]
    return [(feat_names[t], float(sv[t])) for t in top]


# ---------------------------------------------------------------------------
# Golden Candidates Report
# ---------------------------------------------------------------------------

def identify_golden(df: pd.DataFrame) -> list[int]:
    """Identify golden candidates: HIGH + drug-like."""
    golden = []
    for i, row in df.iterrows():
        is_high = row['div'] > 0.7 and row['fc'] > 10
        is_druglike = row['lip_violations'] <= 1 and row['qed'] > 0.3
        if is_high and is_druglike:
            golden.append(i)
    return golden


def generate_golden_report(df, X, valid_idx, model_reg, model_clf,
                           global_preds, shap_reg, shap_clf, feat_names):
    """Generate predictions table for golden candidates."""
    golden_idx = identify_golden(df)
    log.info("Golden candidates: %d", len(golden_idx))

    # Map df index -> X row index
    idx_map = {v: i for i, v in enumerate(valid_idx)}

    rows = []
    for df_idx in golden_idx:
        row = df.loc[df_idx]
        x_idx = idx_map.get(df_idx)
        if x_idx is None:
            continue

        fc_pred = float(model_reg.predict(X[x_idx:x_idx+1])[0])
        high_prob = float(model_clf.predict_proba(X[x_idx:x_idx+1])[0, 1])
        global_fc = float(global_preds[df_idx]) if not np.isnan(global_preds[df_idx]) else None

        shap_top3_reg = get_shap_top3(
            shap_reg['shap_values'] if shap_reg else None, feat_names, x_idx)
        shap_top3_clf = get_shap_top3(
            shap_clf['shap_values'] if shap_clf else None, feat_names, x_idx)

        rows.append({
            "id_a": row['id_a'],
            "id_b": row['id_b'],
            "target": row['target'],
            "pathway": row['tag'],
            "actual_fc": float(row['fc']),
            "actual_log_fc": float(np.log10(row['fc'])),
            "divergence": float(row['div']),
            "qed": float(row['qed']),
            "n_centers": int(row['n_centers']),
            "predicted_log_fc": fc_pred,
            "high_probability": high_prob,
            "global_predicted_log_fc": global_fc,
            "shap_top3_regressor": shap_top3_reg,
            "shap_top3_classifier": shap_top3_clf,
        })

    return sorted(rows, key=lambda r: r['actual_fc'], reverse=True)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_markdown_report(output_dir, reg_metrics, clf_metrics, global_metrics,
                          global_mode, golden_rows, n_total, shap_reg, shap_clf):
    """Write elixir_results.md."""
    path = os.path.join(output_dir, "elixir_results.md")
    with open(path, "w") as f:
        f.write("# ElixirML — Longevity Chirality ML Results\n\n")
        f.write(f"**Dataset**: {n_total} enantiomer pairs across {len(PATHWAYS)} pathways\n\n")

        # Model 1
        f.write("## Model 1: Fold-Change Regressor\n\n")
        f.write("Predicts log10(fold_change) from differential FP pairs + pathway context.\n\n")
        f.write("| Metric | LOGO CV | 5-Fold CV |\n")
        f.write("|--------|---------|----------|\n")
        f.write(f"| Spearman rho | {reg_metrics['logo']['rho']:.3f} (p={reg_metrics['logo']['p']:.1e}) "
                f"| {reg_metrics['kfold']['rho']:.3f} (p={reg_metrics['kfold']['p']:.1e}) |\n")
        f.write(f"| MAE | {reg_metrics['logo']['mae']:.3f} | {reg_metrics['kfold']['mae']:.3f} |\n")
        f.write(f"| R² | {reg_metrics['logo']['r2']:.3f} | {reg_metrics['kfold']['r2']:.3f} |\n\n")

        f.write("**Baselines:**\n")
        f.write(f"- Mean predictor MAE: {reg_metrics['baseline_mean_mae']:.3f}\n")
        f.write(f"- Divergence-only: rho={reg_metrics['baseline_div_only']['rho']:.3f}, "
                f"MAE={reg_metrics['baseline_div_only']['mae']:.3f}\n\n")

        if reg_metrics['per_pathway']:
            f.write("**Per-pathway (LOGO):**\n\n")
            f.write("| Pathway | N | Spearman rho | MAE |\n")
            f.write("|---------|---|-------------|-----|\n")
            for pw, m in sorted(reg_metrics['per_pathway'].items()):
                rho_str = f"{m['rho']:.3f}" if not np.isnan(m['rho']) else "N/A"
                f.write(f"| {pw} | {m['n']} | {rho_str} | {m['mae']:.3f} |\n")
            f.write("\n")

        # Model 2
        f.write("## Model 2: HIGH Classifier\n\n")
        f.write("Binary: HIGH (divergence > 0.7 AND FC > 10x) vs not-HIGH.\n\n")
        f.write("| Metric | LOGO CV | Stratified 5-Fold CV |\n")
        f.write("|--------|---------|---------------------|\n")
        logo = clf_metrics['logo']
        kfold = clf_metrics['kfold']
        f.write(f"| ROC-AUC | {logo['auc']:.3f} | {kfold['auc']:.3f} |\n")
        f.write(f"| Precision | {logo['precision']:.3f} | {kfold['precision']:.3f} |\n")
        f.write(f"| Recall | {logo['recall']:.3f} | {kfold['recall']:.3f} |\n")
        f.write(f"| F1 | {logo['f1']:.3f} | {kfold['f1']:.3f} |\n\n")

        # Model 3
        f.write("## Model 3: Global Model\n\n")
        if global_mode == "pretrained":
            f.write("Pre-trained on 138K ChEMBL records (3551 targets).\n\n")
        else:
            f.write("Global-style model (pair chirality FP + target one-hot) trained on longevity data, LOO-CV.\n\n")
        gm = global_metrics
        f.write(f"- Mode: {gm['mode']}\n")
        f.write(f"- Valid predictions: {gm['n_valid']}\n")
        if 'rho' in gm:
            f.write(f"- Spearman rho: {gm['rho']:.3f} (p={gm['p']:.1e})\n")
            f.write(f"- MAE: {gm['mae']:.3f}\n")
            f.write(f"- R²: {gm['r2']:.3f}\n")
        f.write("\n")

        # SHAP
        if shap_reg:
            f.write("## SHAP Feature Importance\n\n")
            f.write("### Regressor — Top 10 features (mean |SHAP|)\n\n")
            f.write("| Feature | Importance |\n")
            f.write("|---------|------------|\n")
            for fname, val in shap_reg['top_features']:
                f.write(f"| {fname} | {val:.4f} |\n")
            f.write("\n")

        if shap_clf:
            f.write("### Classifier — Top 10 features (mean |SHAP|)\n\n")
            f.write("| Feature | Importance |\n")
            f.write("|---------|------------|\n")
            for fname, val in shap_clf['top_features']:
                f.write(f"| {fname} | {val:.4f} |\n")
            f.write("\n")

        # Golden candidates
        f.write(f"## Golden Candidates ({len(golden_rows)})\n\n")
        f.write("HIGH + drug-like (Lipinski violations <= 1, QED > 0.3).\n\n")
        f.write("| # | ID_A | Pathway | Actual FC | Pred log₁₀FC | HIGH Prob | Global log₁₀FC | QED | Div |\n")
        f.write("|---|------|---------|-----------|-------------|-----------|---------------|-----|-----|\n")
        for i, r in enumerate(golden_rows, 1):
            gfc = f"{r['global_predicted_log_fc']:.2f}" if r['global_predicted_log_fc'] is not None else "—"
            f.write(f"| {i} | {r['id_a']} | {r['pathway']} | {r['actual_fc']:.1f}x "
                    f"| {r['predicted_log_fc']:.2f} | {r['high_probability']:.2f} "
                    f"| {gfc} | {r['qed']:.2f} | {r['divergence']:.2f} |\n")
        f.write("\n")

        # Atlas v8 context
        f.write("## Scientific Context\n\n")
        f.write("- Atlas v8 showed standalone FP divergence cannot predict FC (rho=-0.008, n=1998)\n")
        f.write("- This pipeline tests whether **FP + pathway/target context** rescues prediction\n")
        f.write("- Leave-one-pathway-out CV prevents pathway information leakage\n")
        f.write("- No fake augmentation — each pair produces exactly 1 feature vector (N=170)\n")

    log.info("Wrote %s", path)


def write_json_results(output_dir, reg_metrics, clf_metrics, global_metrics,
                       global_mode, golden_rows):
    """Write elixir_results.json."""
    path = os.path.join(output_dir, "elixir_results.json")

    def sanitize(obj):
        """Make numpy types JSON-serializable."""
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            v = float(obj)
            return None if np.isnan(v) else v
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    results = {
        "model_1_regressor": sanitize(reg_metrics),
        "model_2_classifier": sanitize(clf_metrics),
        "model_3_global": sanitize(global_metrics),
        "global_mode": global_mode,
        "golden_candidates": sanitize(golden_rows),
    }

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ElixirML — Longevity Chirality ML Pipeline")
    parser.add_argument("--data", default="chembl_work/longevity_full_analysis.json",
                        help="Path to longevity pairs JSON")
    parser.add_argument("--global-model", default="targets_work/global_model.pkl",
                        help="Path to pre-trained global model PKL")
    parser.add_argument("--output-dir", default="elixir_work",
                        help="Output directory for results")
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip SHAP analysis (faster)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of cross-validation folds")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # Load data
    df = load_data(args.data)

    # Auto-detect pathways from data
    global PATHWAYS
    detected = sorted(df['tag'].unique().tolist())
    if detected != PATHWAYS:
        log.info("Auto-detected pathways: %s", detected)
        PATHWAYS = detected

    # Feature engineering
    log.info("Computing features...")
    X, valid_idx = compute_all_features(df)
    df_valid = df.loc[valid_idx].reset_index(drop=True)

    # Labels
    y_reg = np.log10(df_valid['fc'].values)
    y_clf = ((df_valid['div'] > 0.7) & (df_valid['fc'] > 10)).astype(int).values
    groups = df_valid['tag'].values

    feat_names = feature_names()

    # Model 1: FC Regressor
    model_reg, reg_metrics, y_pred_logo = train_fc_regressor(
        X, y_reg, groups, args.cv_folds)

    # Model 2: HIGH Classifier
    model_clf, clf_metrics, clf_proba = train_high_classifier(
        X, y_clf, groups, args.cv_folds)

    # Model 3: Global Model (always runs — pretrained or global-style)
    global_metrics, global_preds, global_mode = evaluate_global_model(
        args.global_model, df, )

    # Expand global_preds to full df index space
    # global_preds is indexed by df original index (from evaluate_global_model)

    # SHAP
    shap_reg = shap_clf = None
    if not args.no_shap:
        shap_reg = run_shap_analysis(model_reg, X, args.output_dir, "regressor", feat_names)
        shap_clf = run_shap_analysis(model_clf, X, args.output_dir, "classifier", feat_names)

    # Golden candidates
    golden_rows = generate_golden_report(
        df, X, valid_idx, model_reg, model_clf,
        global_preds, shap_reg, shap_clf, feat_names)

    # Write reports
    write_markdown_report(args.output_dir, reg_metrics, clf_metrics, global_metrics,
                          global_mode, golden_rows, len(df), shap_reg, shap_clf)
    write_json_results(args.output_dir, reg_metrics, clf_metrics, global_metrics,
                       global_mode, golden_rows)

    elapsed = time.time() - t0
    log.info("ElixirML complete in %.1fs — results in %s/", elapsed, args.output_dir)


if __name__ == "__main__":
    main()
