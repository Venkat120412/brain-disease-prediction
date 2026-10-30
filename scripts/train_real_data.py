"""
Train all 4 brain-disease models using REAL datasets from Kaggle / UCI.

  Stroke       — Kaggle fedesoriano/stroke-prediction-dataset     (5 110 rows)
  Alzheimer's  — Kaggle rabieelkharoua/alzheimers-disease-dataset  (2 149 rows)
  Parkinson's  — UCI Parkinson's Data Set                          (195 rows)
  Brain Tumor  — SYNTHETIC (no real tabular dataset on Kaggle)

The existing ml_engine.train_disease() already prevents data leakage:
  1. train/test split FIRST
  2. preprocessor fit on TRAIN only, transform on TEST
  3. feature selector fit on TRAIN only, transform on TEST
  4. models fit on TRAIN, evaluated on TEST

Usage:  python scripts/train_real_data.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pandas as pd
from app.core.config import settings, DISEASES
from app.services.disease_configs import DISEASE_CONFIGS
from app.services.ml_engine import train_disease

# ── per-disease hyperparameters ───────────────────────────────────────────────
# Alzheimer's has 32 features; default n_features=8 would discard too many.
TRAIN_CONFIGS = {
    "stroke":       {"n_features": 8},    # 10 input features → 8 selected
    "alzheimers":   {"n_features": 15},   # 32 input features → 15 selected
    "parkinsons":   {"n_features": 12},    # 22 input features → 12 selected
    "brain_tumor":  {"n_features": 8},    # 12 input features → 8 selected
}

REAL_DATA_NOTE = {
    "stroke":      "Real data — Kaggle fedesoriano/stroke-prediction-dataset (5110 rows)",
    "alzheimers":  "Real data — Kaggle rabieelkharoua/alzheimers-disease-dataset (2149 rows, CC BY 4.0)",
    "parkinsons":  "Real data — UCI Parkinson's Data Set (195 voice recordings, 31 subjects)",
    "brain_tumor": "SYNTHETIC data — no real tabular brain-tumor dataset available on Kaggle",
}


def main():
    print("=" * 72)
    print("  TRAINING BRAIN DISEASE MODELS WITH REAL-WORLD DATASETS")
    print("=" * 72)

    results_summary = {}

    for key in DISEASES:
        cfg = DISEASE_CONFIGS[key]
        path = settings.raw_data_dir / cfg.data_file
        if not path.exists():
            print(f"\n✗ {key}: data file missing at {path}")
            continue

        df = pd.read_csv(path)
        tc = TRAIN_CONFIGS.get(key, {"n_features": 8})
        data_note = REAL_DATA_NOTE.get(key, "")

        print(f"\n{'─' * 72}")
        print(f"  {cfg.label}")
        print(f"  Data: {data_note}")
        print(f"  Rows: {len(df)}  Target: {cfg.target_col}")
        print(f"  Positive: {(df[cfg.target_col]==1).sum()}  "
              f"Negative: {(df[cfg.target_col]==0).sum()}")
        print(f"  Features (numerical={len(cfg.numerical_cols)}, "
              f"binary={len(cfg.binary_cols)}, "
              f"categorical={len(cfg.categorical_cols)})")
        print(f"  Selecting top {tc['n_features']} features")
        print(f"{'─' * 72}")

        r = train_disease(df, cfg, n_features=tc["n_features"])

        print(f"\n  Logistic Regression:")
        lr = r["lr"]
        print(f"    Accuracy:  {lr['accuracy']:.4f}")
        print(f"    Precision: {lr['precision_weighted']:.4f}")
        print(f"    Recall:    {lr['recall_weighted']:.4f}")
        print(f"    F1-score:  {lr['f1_weighted']:.4f}")
        if lr.get("roc_auc"):
            print(f"    ROC-AUC:   {lr['roc_auc']:.4f}")
        print(f"    Confusion: {lr['confusion_matrix']}")

        print(f"\n  ANN (MLPClassifier):")
        ann = r["ann"]
        print(f"    Accuracy:  {ann['accuracy']:.4f}")
        print(f"    Precision: {ann['precision_weighted']:.4f}")
        print(f"    Recall:    {ann['recall_weighted']:.4f}")
        print(f"    F1-score:  {ann['f1_weighted']:.4f}")
        if ann.get("roc_auc"):
            print(f"    ROC-AUC:   {ann['roc_auc']:.4f}")
        print(f"    Confusion: {ann['confusion_matrix']}")

        print(f"\n  Selected features ({len(r['selected_features'])}):")
        for fi in r["feature_importance"][:8]:
            print(f"    #{fi['rank']:2d}  {fi['feature']:<30s}  importance={fi['importance']:.6f}")

        print(f"\n  Train rows: {r['train_rows']}  Test rows: {r['test_rows']}")

        results_summary[key] = {
            "label": cfg.label,
            "data_source": data_note,
            "rows": len(df),
            "train_rows": r["train_rows"],
            "test_rows": r["test_rows"],
            "lr": {
                "accuracy": lr["accuracy"],
                "precision": lr["precision_weighted"],
                "recall": lr["recall_weighted"],
                "f1": lr["f1_weighted"],
                "roc_auc": lr.get("roc_auc"),
            },
            "ann": {
                "accuracy": ann["accuracy"],
                "precision": ann["precision_weighted"],
                "recall": ann["recall_weighted"],
                "f1": ann["f1_weighted"],
                "roc_auc": ann.get("roc_auc"),
            },
            "selected_features": r["selected_features"],
        }

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 72}")
    print(f"{'Disease':<25s} {'Data':<12s} {'LR Acc':>8s} {'ANN Acc':>8s} "
          f"{'LR F1':>8s} {'ANN F1':>8s} {'LR AUC':>8s}")
    print(f"{'─' * 72}")
    for key, s in results_summary.items():
        data_tag = "REAL" if "Real" in s["data_source"] else "SYNTH"
        lr_auc = f"{s['lr']['roc_auc']:.4f}" if s["lr"]["roc_auc"] else "N/A"
        print(f"{s['label']:<25s} {data_tag:<12s} "
              f"{s['lr']['accuracy']:>8.4f} {s['ann']['accuracy']:>8.4f} "
              f"{s['lr']['f1']:>8.4f} {s['ann']['f1']:>8.4f} "
              f"{lr_auc:>8s}")

    # write summary JSON
    summary_path = settings.artifacts_dir / "real_data_training_summary.json"
    summary_path.write_text(json.dumps(results_summary, indent=2))
    print(f"\n  Summary saved to: {summary_path}")
    print(f"\n  Models saved to:  {settings.models_dir}")
    print(f"\n  Note: Brain tumor uses synthetic data (no real tabular source).")
    print(f"  Synthetic model backup: data/artifacts/models_synthetic_backup/")


if __name__ == "__main__":
    main()
