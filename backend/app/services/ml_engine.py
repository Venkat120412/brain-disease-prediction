"""
Core ML engine: trains, evaluates, saves, and loads models for every disease.
One Logistic Regression + one ANN (MLPClassifier) per disease.
"""
from __future__ import annotations
import json, logging, math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from app.core.config import settings
from app.services.disease_configs import DiseaseConfig

log = logging.getLogger("brain_disease.ml")

# ── helpers ───────────────────────────────────────────────────────────────────

def _clean(v):
    """Make a value JSON-safe (replace inf/nan with None)."""
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
    return v

def _clean_list(lst):
    return [_clean(x) for x in lst]

def _metrics(name, y_true, y_pred, y_prob=None, train_loss=None, val_loss=None):
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec  = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1   = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    roc_auc = fpr_l = tpr_l = thr_l = None
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            fpr, tpr, thr = roc_curve(y_true, y_prob)
            fpr_l = _clean_list(fpr.tolist())
            tpr_l = _clean_list(tpr.tolist())
            thr_l = [_clean(t) if math.isfinite(t) else 1.0 for t in thr.tolist()]
        except Exception:
            pass

    cm = confusion_matrix(y_true, y_pred).tolist()
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    class_rep = [
        {"label": k,
         "precision": round(float(v.get("precision",0)),4),
         "recall":    round(float(v.get("recall",0)),4),
         "f1_score":  round(float(v.get("f1-score",0)),4),
         "support":   int(v.get("support",0))}
        for k, v in rep.items() if isinstance(v, dict)
    ]

    return {
        "model_name":          name,
        "accuracy":            round(acc,4),
        "precision_weighted":  round(prec,4),
        "recall_weighted":     round(rec,4),
        "f1_weighted":         round(f1,4),
        "roc_auc":             round(roc_auc,4) if roc_auc else None,
        "confusion_matrix":    cm,
        "class_report":        class_rep,
        "fpr":                 fpr_l or [],
        "tpr":                 tpr_l or [],
        "roc_thresholds":      thr_l or [],
        "train_loss":          _clean_list(train_loss or []),
        "val_loss":            _clean_list(val_loss or []),
    }

# ── preprocessing ─────────────────────────────────────────────────────────────

def _build_preprocessor(cfg: DiseaseConfig):
    transformers = []
    if cfg.numerical_cols:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
        ]), cfg.numerical_cols))
    if cfg.categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cfg.categorical_cols))
    if cfg.binary_cols:
        transformers.append(("bin", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
        ]), cfg.binary_cols))
    return ColumnTransformer(transformers, remainder="drop")

def _feature_names(preprocessor, cfg: DiseaseConfig):
    names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            names.extend(transformer.named_steps["ohe"].get_feature_names_out(cols).tolist())
        elif name == "bin":
            names.extend(cols)
    return names

# ── train one disease ─────────────────────────────────────────────────────────

def train_disease(df, cfg: DiseaseConfig, seed=42, test_size=0.20, n_features=8):
    import pandas as pd
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    X = df.drop(columns=[cfg.target_col])
    y = df[cfg.target_col].astype(int)

    # keep only columns actually present
    num  = [c for c in cfg.numerical_cols  if c in X.columns]
    cat  = [c for c in cfg.categorical_cols if c in X.columns]
    bn   = [c for c in cfg.binary_cols     if c in X.columns]

    # rebuild cfg with present columns for this call
    cfg_local = DiseaseConfig(
        key=cfg.key, label=cfg.label, target_col=cfg.target_col,
        numerical_cols=num, categorical_cols=cat, binary_cols=bn,
        data_file=cfg.data_file, form_fields=cfg.form_fields,
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=y if y.nunique() > 1 else None)

    prep = _build_preprocessor(cfg_local)
    Xtr  = prep.fit_transform(X_tr)
    Xte  = prep.transform(X_te)
    feat_names = _feature_names(prep, cfg_local)

    # feature selection
    k = min(n_features, Xtr.shape[1])
    sel = SelectKBest(mutual_info_classif, k=k)
    sel.fit(Xtr, y_tr)
    Xtr_s = sel.transform(Xtr)
    Xte_s = sel.transform(Xte)
    mask  = sel.get_support()
    sel_feats = [f for f, m in zip(feat_names, mask) if m]
    mi_scores = sel.scores_[mask]

    # class weights — handle imbalance for BOTH models
    cw = "balanced" if y_tr.nunique() > 1 else None

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=seed,
                            class_weight=cw, solver="liblinear", C=1.0)
    lr.fit(Xtr_s, y_tr)
    lr_pred = lr.predict(Xte_s)
    lr_prob = lr.predict_proba(Xte_s)[:,1]
    lr_m    = _metrics("Logistic Regression", y_te, lr_pred, lr_prob)

    # ANN — oversample minority class so it doesn't collapse to majority
    # (MLPClassifier doesn't support sample_weight; random oversampling is
    #  the standard workaround for heavily imbalanced tabular data.)
    y_arr = np.asarray(y_tr)
    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) == 2 and counts.min() / counts.sum() < 0.15:
        # oversample minority to 50% of majority
        maj_c, min_c = classes[np.argmax(counts)], classes[np.argmin(counts)]
        n_target = counts.max()
        maj_idx = np.where(y_arr == maj_c)[0]
        min_idx = np.where(y_arr == min_c)[0]
        min_up  = np.random.choice(min_idx, size=n_target, replace=True)
        oversample_idx = np.concatenate([maj_idx, min_up])
        np.random.shuffle(oversample_idx)
        Xtr_ann = Xtr_s[oversample_idx]
        ytr_ann = y_arr[oversample_idx]
    else:
        Xtr_ann, ytr_ann = Xtr_s, y_arr

    ann = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu", solver="adam",
                        alpha=0.001, batch_size=64, learning_rate_init=0.001,
                        max_iter=300, early_stopping=True, validation_fraction=0.15,
                        n_iter_no_change=20, random_state=seed)
    ann.fit(Xtr_ann, ytr_ann)
    ann_pred = ann.predict(Xte_s)
    ann_prob = ann.predict_proba(Xte_s)[:,1]
    tl = getattr(ann, "loss_curve_", [])
    vl = [1-v for v in getattr(ann, "validation_scores_", [])]
    ann_m = _metrics("ANN (MLPClassifier)", y_te, ann_pred, ann_prob, tl, vl)

    # feature importance
    fi = sorted([{"feature": f, "importance": round(float(s),6), "rank": 0}
                 for f, s in zip(sel_feats, mi_scores)],
                key=lambda x: -x["importance"])
    for i, f in enumerate(fi): f["rank"] = i + 1

    # save
    mdir = settings.models_dir / cfg.key
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(prep, mdir / "preprocessor.joblib")
    joblib.dump(sel,  mdir / "selector.joblib")
    joblib.dump(lr,   mdir / "lr_model.joblib")
    joblib.dump(ann,  mdir / "ann_model.joblib")

    meta = {"trained_at": datetime.now(timezone.utc).isoformat(),
            "selected_features": sel_feats, "feature_importance": fi,
            "train_rows": int(len(y_tr)), "test_rows": int(len(y_te)), "seed": seed}
    metrics_data = {"logistic_regression": lr_m, "ann": ann_m}

    (mdir / "meta.json").write_text(json.dumps(meta, indent=2))
    (mdir / "metrics.json").write_text(json.dumps(metrics_data, indent=2))
    log.info("✓ %s — LR acc=%.3f  ANN acc=%.3f", cfg.key, lr_m["accuracy"], ann_m["accuracy"])
    return {"lr": lr_m, "ann": ann_m, "selected_features": sel_feats, "feature_importance": fi,
            "train_rows": int(len(y_tr)), "test_rows": int(len(y_te))}

# ── load helpers ──────────────────────────────────────────────────────────────

def disease_ready(key: str) -> bool:
    mdir = settings.models_dir / key
    return all((mdir / f).exists() for f in ["lr_model.joblib","ann_model.joblib","preprocessor.joblib","selector.joblib"])

def load_models(key: str):
    mdir = settings.models_dir / key
    return {
        "prep": joblib.load(mdir / "preprocessor.joblib"),
        "sel":  joblib.load(mdir / "selector.joblib"),
        "lr":   joblib.load(mdir / "lr_model.joblib"),
        "ann":  joblib.load(mdir / "ann_model.joblib"),
    }

def load_metrics(key: str) -> dict | None:
    p = settings.models_dir / key / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None

def load_meta(key: str) -> dict | None:
    p = settings.models_dir / key / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None

# ── Evidence thresholds: what counts as an "abnormal" value per feature ───────
# Format: feature_key -> (direction, threshold, human_label, unit)
# direction: "high"  → value > threshold is bad
#            "low"   → value < threshold is bad
#            "flag1" → binary feature; value == 1 is a risk flag
_EVIDENCE_RULES: dict[str, tuple] = {
    # Stroke
    "age":               ("high",  55,   "Age",                    "years"),
    "avg_glucose_level": ("high", 126,   "Blood Glucose",          "mg/dL"),
    "bmi":               ("high",  30,   "BMI",                    ""),
    "hypertension":      ("flag1", 0.5,  "Hypertension",           ""),
    "heart_disease":     ("flag1", 0.5,  "Heart Disease",          ""),
    # Alzheimer's
    "Age":               ("high",  70,   "Age",                    "years"),
    "MMSE":              ("low",   24,   "MMSE Score",             "/30"),
    "FunctionalAssessment": ("low", 5,   "Functional Assessment",  "/10"),
    "ADL":               ("low",    5,   "ADL Score",              "/10"),
    "MemoryComplaints":  ("flag1",  0.5, "Memory Complaints",      ""),
    "BehavioralProblems":("flag1",  0.5, "Behavioral Problems",    ""),
    "Confusion":         ("flag1",  0.5, "Confusion",              ""),
    "Disorientation":    ("flag1",  0.5, "Disorientation",         ""),
    "FamilyHistoryAlzheimers": ("flag1", 0.5, "Family History (Alzheimer's)", ""),
    "Depression":        ("flag1",  0.5, "Depression",             ""),
    "Diabetes":          ("flag1",  0.5, "Diabetes",               ""),
    "SystolicBP":        ("high", 140,   "Systolic BP",            "mmHg"),
    "DiastolicBP":       ("high",  90,   "Diastolic BP",           "mmHg"),
    "CholesterolTotal":  ("high", 240,   "Total Cholesterol",      "mg/dL"),
    "CholesterolLDL":    ("high", 160,   "LDL Cholesterol",        "mg/dL"),
    "CholesterolHDL":    ("low",   40,   "HDL Cholesterol",        "mg/dL"),
    "CholesterolTriglycerides": ("high", 200, "Triglycerides",     "mg/dL"),
    # Parkinson's voice
    "MDVP_Jitter":       ("high", 0.006,  "Jitter(%)",             ""),
    "MDVP_Shimmer":      ("high", 0.03,   "Shimmer",               ""),
    "NHR":               ("high", 0.02,   "Noise-to-Harmonics Ratio", ""),
    "HNR":               ("low",  20,     "Harmonics-to-Noise Ratio", "dB"),
    "PPE":               ("high", 0.2,    "Pitch Period Entropy",  ""),
    "Spread1":           ("high", -5.0,   "Spread1",               ""),
    "RPDE":              ("high", 0.55,   "RPDE",                  ""),
    # Brain tumor
    "TumorSize":         ("high",  0.5,  "Tumor Size",             "cm"),
    "Seizure":           ("flag1", 0.5,  "Seizures",               ""),
    "Headache":          ("flag1", 0.5,  "Persistent Headache",    ""),
    "VisionProblems":    ("flag1", 0.5,  "Vision Problems",        ""),
    "SpeechDifficulty":  ("flag1", 0.5,  "Speech Difficulty",      ""),
    "BalanceIssues":     ("flag1", 0.5,  "Balance Issues",         ""),
    "MemoryLoss":        ("flag1", 0.5,  "Memory Loss",            ""),
}

def _build_evidence(key: str, record: dict, selected_features: list[str]) -> list[dict]:
    """
    Compare the patient's values for the top selected features against
    known clinical thresholds.  Returns a list of evidence items
    (only for features that are actually abnormal / flag-positive).
    """
    evidence = []
    for feat in selected_features:
        raw_val = record.get(feat)
        if raw_val is None:
            continue
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            continue

        rule = _EVIDENCE_RULES.get(feat)
        if rule is None:
            continue

        direction, threshold, human_label, unit = rule
        abnormal = False
        explanation = ""

        if direction == "high" and val > threshold:
            abnormal = True
            explanation = f"{human_label} is {val}{(' ' + unit) if unit else ''} — above the risk threshold of {threshold}{(' ' + unit) if unit else ''}"
        elif direction == "low" and val < threshold:
            abnormal = True
            explanation = f"{human_label} is {val}{(' ' + unit) if unit else ''} — below the normal threshold of {threshold}{(' ' + unit) if unit else ''}"
        elif direction == "flag1" and val >= 0.5:
            abnormal = True
            explanation = f"{human_label} is reported as present — a significant risk factor"

        if abnormal:
            evidence.append({
                "feature":     feat,
                "label":       human_label,
                "value":       val,
                "threshold":   threshold,
                "direction":   direction,
                "explanation": explanation,
            })

    return evidence


# ── single prediction ─────────────────────────────────────────────────────────

def predict_single_disease(key: str, record: dict) -> dict:
    import pandas as pd
    mods = load_models(key)
    meta = load_meta(key) or {}
    selected_features = meta.get("selected_features", [])

    df   = pd.DataFrame([record])
    Xp   = mods["prep"].transform(df)
    Xs   = mods["sel"].transform(Xp)

    results = []
    for name, model in [("Logistic Regression", mods["lr"]), ("ANN (MLPClassifier)", mods["ann"])]:
        prob = float(model.predict_proba(Xs)[0][1])
        pred = int(model.predict(Xs)[0])
        risk_score = round(prob * 100, 1)
        if risk_score >= 65:   risk_cat = "High Risk"
        elif risk_score >= 35: risk_cat = "Medium Risk"
        else:                  risk_cat = "Low Risk"
        results.append({"model": name, "prediction": pred, "probability": round(prob,4),
                         "risk_score": risk_score, "risk_category": risk_cat})
    # ensemble: average probability
    avg_prob   = round(sum(r["probability"] for r in results) / len(results), 4)
    avg_score  = round(avg_prob * 100, 1)
    if avg_score >= 65:   cat = "High Risk"
    elif avg_score >= 35: cat = "Medium Risk"
    else:                 cat = "Low Risk"

    # Build evidence: which features are abnormal and why
    evidence = _build_evidence(key, record, selected_features)

    return {"disease": key, "models": results,
            "ensemble_probability": avg_prob, "risk_score": avg_score,
            "risk_category": cat, "evidence": evidence,
            "selected_features": selected_features}
