"""
Prepare real-world datasets from Kaggle/UCI for training.

Reads raw downloads from data/real_raw/, cleans them, renames columns to
match the updated disease configs, and writes the prepared CSVs to data/raw/
(overwriting the synthetic files).  The original synthetic models are backed
up in data/artifacts/models_synthetic_backup/.

Datasets:
  Stroke       — fedesoriano/stroke-prediction-dataset  (Kaggle, 5 110 rows)
  Alzheimer's  — rabieelkharoua/alzheimers-disease-dataset (Kaggle, 2 149 rows)
  Parkinson's  — UCI Parkinson's Data Set                  (UCI,    195 rows)
  Brain Tumor  — NO real tabular dataset on Kaggle (MRI images only).
                 Keeps the existing synthetic CSV so the model still works.

Usage:  python scripts/prepare_real_data.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "real_raw"
OUT = PROJECT_ROOT / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)


# ── 1. STROKE ────────────────────────────────────────────────────────────────
def prepare_stroke():
    src = RAW / "healthcare-dataset-stroke-data.csv"
    df = pd.read_csv(src)

    # drop the id column (not a predictive feature)
    df = df.drop(columns=["id"])

    # bmi has 201 NaN values — keep them as NaN; the SimpleImputer in the
    # training pipeline will fill with the median.  (The Kaggle file already
    # stores them as NaN, not the string "N/A".)
    # Remove any rows with gender == "Other" (only 1 row) to keep binary-ish,
    # but actually keep it — OneHotEncoder handles it.
    # Just make sure dtypes are correct.
    df["age"] = df["age"].astype(float)
    df["avg_glucose_level"] = df["avg_glucose_level"].astype(float)
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["hypertension"] = df["hypertension"].astype(int)
    df["heart_disease"] = df["heart_disease"].astype(int)
    df["stroke"] = df["stroke"].astype(int)

    # remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    df.to_csv(OUT / "stroke_data.csv", index=False)
    print(f"✓ stroke_data.csv      {df.shape[0]} rows  "
          f"(removed {before - after} duplicates)  "
          f"stroke=1: {(df['stroke']==1).sum()}  "
          f"missing bmi: {df['bmi'].isna().sum()}")


# ── 2. ALZHEIMER'S ───────────────────────────────────────────────────────────
def prepare_alzheimers():
    src = RAW / "alzheimers_disease_data.csv"
    df = pd.read_csv(src)

    # drop non-predictive columns
    df = df.drop(columns=["PatientID", "DoctorInCharge"])

    # ensure correct dtypes
    for c in ["Gender", "Ethnicity", "EducationLevel", "Smoking",
              "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes",
              "Depression", "HeadInjury", "Hypertension", "MemoryComplaints",
              "BehavioralProblems", "Confusion", "Disorientation",
              "PersonalityChanges", "DifficultyCompletingTasks",
              "Forgetfulness", "Diagnosis"]:
        df[c] = df[c].astype(int)

    for c in ["Age", "SystolicBP", "DiastolicBP"]:
        df[c] = df[c].astype(int)

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    df.to_csv(OUT / "alzheimers_data.csv", index=False)
    print(f"✓ alzheimers_data.csv   {df.shape[0]} rows  "
          f"(removed {before - after} duplicates)  "
          f"Diagnosis=1: {(df['Diagnosis']==1).sum()}  "
          f"missing: {df.isnull().sum().sum()}")


# ── 3. PARKINSON'S ───────────────────────────────────────────────────────────
# UCI column names contain colons/parentheses — rename to underscore format
# that matches the updated DiseaseConfig.
PARKINSONS_RENAME = {
    "name": "name",                      # will be dropped
    "MDVP:Fo(Hz)": "MDVP_Fo",
    "MDVP:Fhi(Hz)": "MDVP_Fhi",
    "MDVP:Flo(Hz)": "MDVP_Flo",
    "MDVP:Jitter(%)": "MDVP_Jitter",
    "MDVP:Jitter(Abs)": "MDVP_Jitter_Abs",
    "MDVP:RAP": "MDVP_RAP",
    "MDVP:PPQ": "MDVP_PPQ",
    "Jitter:DDP": "Jitter_DDP",
    "MDVP:Shimmer": "MDVP_Shimmer",
    "MDVP:Shimmer(dB)": "MDVP_Shimmer_dB",
    "Shimmer:APQ3": "Shimmer_APQ3",
    "Shimmer:APQ5": "Shimmer_APQ5",
    "MDVP:APQ": "MDVP_APQ",
    "Shimmer:DDA": "Shimmer_DDA",
    "NHR": "NHR",
    "HNR": "HNR",
    "status": "status",                   # target — keep
    "RPDE": "RPDE",
    "DFA": "DFA",
    "spread1": "Spread1",
    "spread2": "Spread2",
    "D2": "D2",
    "PPE": "PPE",
}

def prepare_parkinsons():
    src = RAW / "parkinsons_uci_raw.csv"
    df = pd.read_csv(src)
    df = df.rename(columns=PARKINSONS_RENAME)
    df = df.drop(columns=["name"])       # patient/recording ID — not a feature
    df["status"] = df["status"].astype(int)

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    df.to_csv(OUT / "parkinsons_data.csv", index=False)
    print(f"✓ parkinsons_data.csv   {df.shape[0]} rows  "
          f"(removed {before - after} duplicates)  "
          f"status=1: {(df['status']==1).sum()}  "
          f"features: {len(df.columns) - 1}")


# ── 4. BRAIN TUMOR ────────────────────────────────────────────────────────────
# No real tabular brain-tumor dataset is publicly available on Kaggle.
# The existing synthetic CSV (generated by generate_data.py) is kept as-is.
# This is clearly documented; a real brain-tumor model would require an MRI
# image-classification pipeline (CNN), which is outside the scope of this
# tabular-data pipeline.
def brain_tumor_note():
    bt = OUT / "brain_tumor_data.csv"
    if bt.exists():
        df = pd.read_csv(bt)
        print(f"⊙ brain_tumor_data.csv {df.shape[0]} rows  "
              f"(SYNTHETIC — no real tabular dataset available; "
              f"kept as-is)")
    else:
        print("⊙ brain_tumor_data.csv — not found (run generate_data.py first)")


if __name__ == "__main__":
    print("Preparing real-world datasets…\n")
    prepare_stroke()
    prepare_alzheimers()
    prepare_parkinsons()
    brain_tumor_note()
    print("\nDone. Real data written to data/raw/.")
    print("Brain tumor retains synthetic data (no real tabular source exists).")
