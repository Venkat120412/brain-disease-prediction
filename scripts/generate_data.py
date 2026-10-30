"""
Generate larger, more realistic synthetic datasets for all 4 brain diseases.

The features are generated CONDITIONAL on the disease status so the models
actually learn a real signal (instead of near-random noise that makes every
prediction collapse to ~45%).  This mirrors the discriminative structure of
real clinical datasets, e.g. the UCI Parkinson's voice dataset.

FOR SOFTWARE TESTING ONLY — not clinically valid.

Usage:  python scripts/generate_data.py
Output: data/raw/{stroke, alzheimers, parkinsons, brain_tumor}_data.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

N = 5000   # rows per disease (10× the original Parkinson's set)


def _norm(mean, sd, lo, hi, n=N):
    """Clipped normal sample helper."""
    return np.clip(np.random.normal(mean, sd, n), lo, hi)


# ── 1. STROKE ────────────────────────────────────────────────────────────────
def stroke(n=N):
    y = np.random.binomial(1, 0.30, n)                                  # ~30% positive
    age = np.where(y, _norm(70, 11, 0.1, 100), _norm(48, 17, 0.1, 100)).round(1)
    hyp = np.where(y, np.random.binomial(1, 0.55, n), np.random.binomial(1, 0.12, n))
    hd  = np.where(y, np.random.binomial(1, 0.30, n), np.random.binomial(1, 0.05, n))
    gluc = np.where(y, _norm(138, 28, 55, 270), _norm(98, 26, 55, 270)).round(2)
    bmi = np.where(y, _norm(31, 6, 10, 60), _norm(27, 5, 10, 60)).round(1)
    bmi = bmi.astype(object)
    bmi[np.random.random(n) < 0.04] = np.nan                            # a few missing
    gender = np.random.choice(["Male", "Female", "Other"], n, p=[0.49, 0.50, 0.01])
    marry  = np.random.choice(["Yes", "No"], n, p=[0.66, 0.34])
    work   = np.random.choice(["Private", "Govt_job", "Self-employed", "children",
                                "Never_worked"], n, p=[0.57, 0.13, 0.16, 0.13, 0.01])
    res    = np.random.choice(["Urban", "Rural"], n, p=[0.51, 0.49])
    smoke  = np.where(y,
        np.random.choice(["never smoked", "formerly smoked", "smokes", "Unknown"], n,
                         p=[0.22, 0.30, 0.28, 0.20]),
        np.random.choice(["never smoked", "formerly smoked", "smokes", "Unknown"], n,
                         p=[0.45, 0.15, 0.10, 0.30]))
    df = pd.DataFrame(dict(id=range(1, n + 1), gender=gender, age=age, hypertension=hyp,
        heart_disease=hd, ever_married=marry, work_type=work, Residence_type=res,
        avg_glucose_level=gluc, bmi=bmi, smoking_status=smoke, stroke=y))
    df.to_csv(OUT / "stroke_data.csv", index=False)
    print(f"✓ stroke_data.csv  {n} rows  positives={int(y.sum())}")


# ── 2. ALZHEIMER'S ───────────────────────────────────────────────────────────
def alzheimers(n=N):
    y = np.random.binomial(1, 0.40, n)                                  # ~40% positive
    age = np.where(y, _norm(78, 6, 50, 95), _norm(66, 8, 50, 95)).round(0).astype(int)
    gender = np.random.choice([0, 1], n)                                # 0=M 1=F
    educ = np.random.randint(4, 20, n)
    mmse = np.where(y, _norm(18, 5, 0, 30), _norm(27, 2.5, 0, 30)).round(1)
    cdrsb = np.where(y, np.clip(np.random.exponential(3.5, n), 0, 18),
                     np.clip(np.random.exponential(0.5, n), 0, 18)).round(2)
    adas11 = np.where(y, _norm(24, 8, 0, 70), _norm(10, 4, 0, 70)).round(1)
    func_act = np.where(y, _norm(11, 5, 0, 30), _norm(1.5, 2, 0, 30)).round(1)
    apoe4 = np.where(y, np.random.choice([0, 1, 2], n, p=[0.35, 0.40, 0.25]),
                      np.random.choice([0, 1, 2], n, p=[0.70, 0.25, 0.05]))
    hippo = np.where(y, _norm(5600, 900, 2000, 10000), _norm(7200, 800, 2000, 10000)).round(0)
    wbv = np.where(y, _norm(1.05, 0.06, 0.9, 1.4), _norm(1.15, 0.05, 0.9, 1.4)).round(4)
    df = pd.DataFrame(dict(Age=age, Gender=gender, Education=educ, MMSE=mmse,
        CDR_SB=cdrsb, ADAS11=adas11, FunctionalActivities=func_act, APOE4=apoe4,
        Hippocampus=hippo, WholeBrainVolume=wbv, Diagnosis=y))
    df.to_csv(OUT / "alzheimers_data.csv", index=False)
    print(f"✓ alzheimers_data.csv  {n} rows  positives={int(y.sum())}")


# ── 3. PARKINSON'S ───────────────────────────────────────────────────────────
# Voice/acoustic features mirror the UCI Parkinson's dataset:
# Parkinson's patients have IRREGULAR voice → high jitter/shimmer/NHR,
# low HNR, more nonlinear complexity (high RPDE/DFA/Spread/D2).
def parkinsons(n=N):
    y = np.random.binomial(1, 0.50, n)                                  # balanced
    fo  = np.where(y, _norm(145, 35, 80, 260),  _norm(175, 24, 80, 260)).round(3)
    fhi = np.where(y, _norm(190, 70, 100, 592), _norm(205, 38, 100, 592)).round(3)
    flo = np.where(y, _norm(104, 30, 60, 239),  _norm(150, 18, 60, 239)).round(3)
    jitter = np.where(y,
        np.clip(np.random.exponential(0.012, n) + 0.004, 0.001, 0.1),
        np.clip(np.random.exponential(0.003, n) + 0.001, 0.001, 0.1)).round(5)
    shimmer = np.where(y,
        np.clip(np.random.exponential(0.05, n) + 0.02, 0.005, 0.3),
        np.clip(np.random.exponential(0.012, n) + 0.005, 0.005, 0.3)).round(5)
    nhr = np.where(y,
        np.clip(np.random.exponential(0.03, n) + 0.02, 0.001, 0.7),
        np.clip(np.random.exponential(0.008, n) + 0.002, 0.001, 0.7)).round(5)
    hnr = np.where(y, _norm(18, 3, 8, 34), _norm(26, 2.5, 8, 34)).round(3)
    rpde = np.where(y, _norm(0.62, 0.08, 0.25, 0.85), _norm(0.42, 0.06, 0.25, 0.85)).round(6)
    dfa = np.where(y, _norm(0.76, 0.07, 0.5, 0.97),  _norm(0.67, 0.06, 0.5, 0.97)).round(6)
    spread1 = np.where(y, _norm(-4.7, 0.9, -8, -1), _norm(-6.2, 0.7, -8, -1)).round(6)
    spread2 = np.where(y, _norm(0.29, 0.08, 0, 0.45), _norm(0.18, 0.06, 0, 0.45)).round(6)
    d2 = np.where(y, _norm(2.65, 0.35, 1.5, 3.7), _norm(2.05, 0.28, 1.5, 3.7)).round(6)
    df = pd.DataFrame(dict(MDVP_Fo=fo, MDVP_Fhi=fhi, MDVP_Flo=flo,
        MDVP_Jitter=jitter, MDVP_Shimmer=shimmer, NHR=nhr, HNR=hnr,
        RPDE=rpde, DFA=dfa, Spread1=spread1, Spread2=spread2, D2=d2, status=y))
    df.to_csv(OUT / "parkinsons_data.csv", index=False)
    print(f"✓ parkinsons_data.csv  {n} rows  positives={int(y.sum())}")


# ── 4. BRAIN TUMOR ──────────────────────────────────────────────────────────
# Tumor size & location are the dominant predictors (from imaging).
# A patient with NO imaging (size≈0) and only vague symptoms → low risk.
def brain_tumor(n=N):
    y = np.random.binomial(1, 0.30, n)                                  # ~30% positive
    age = np.where(y, _norm(52, 16, 5, 85), _norm(42, 17, 5, 85)).round(0).astype(int)
    gender = np.random.choice([0, 1], n)
    headache = np.where(y, np.random.binomial(1, 0.62, n), np.random.binomial(1, 0.30, n))
    seizure  = np.where(y, np.random.binomial(1, 0.50, n), np.random.binomial(1, 0.08, n))
    nausea   = np.where(y, np.random.binomial(1, 0.50, n), np.random.binomial(1, 0.18, n))
    vision   = np.where(y, np.random.binomial(1, 0.52, n), np.random.binomial(1, 0.15, n))
    speech   = np.where(y, np.random.binomial(1, 0.42, n), np.random.binomial(1, 0.10, n))
    memory   = np.where(y, np.random.binomial(1, 0.45, n), np.random.binomial(1, 0.20, n))
    balance  = np.where(y, np.random.binomial(1, 0.48, n), np.random.binomial(1, 0.12, n))
    fatigue  = np.where(y, np.random.binomial(1, 0.58, n), np.random.binomial(1, 0.32, n))
    tumor_sz = np.where(y,
        np.clip(np.random.exponential(2.6, n), 0, 8),
        np.clip(np.random.exponential(0.35, n), 0, 8)).round(2)
    location = np.where(y,
        np.random.choice([0, 1, 2, 3], n, p=[0.10, 0.40, 0.30, 0.20]),   # known lobe
        np.random.choice([0, 1, 2, 3], n, p=[0.55, 0.18, 0.15, 0.12]))   # often unknown
    df = pd.DataFrame(dict(Age=age, Gender=gender, Headache=headache,
        Seizure=seizure, Nausea=nausea, VisionProblems=vision, SpeechDifficulty=speech,
        MemoryLoss=memory, BalanceIssues=balance, Fatigue=fatigue,
        TumorSize=tumor_sz, Location=location, tumor_present=y))
    df.to_csv(OUT / "brain_tumor_data.csv", index=False)
    print(f"✓ brain_tumor_data.csv  {n} rows  positives={int(y.sum())}")


if __name__ == "__main__":
    stroke()
    alzheimers()
    parkinsons()
    brain_tumor()
    print(f"\n⚠  SYNTHETIC DATA — {N} rows per disease — for software testing only. Not clinically valid.")
