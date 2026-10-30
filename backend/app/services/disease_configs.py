"""
Configuration for each disease: feature columns, target column,
categorical columns, numerical columns, and display metadata.
Adding a new disease = add one entry here.
"""
from dataclasses import dataclass, field

@dataclass
class DiseaseConfig:
    key: str                           # internal key e.g. "stroke"
    label: str                         # display name
    target_col: str                    # CSV column that is 0/1
    numerical_cols: list[str]
    categorical_cols: list[str]
    binary_cols: list[str]
    data_file: str                     # filename in data/raw/
    positive_label: str = "Detected"
    negative_label: str = "Not Detected"
    description: str = ""
    # patient form field definitions for the UI
    form_fields: list[dict] = field(default_factory=list)


DISEASE_CONFIGS: dict[str, DiseaseConfig] = {

    # ── STROKE ────────────────────────────────────────────────────────────────
    "stroke": DiseaseConfig(
        key="stroke", label="Stroke",
        target_col="stroke",
        numerical_cols=["age", "avg_glucose_level", "bmi"],
        categorical_cols=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"],
        binary_cols=["hypertension", "heart_disease"],
        data_file="stroke_data.csv",
        description="A stroke occurs when blood supply to part of the brain is cut off. "
                    "Trained on real patient data from Kaggle (5 110 records).",
        form_fields=[
            {"id":"age",               "label":"Age (years)",          "type":"number", "min":0,  "max":120},
            {"id":"avg_glucose_level", "label":"Avg Glucose (mg/dL)",  "type":"number", "min":55, "max":300},
            {"id":"bmi",               "label":"BMI",                  "type":"number", "min":10, "max":60, "optional":True},
            {"id":"hypertension",      "label":"Hypertension",         "type":"select", "options":["0","1"]},
            {"id":"heart_disease",     "label":"Heart Disease",        "type":"select", "options":["0","1"]},
            {"id":"gender",            "label":"Gender",               "type":"select", "options":["Male","Female","Other"]},
            {"id":"ever_married",      "label":"Ever Married",         "type":"select", "options":["Yes","No"]},
            {"id":"work_type",         "label":"Work Type",            "type":"select", "options":["Private","Govt_job","Self-employed","children","Never_worked"]},
            {"id":"Residence_type",    "label":"Residence",            "type":"select", "options":["Urban","Rural"]},
            {"id":"smoking_status",    "label":"Smoking Status",       "type":"select", "options":["never smoked","formerly smoked","smokes","Unknown"]},
        ],
    ),

    # ── ALZHEIMER'S (real data: Kaggle rabieelkharoua/alzheimers-disease-dataset) ──
    "alzheimers": DiseaseConfig(
        key="alzheimers", label="Alzheimer's Disease",
        target_col="Diagnosis",
        numerical_cols=["Age","BMI","AlcoholConsumption","PhysicalActivity",
                        "DietQuality","SleepQuality","SystolicBP","DiastolicBP",
                        "CholesterolTotal","CholesterolLDL","CholesterolHDL",
                        "CholesterolTriglycerides","MMSE","FunctionalAssessment","ADL"],
        categorical_cols=[],
        binary_cols=["Gender","Ethnicity","EducationLevel","Smoking",
                      "FamilyHistoryAlzheimers","CardiovascularDisease","Diabetes",
                      "Depression","HeadInjury","Hypertension","MemoryComplaints",
                      "BehavioralProblems","Confusion","Disorientation",
                      "PersonalityChanges","DifficultyCompletingTasks","Forgetfulness"],
        data_file="alzheimers_data.csv",
        description="A progressive brain disease that slowly destroys memory and thinking skills. "
                    "Trained on real patient data from Kaggle (2 149 records, CC BY 4.0).",
        form_fields=[
            {"id":"Age",                    "label":"Age (years)",            "type":"number","min":60,"max":90},
            {"id":"Gender",                 "label":"Gender",                 "type":"select","options":["0","1"],"hint":"0=Male 1=Female"},
            {"id":"Ethnicity",              "label":"Ethnicity",              "type":"select","options":["0","1","2","3"],"hint":"0=Caucasian 1=AfricanAmerican 2=Asian 3=Other"},
            {"id":"EducationLevel",         "label":"Education Level",        "type":"select","options":["0","1","2","3"],"hint":"0=None 1=HighSchool 2=Bachelors 3=Higher"},
            {"id":"BMI",                    "label":"BMI",                    "type":"number","min":15,"max":40},
            {"id":"Smoking",                "label":"Smoking",                "type":"select","options":["0","1"],"hint":"0=No 1=Yes"},
            {"id":"AlcoholConsumption",     "label":"Alcohol (units/week)",   "type":"number","min":0,"max":20},
            {"id":"PhysicalActivity",       "label":"Physical Activity (h/wk)","type":"number","min":0,"max":10},
            {"id":"DietQuality",            "label":"Diet Quality (0-10)",    "type":"number","min":0,"max":10},
            {"id":"SleepQuality",           "label":"Sleep Quality (4-10)",   "type":"number","min":4,"max":10},
            {"id":"FamilyHistoryAlzheimers","label":"Family History AD",      "type":"select","options":["0","1"]},
            {"id":"CardiovascularDisease",   "label":"Cardiovascular Disease", "type":"select","options":["0","1"]},
            {"id":"Diabetes",               "label":"Diabetes",              "type":"select","options":["0","1"]},
            {"id":"Depression",             "label":"Depression",             "type":"select","options":["0","1"]},
            {"id":"HeadInjury",             "label":"Head Injury",           "type":"select","options":["0","1"]},
            {"id":"Hypertension",           "label":"Hypertension",          "type":"select","options":["0","1"]},
            {"id":"SystolicBP",             "label":"Systolic BP (mmHg)",     "type":"number","min":90,"max":180},
            {"id":"DiastolicBP",            "label":"Diastolic BP (mmHg)",    "type":"number","min":60,"max":120},
            {"id":"CholesterolTotal",       "label":"Total Cholesterol",     "type":"number","min":150,"max":300},
            {"id":"CholesterolLDL",         "label":"LDL Cholesterol",        "type":"number","min":50,"max":200},
            {"id":"CholesterolHDL",         "label":"HDL Cholesterol",       "type":"number","min":20,"max":100},
            {"id":"CholesterolTriglycerides","label":"Triglycerides",         "type":"number","min":50,"max":400},
            {"id":"MMSE",                   "label":"MMSE Score (0-30)",     "type":"number","min":0,"max":30,"hint":"Mini-Mental State Exam"},
            {"id":"FunctionalAssessment",   "label":"Functional Assessment (0-10)","type":"number","min":0,"max":10},
            {"id":"MemoryComplaints",       "label":"Memory Complaints",     "type":"select","options":["0","1"]},
            {"id":"BehavioralProblems",     "label":"Behavioral Problems",    "type":"select","options":["0","1"]},
            {"id":"ADL",                   "label":"ADL Score (0-10)",       "type":"number","min":0,"max":10,"hint":"Activities of Daily Living"},
            {"id":"Confusion",             "label":"Confusion",              "type":"select","options":["0","1"]},
            {"id":"Disorientation",         "label":"Disorientation",         "type":"select","options":["0","1"]},
            {"id":"PersonalityChanges",    "label":"Personality Changes",    "type":"select","options":["0","1"]},
            {"id":"DifficultyCompletingTasks","label":"Difficulty Completing Tasks","type":"select","options":["0","1"]},
            {"id":"Forgetfulness",         "label":"Forgetfulness",          "type":"select","options":["0","1"]},
        ],
    ),

    # ── PARKINSON'S (real data: UCI Parkinson's Data Set, 195 voice recordings) ──
    "parkinsons": DiseaseConfig(
        key="parkinsons", label="Parkinson's Disease",
        target_col="status",
        numerical_cols=["MDVP_Fo","MDVP_Fhi","MDVP_Flo","MDVP_Jitter","MDVP_Jitter_Abs",
                        "MDVP_RAP","MDVP_PPQ","Jitter_DDP","MDVP_Shimmer","MDVP_Shimmer_dB",
                        "Shimmer_APQ3","Shimmer_APQ5","MDVP_APQ","Shimmer_DDA",
                        "NHR","HNR","RPDE","DFA","Spread1","Spread2","D2","PPE"],
        categorical_cols=[],
        binary_cols=[],
        data_file="parkinsons_data.csv",
        description="A progressive nervous system disorder affecting movement, detected through voice measurements. "
                    "Trained on real UCI voice data (195 recordings, 31 subjects).",
        form_fields=[
            {"id":"MDVP_Fo",          "label":"MDVP:Fo(Hz) — Avg vocal freq",       "type":"number","min":80, "max":260},
            {"id":"MDVP_Fhi",         "label":"MDVP:Fhi(Hz) — Max vocal freq",      "type":"number","min":100,"max":600},
            {"id":"MDVP_Flo",         "label":"MDVP:Flo(Hz) — Min vocal freq",      "type":"number","min":60, "max":240},
            {"id":"MDVP_Jitter",     "label":"MDVP:Jitter(%) — Freq variation",    "type":"number","min":0,  "max":0.1},
            {"id":"MDVP_Jitter_Abs",  "label":"MDVP:Jitter(Abs) — Abs freq var",    "type":"number","min":0,  "max":0.0001},
            {"id":"MDVP_RAP",         "label":"MDVP:RAP — Rel amp perturbation",   "type":"number","min":0,  "max":0.01},
            {"id":"MDVP_PPQ",         "label":"MDVP:PPQ — 5-pt period perturb",    "type":"number","min":0,  "max":0.01},
            {"id":"Jitter_DDP",      "label":"Jitter:DDP — Avg abs diff",         "type":"number","min":0,  "max":0.03},
            {"id":"MDVP_Shimmer",    "label":"MDVP:Shimmer — Amp variation",       "type":"number","min":0,  "max":0.3},
            {"id":"MDVP_Shimmer_dB", "label":"MDVP:Shimmer(dB) — Amp var in dB",  "type":"number","min":0,  "max":2},
            {"id":"Shimmer_APQ3",    "label":"Shimmer:APQ3 — 3-pt amp perturb",   "type":"number","min":0,  "max":0.1},
            {"id":"Shimmer_APQ5",    "label":"Shimmer:APQ5 — 5-pt amp perturb",   "type":"number","min":0,  "max":0.1},
            {"id":"MDVP_APQ",        "label":"MDVP:APQ — Amp perturbation q",      "type":"number","min":0,  "max":0.1},
            {"id":"Shimmer_DDA",     "label":"Shimmer:DDA — Avg abs diff",        "type":"number","min":0,  "max":0.3},
            {"id":"NHR",              "label":"NHR — Noise-to-Harmonics Ratio",     "type":"number","min":0,  "max":0.7},
            {"id":"HNR",              "label":"HNR — Harmonics-to-Noise Ratio",   "type":"number","min":8,  "max":34},
            {"id":"RPDE",             "label":"RPDE — Recurrence period density",   "type":"number","min":0.25,"max":0.85},
            {"id":"DFA",              "label":"DFA — Signal fractal scaling",        "type":"number","min":0.5,"max":0.97},
            {"id":"Spread1",          "label":"Spread1 — Nonlinear measure",        "type":"number","min":-8, "max":-1},
            {"id":"Spread2",          "label":"Spread2 — Nonlinear measure",        "type":"number","min":0,  "max":0.45},
            {"id":"D2",               "label":"D2 — Correlation dimension",         "type":"number","min":1.5,"max":3.7},
            {"id":"PPE",              "label":"PPE — Pitch period entropy",         "type":"number","min":0.04,"max":0.5},
        ],
    ),

    # ── BRAIN TUMOR ───────────────────────────────────────────────────────────
    "brain_tumor": DiseaseConfig(
        key="brain_tumor", label="Brain Tumor",
        target_col="tumor_present",
        numerical_cols=["Age","TumorSize"],
        categorical_cols=[],
        binary_cols=["Gender","Headache","Seizure","Nausea","VisionProblems",
                     "SpeechDifficulty","MemoryLoss","BalanceIssues","Fatigue","Location"],
        data_file="brain_tumor_data.csv",
        description="Abnormal growth of cells in the brain, detected through symptoms and imaging. "
                    "NOTE: trained on synthetic data — no real tabular brain-tumor dataset exists on Kaggle "
                    "(all are MRI images requiring a separate CNN pipeline).",
        form_fields=[
            {"id":"Age",             "label":"Age (years)",                "type":"number","min":5,"max":85},
            {"id":"Gender",          "label":"Gender",                     "type":"select","options":["0","1"],"hint":"0=Male 1=Female"},
            {"id":"Headache",        "label":"Persistent Headache",        "type":"select","options":["0","1"]},
            {"id":"Seizure",         "label":"Seizures",                   "type":"select","options":["0","1"]},
            {"id":"Nausea",          "label":"Nausea / Vomiting",          "type":"select","options":["0","1"]},
            {"id":"VisionProblems",  "label":"Vision Problems",            "type":"select","options":["0","1"]},
            {"id":"SpeechDifficulty","label":"Speech Difficulty",          "type":"select","options":["0","1"]},
            {"id":"MemoryLoss",      "label":"Memory Loss",                "type":"select","options":["0","1"]},
            {"id":"BalanceIssues",   "label":"Balance Issues",             "type":"select","options":["0","1"]},
            {"id":"Fatigue",         "label":"Chronic Fatigue",            "type":"select","options":["0","1"]},
            {"id":"TumorSize",       "label":"Tumor Size (cm, if known)",  "type":"number","min":0,"max":8,"optional":True},
            {"id":"Location",        "label":"Location (0-3 lobe)",        "type":"select","options":["0","1","2","3"]},
        ],
    ),
}
