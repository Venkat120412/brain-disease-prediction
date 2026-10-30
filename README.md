# Brain Disease Prediction System
**Mathematical Model for Brain Disease Prediction using Artificial Neural Network (ANN)**

> Easwari Engineering College — Department of Information Technology  
> Guide: Mrs V. Kavitha, M.Tech., Ph.D.  
> Team: Sabarish R (310623205132), Venkatesh Kumar D (310623205173)

---

## What It Does
- Patient fills ONE form on the website
- System predicts risk for **all 4 brain diseases simultaneously**:
  - 🩸 Stroke | 🧩 Alzheimer's | 🤝 Parkinson's | 🔬 Brain Tumor
- Each disease gets a **Risk Score (0–100%)** and **Low / Medium / High** category
- Full report with charts, graphs, confusion matrix, ROC curve, feature importance
- Doctor Decision Support output

---

## Quick Start

### Step 1 — Backend
```bash
cd brain-disease-prediction/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Generate Data & Train Models
```bash
cd ..
python scripts/generate_data.py    # creates synthetic test data
python scripts/train_all.py        # trains all 4 models
```

### Step 3 — Start Backend
```bash
cd backend
uvicorn app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### Step 4 — Start Frontend
```bash
cd ../frontend
npm install
npm run dev
# Website: http://localhost:5173
```

---

## Website Pages
| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Project overview and disease cards |
| Predict | `/predict` | Patient data form → all disease predictions |
| Report | `/report` | Full patient report with charts and graphs |
| Dashboard | `/dashboard` | Model metrics, train all models |
| About | `/about` | Project details and team info |

---

## Disclaimer
This is an **educational research prototype**. Not a medical device.
Predictions are not diagnoses. Consult a qualified healthcare professional.
