"""Train all 4 brain disease models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pandas as pd
from app.core.config import settings, DISEASES
from app.services.disease_configs import DISEASE_CONFIGS
from app.services.ml_engine import train_disease

for key in DISEASES:
    cfg  = DISEASE_CONFIGS[key]
    path = settings.raw_data_dir / cfg.data_file
    if not path.exists():
        print(f"✗ {key}: data file missing at {path}"); continue
    df   = pd.read_csv(path)
    r    = train_disease(df, cfg)
    print(f"✓ {cfg.label:<25}  LR={r['lr']['accuracy']:.3f}  ANN={r['ann']['accuracy']:.3f}  "
          f"features={r['selected_features'][:3]}")

print("\n✓ All models trained. Run the backend: uvicorn app.main:app --reload")
