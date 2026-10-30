"""
Training endpoint — POST /api/train/{disease} or POST /api/train/all
Accepts a CSV upload or uses the pre-generated synthetic data.
"""
import io, logging
from fastapi import APIRouter, File, HTTPException, UploadFile, status
import pandas as pd

from app.core.config import DISEASES, DISEASE_LABELS, settings
from app.services.disease_configs import DISEASE_CONFIGS
from app.services.ml_engine import train_disease

log = logging.getLogger("brain_disease.train_api")
router = APIRouter()

@router.post("/train/{disease}", summary="Train models for one disease")
async def train_one(disease: str, file: UploadFile = File(None)):
    if disease not in DISEASES:
        raise HTTPException(400, f"Unknown disease '{disease}'. Valid: {DISEASES}")
    cfg = DISEASE_CONFIGS[disease]

    if file:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(400, f"Cannot parse CSV: {e}")
    else:
        data_path = settings.raw_data_dir / cfg.data_file
        if not data_path.exists():
            raise HTTPException(404,
                f"Dataset not found at {data_path}. "
                "Run: python scripts/generate_data.py")
        df = pd.read_csv(data_path)

    if cfg.target_col not in df.columns:
        raise HTTPException(422, f"Target column '{cfg.target_col}' missing in dataset.")

    try:
        result = train_disease(df, cfg)
    except Exception as e:
        log.exception("Training failed for %s", disease)
        raise HTTPException(500, f"Training failed: {e}")

    return {
        "success": True,
        "disease": disease,
        "disease_label": DISEASE_LABELS[disease],
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "selected_features": result["selected_features"],
        "logistic_regression": result["lr"],
        "ann": result["ann"],
    }

@router.post("/train/all/run", summary="Train all 4 disease models at once")
async def train_all():
    results = {}
    for disease in DISEASES:
        cfg = DISEASE_CONFIGS[disease]
        data_path = settings.raw_data_dir / cfg.data_file
        if not data_path.exists():
            results[disease] = {"success": False, "error": "Dataset not found"}
            continue
        try:
            df = pd.read_csv(data_path)
            r  = train_disease(df, cfg)
            results[disease] = {"success": True,
                "train_rows": r["train_rows"], "test_rows": r["test_rows"],
                "lr_accuracy": r["lr"]["accuracy"], "ann_accuracy": r["ann"]["accuracy"]}
        except Exception as e:
            log.exception("Training failed for %s", disease)
            results[disease] = {"success": False, "error": str(e)}
    return results
