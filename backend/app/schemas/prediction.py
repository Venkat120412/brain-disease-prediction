"""Pydantic schemas for prediction request/response."""
from typing import Any
from pydantic import BaseModel, Field

DISCLAIMER = (
    "This application is an educational research prototype and is NOT a medical device. "
    "Predictions are NOT a diagnosis. Consult a qualified healthcare professional."
)

class ModelResult(BaseModel):
    model: str
    prediction: int
    probability: float
    risk_score: float
    risk_category: str

class EvidenceItem(BaseModel):
    feature: str
    label: str
    value: float
    threshold: float
    direction: str          # "high" | "low" | "flag1"
    explanation: str        # human-readable sentence

class DiseaseResult(BaseModel):
    disease: str
    disease_label: str
    models: list[ModelResult]
    ensemble_probability: float
    risk_score: float
    risk_category: str
    evidence: list[EvidenceItem] = Field(default_factory=list)  # WHY the risk is high

class PatientReportResponse(BaseModel):
    disclaimer: str = DISCLAIMER
    patient_data: dict[str, Any]
    results: list[DiseaseResult]
    primary_risk_disease: str
    summary_message: str
    extraction: dict[str, Any] | None = None

class ModelStatusItem(BaseModel):
    disease: str
    disease_label: str
    trained: bool
    trained_at: str = ""
    selected_features: list[str] = Field(default_factory=list)
    train_rows: int = 0
    test_rows: int  = 0
