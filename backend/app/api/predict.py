"""
Prediction endpoints.
POST /api/predict       — patient submits JSON data, get all-disease report
POST /api/predict-pdf   — patient uploads a PDF report; data is read from it
POST /api/debug-pdf     — returns raw extracted text from a PDF (diagnostic)
GET  /api/status        — model readiness
GET  /api/metrics/{disease}
GET  /api/features/{disease}
GET  /api/form-fields   — form schema for frontend
"""
import io
import logging
import re

import pdfplumber
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from app.core.config import DISEASES, DISEASE_LABELS
from app.services.disease_configs import DISEASE_CONFIGS
from app.services.ml_engine import (
    disease_ready, load_metrics, load_meta, predict_single_disease
)
from app.services.pdf_extractor import extract_pdf
from app.schemas.prediction import (
    DiseaseResult, EvidenceItem, ModelResult, ModelStatusItem, PatientReportResponse
)

log = logging.getLogger("brain_disease.api")
router = APIRouter()

DISCLAIMER = (
    "This application is an educational research prototype and is NOT a medical device. "
    "Predictions are NOT a diagnosis. Consult a qualified healthcare professional."
)


def _build_report(patient_data: dict, extraction: dict | None = None) -> PatientReportResponse:
    """Run all-disease prediction for a patient data dict.

    Shared by the JSON and PDF endpoints so both paths stay identical.
    """
    results: list[DiseaseResult] = []

    for key in DISEASES:
        if not disease_ready(key):
            log.warning("Model not ready: %s", key)
            continue
        cfg = DISEASE_CONFIGS[key]
        # Build feature dict for this disease (only its columns)
        all_cols = cfg.numerical_cols + cfg.categorical_cols + cfg.binary_cols
        record = {}
        for col in all_cols:
            val = patient_data.get(col)
            if val is not None:
                # coerce numeric strings
                try:
                    if col in cfg.numerical_cols or col in cfg.binary_cols:
                        val = float(val)
                except (ValueError, TypeError):
                    pass
            record[col] = val

        try:
            raw = predict_single_disease(key, record)
        except Exception as exc:
            log.error("Prediction failed for %s: %s", key, exc)
            continue

        model_results = [ModelResult(**m) for m in raw["models"]]
        evidence_items = [EvidenceItem(**e) for e in raw.get("evidence", [])]

        # Build a natural-language evidence summary for high-risk diseases
        ev_summary = ""
        if raw["risk_category"] == "High Risk" and evidence_items:
            bullets = "; ".join(e.explanation for e in evidence_items[:5])
            ev_summary = f"Key risk indicators: {bullets}."
        elif raw["risk_category"] == "Medium Risk" and evidence_items:
            bullets = "; ".join(e.explanation for e in evidence_items[:3])
            ev_summary = f"Elevated risk factors: {bullets}."

        results.append(DiseaseResult(
            disease=key,
            disease_label=DISEASE_LABELS[key],
            models=model_results,
            ensemble_probability=raw["ensemble_probability"],
            risk_score=raw["risk_score"],
            risk_category=raw["risk_category"],
            evidence=evidence_items,
        ))

    if not results:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No trained models found. Run: python scripts/train_all.py"
        )

    # Find primary risk disease
    primary = max(results, key=lambda r: r.risk_score)
    high_risks = [r for r in results if r.risk_category == "High Risk"]

    if high_risks:
        disease_names = ", ".join(r.disease_label for r in high_risks)
        ev_lines = []
        for r in high_risks:
            if r.evidence:
                top = r.evidence[:3]
                ev_lines.append(
                    f"{r.disease_label}: " +
                    "; ".join(e.explanation for e in top)
                )
        ev_text = "  |  ".join(ev_lines)
        msg = (
            f"⚠ High risk detected for: {disease_names}. "
            f"Please consult a healthcare professional immediately. "
            + (f" Evidence — {ev_text}." if ev_text else "")
        )
    elif any(r.risk_category == "Medium Risk" for r in results):
        med = [r for r in results if r.risk_category == "Medium Risk"]
        ev_lines = []
        for r in med:
            if r.evidence:
                ev_lines.append(f"{r.disease_label}: {r.evidence[0].explanation}")
        ev_text = "  ".join(ev_lines)
        msg = (
            "Moderate risk factors detected. Regular monitoring and medical consultation recommended."
            + (f" {ev_text}." if ev_text else "")
        )
    else:
        msg = "Current risk levels appear low. Maintain a healthy lifestyle and schedule regular check-ups."

    return PatientReportResponse(
        disclaimer=DISCLAIMER,
        patient_data=patient_data,
        results=results,
        primary_risk_disease=primary.disease_label,
        summary_message=msg,
        extraction=extraction,
    )


@router.post("/predict", response_model=PatientReportResponse, summary="Predict all brain diseases")
async def predict(patient_data: dict):
    """
    Submit a patient data dict (all fields combined).
    The system predicts risk for all 4 brain diseases and returns a full report.
    """
    return _build_report(patient_data)


@router.post("/predict-pdf", response_model=PatientReportResponse,
             summary="Predict from an uploaded medical report PDF")
async def predict_pdf(file: UploadFile = File(...)):
    """
    Upload a patient medical report in PDF format.  The system reads the
    patient data straight from the document, predicts risk for all 4 brain
    diseases, and returns a full report plus a summary of which fields were
    extracted.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty PDF file.")

    try:
        ex = extract_pdf(raw)
    except Exception as exc:
        log.error("PDF extraction failed: %s", exc)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Could not read the PDF. Make sure it is a text-based (not scanned) PDF.",
        )

    if ex.text_length < 20:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "No readable text found in the PDF. It may be a scanned image — "
            "please enter the patient data manually instead.",
        )

    extraction = {
        "fields_read": len(ex.found),
        "fields_missing": len(ex.missing),
        "found": ex.found,
        "missing": ex.missing,
        "page_count": ex.page_count,
    }
    log.info("PDF parsed: %d pages, %d fields extracted", ex.page_count, len(ex.found))
    return _build_report(ex.patient_data, extraction=extraction)


@router.get("/status", response_model=list[ModelStatusItem], summary="Model readiness status")
async def status_all():
    items = []
    for key in DISEASES:
        meta = load_meta(key) if disease_ready(key) else None
        items.append(ModelStatusItem(
            disease=key,
            disease_label=DISEASE_LABELS[key],
            trained=disease_ready(key),
            trained_at=meta.get("trained_at","") if meta else "",
            selected_features=meta.get("selected_features",[]) if meta else [],
            train_rows=meta.get("train_rows",0) if meta else 0,
            test_rows=meta.get("test_rows",0) if meta else 0,
        ))
    return items


@router.get("/metrics/{disease}", summary="Evaluation metrics for one disease")
async def get_metrics(disease: str):
    if disease not in DISEASES:
        raise HTTPException(404, f"Unknown disease: {disease}")
    m = load_metrics(disease)
    if not m:
        raise HTTPException(404, f"Metrics not found for {disease}. Train first.")
    return m


@router.get("/features/{disease}", summary="Feature importance for one disease")
async def get_features(disease: str):
    if disease not in DISEASES:
        raise HTTPException(404, f"Unknown disease: {disease}")
    meta = load_meta(disease)
    if not meta:
        raise HTTPException(404, f"Training metadata not found for {disease}.")
    return {"disease": disease, "feature_importance": meta.get("feature_importance", [])}


@router.get("/form-fields", summary="Form field definitions for all diseases")
async def form_fields():
    """Returns field schemas so the frontend can build forms dynamically."""
    return {key: cfg.form_fields for key, cfg in DISEASE_CONFIGS.items()}


@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "service": "Brain Disease Prediction API"}


@router.post("/debug-pdf", summary="Diagnose PDF extraction — returns raw text per page")
async def debug_pdf(file: UploadFile = File(...)):
    """
    Upload any PDF and get back the raw text pdfplumber extracted per page,
    plus any tables found.  Use this to see exactly what the extractor sees
    so label patterns can be tuned to match your report format.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file.")

    pages_info = []
    all_text_parts = []

    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                tables = page.extract_tables() or []
                table_rows = []
                for tbl in tables:
                    for row in tbl:
                        cells = [str(c or "").strip() for c in row if c is not None and str(c).strip()]
                        if cells:
                            table_rows.append(" | ".join(cells))

                pages_info.append({
                    "page": i + 1,
                    "text_length": len(page_text),
                    "text_preview": page_text[:1000],
                    "table_rows": table_rows[:30],
                })
                if page_text:
                    all_text_parts.append(page_text)
                if table_rows:
                    all_text_parts.append("\n".join(table_rows))
    except Exception as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"pdfplumber could not open this file: {exc}")

    full_text = "\n".join(all_text_parts)
    # normalise
    norm = "\n".join(
        re.sub(r"[^\S\n]+", " ", ln).strip()
        for ln in full_text.split("\n") if ln.strip()
    )

    # also run extraction so we can show which fields matched and which didn't
    from app.services.pdf_extractor import extract_pdf as _extract
    ex = _extract(raw)

    return {
        "total_pages":    len(pages_info),
        "total_text_len": len(full_text),
        "fields_extracted": len(ex.found),
        "fields_found":   ex.found,
        "fields_missing": ex.missing,
        "normalised_text_preview": norm[:3000],
        "pages": pages_info,
    }
