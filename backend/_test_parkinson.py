"""Generate a Parkinson's-indicating PDF and confirm the model flags Parkinson's."""
import httpx
from reportlab.pdfgen import canvas

# Voice values clearly indicating Parkinson's (irregular voice)
# everything else kept normal so other diseases stay low.
LINES = [
    "MEDICAL REPORT",
    "Patient Name: Test Patient",
    "Patient ID: P-2026-009",
    "Age: 64 years",
    "Gender: Male",
    "Marital Status: Married",
    "Occupation: Private Sector",
    "Residence: Urban",
    "",
    "Vital Signs & Blood Reports",
    "Blood Glucose: 98 mg/dL",
    "BMI: 24.0",
    "Hypertension: No",
    "Heart Disease: No",
    "Smoking Status: Never smoked",
    "",
    "Neurological Symptoms",
    "Headache: No",
    "Seizure: No",
    "Memory Loss: No",
    "Speech Difficulty: No",
    "Vision Problems: No",
    "Balance Issues: No",
    "Fatigue: No",
    "Nausea: No",
    "",
    "Cognitive Assessment",
    "MMSE: 29",
    "CDR Sum of Boxes: 0",
    "ADAS-Cog 11: 8",
    "Functional Activities: 0",
    "Education: 14 years",
    "APOE4: 0",
    "Hippocampus Volume: 7100 mm3",
    "Whole Brain Volume: 1.15",
    "",
    "Voice Measurements",
    "MDVP:Fo(Hz) 145",
    "MDVP:Fhi(Hz) 185",
    "MDVP:Flo(Hz) 100",
    "MDVP:Jitter 0.018",
    "MDVP:Shimmer 0.075",
    "NHR 0.045",
    "HNR 16",
    "RPDE 0.68",
    "DFA 0.79",
    "Spread1 -4.2",
    "Spread2 0.32",
    "D2 2.8",
    "",
    "Imaging",
    "Tumor Size: 0 cm",
    "Tumor Location: Not known",
]


def make_pdf(path):
    c = canvas.Canvas(path)
    y = 780
    c.setFont("Helvetica", 10)
    for line in LINES:
        if line:
            c.drawString(50, y, line)
        y -= 15
        if y < 50:
            c.showPage(); c.setFont("Helvetica", 10); y = 780
    c.save()


pdf_path = "/tmp/parkinsons_report.pdf"
make_pdf(pdf_path)
print("Generated:", pdf_path)

with open(pdf_path, "rb") as f:
    resp = httpx.post("http://localhost:8000/api/predict-pdf",
        files={"file": ("parkinsons_report.pdf", f, "application/pdf")}, timeout=120)
print("HTTP", resp.status_code)
data = resp.json() if resp.is_success else resp.text
if isinstance(data, dict):
    ex = data.get("extraction") or {}
    print(f"Extraction: {ex.get('fields_read')} fields read, {ex.get('fields_missing')} missing")
    print(f"\nPrimary risk: {data.get('primary_risk_disease')}")
    print(f"Summary: {data.get('summary_message')}")
    for r in sorted(data.get("results", []), key=lambda r: -r["risk_score"]):
        print(f"  {r['disease_label']}: {r['risk_score']}% ({r['risk_category']})")
