"""
Extract structured patient data from a medical report PDF.

Design goals (v2):
  1. Handle any layout — key:value on one line, key on one line and value on
     the next, multi-column table, pipe-separated, dash-separated, etc.
  2. Wide look-ahead windows (200 chars) so values aren't cut off.
  3. 100+ label synonyms so non-standard report headings still match.
  4. Normalise the full text (collapse whitespace, remove control chars)
     so regex matching is layout-independent.
  5. All fields default to clinically safe values so predictions never
     crash even when a field is absent from the report.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field

import pdfplumber

log = logging.getLogger("brain_disease.pdf")

# ── Defaults ───────────────────────────────────────────────────────────────────
# Every key that any disease model needs.  When the PDF doesn't mention a field
# the model receives this safe default (imputer will further fill NaN numerics
# with training-set medians at inference time).
DEFAULTS: dict = {
    # ── Identity
    "patient_name": "", "patient_id": "", "patient_age": "", "patient_gender": "Male",
    # ── Stroke
    "age": "", "gender": "Male",
    "hypertension": "0", "heart_disease": "0",
    "avg_glucose_level": "", "bmi": "",
    "smoking_status": "never smoked", "ever_married": "Yes",
    "work_type": "Private", "Residence_type": "Urban",
    # ── Neurological symptoms  (shared stroke / tumor)
    "Headache": "0", "Seizure": "0", "MemoryLoss": "0",
    "SpeechDifficulty": "0", "VisionProblems": "0",
    "BalanceIssues": "0", "Fatigue": "0", "Nausea": "0",
    # ── Alzheimer's
    "Age": "", "Gender": "0", "Ethnicity": "0", "EducationLevel": "2",
    "BMI": "25", "Smoking": "0",
    "AlcoholConsumption": "5", "PhysicalActivity": "4",
    "DietQuality": "5", "SleepQuality": "7",
    "FamilyHistoryAlzheimers": "0", "CardiovascularDisease": "0",
    "Diabetes": "0", "Depression": "0", "HeadInjury": "0", "Hypertension": "0",
    "SystolicBP": "120", "DiastolicBP": "80",
    "CholesterolTotal": "200", "CholesterolLDL": "100",
    "CholesterolHDL": "50", "CholesterolTriglycerides": "150",
    "MMSE": "27", "FunctionalAssessment": "7", "MemoryComplaints": "0",
    "BehavioralProblems": "0", "ADL": "8",
    "Confusion": "0", "Disorientation": "0", "PersonalityChanges": "0",
    "DifficultyCompletingTasks": "0", "Forgetfulness": "0",
    # ── Parkinson's voice
    "MDVP_Fo": "154",  "MDVP_Fhi": "197",  "MDVP_Flo": "116",
    "MDVP_Jitter": "0.004", "MDVP_Jitter_Abs": "0.00002",
    "MDVP_RAP": "0.002", "MDVP_PPQ": "0.002", "Jitter_DDP": "0.006",
    "MDVP_Shimmer": "0.025", "MDVP_Shimmer_dB": "0.25",
    "Shimmer_APQ3": "0.012", "Shimmer_APQ5": "0.015",
    "MDVP_APQ": "0.015", "Shimmer_DDA": "0.035",
    "NHR": "0.015", "HNR": "22",
    "RPDE": "0.47", "DFA": "0.72",
    "Spread1": "-5.5", "Spread2": "0.22", "D2": "2.4", "PPE": "0.15",
    # ── Brain tumor
    "TumorSize": "0", "Location": "0",
}

# Fields derived from other fields via _sync_fields — skip direct assignment.
_SYNCED = {
    "age", "Age", "gender", "Gender",
    "BMI", "Hypertension", "CardiovascularDisease", "Smoking",
}

# ── Field specifications ───────────────────────────────────────────────────────
#  type    : "num" | "yesno" | "text" | "gender" | "cat"
#  labels  : list of regex patterns (case-insensitive) — tried in order
#  options : dict  { output_value: [keyword_regex, …] }  (cat only)
#  window  : chars to scan after the matched label (default 200)
SPECS: list[dict] = [

    # ── Identity ──────────────────────────────────────────────────────────────
    {"key": "patient_name", "type": "text", "labels": [
        r"patient['\s]*s?\s*name", r"name\s*of\s*patient",
        r"full\s*name", r"pt\.?\s*name", r"\bname\b"]},

    {"key": "patient_id", "type": "text", "labels": [
        r"patient['\s]*s?\s*i\.?d", r"reg(?:istration)?\.?\s*(?:no|number|#)",
        r"\bmrn\b", r"patient\s*no\.?", r"hospital\s*(?:no|id|number)",
        r"uhid", r"chart\s*(?:no|number|#)", r"case\s*(?:no|number|#)"]},

    {"key": "patient_age", "type": "num", "labels": [
        r"patient\s*age", r"\bage\b\s*(?:\(?\s*yr?s?\.?\)?)?",
        r"age\s*(?:at\s*admission|at\s*presentation|years?)",
        r"years?\s*old"]},

    {"key": "patient_gender", "type": "gender", "labels": [
        r"gender", r"\bsex\b", r"biological\s*sex"]},

    # ── Vitals / Stroke ───────────────────────────────────────────────────────
    {"key": "avg_glucose_level", "type": "num", "labels": [
        r"(?:average|avg|mean|random|fasting|pp|postprandial|ppbs|rbs|fbs)\s*glucose",
        r"blood\s*glucose(?:\s*level)?",
        r"glucose(?:\s*level)?",
        r"blood\s*sugar(?:\s*level)?",
        r"\bsugar\b",
        r"serum\s*glucose",
        r"\bhba1c\b",              # glycated Hb — not exact but common proxy
        r"glycaemi[ac]"]},

    {"key": "bmi", "type": "num", "labels": [
        r"body\s*mass\s*index",
        r"\bbmi\b",
        r"wt\s*/\s*ht\^?2",
        r"weight.*height\s*ratio"]},

    {"key": "hypertension", "type": "yesno", "labels": [
        r"hypertension(?:\s*h/o)?",
        r"high\s*blood\s*pressure",
        r"\bhbp\b",
        r"h/o\s*hypertension",
        r"history\s*of\s*hypertension",
        r"\bhtn\b",
        r"elevated\s*blood\s*pressure"]},

    {"key": "heart_disease", "type": "yesno", "labels": [
        r"heart\s*disease",
        r"cardiac\s*disease",
        r"coronary\s*(?:artery\s*)?disease",
        r"\bchd\b", r"\bcad\b", r"\bihd\b",
        r"ischaemic\s*heart",
        r"history\s*of\s*(?:heart|cardiac)",
        r"h/o\s*(?:heart|cardiac)",
        r"prior\s*(?:mi|myocardial\s*infarction)"]},

    {"key": "smoking_status", "type": "cat",
     "labels": [r"smoking\s*status", r"smoker\s*status",
                r"smoking\s*history", r"tobacco\s*use",
                r"cigarette\s*use", r"smok(?:ing|er)"],
     "options": {
         "never smoked":   [r"never\s*smoked?", r"non[- ]?smoker", r"no\s*history\s*of\s*smok",
                            r"tobacco\s*(?:use\s*)?[:=]\s*no", r"smok.*never"],
         "formerly smoked":[r"former(?:ly)?(?:\s*smok(?:ed?|er))?", r"ex[- ]?smoker",
                            r"quit(?:ted)?(?:\s*smoking)?", r"past\s*smoker",
                            r"ex[- ]?tobacco", r"reformed\s*smoker"],
         "smokes":         [r"\bsmokes?\b", r"current(?:ly)?\s*smok",
                            r"active\s*smoker", r"tobacco\s*(?:use\s*)?[:=]\s*yes",
                            r"cigarette(?:s)?\s*(?:per\s*day|daily)"],
         "Unknown":        [r"unknown", r"not\s*stated", r"not\s*known", r"\bnil\b",
                            r"n/?a", r"not\s*applicable"],
     }},

    {"key": "ever_married", "type": "cat",
     "labels": [r"marital\s*status", r"ever\s*married", r"\bmarried\b",
                r"civil\s*status", r"matrimonial"],
     "options": {
         "Yes": [r"married", r"\byes\b", r"\bwed\b", r"spouse", r"partner"],
         "No":  [r"unmarried", r"single", r"divorced", r"widow(?:er)?",
                 r"never\s*married", r"\bno\b", r"separated"],
     }},

    {"key": "work_type", "type": "cat",
     "labels": [r"work\s*type", r"occupation(?:al\s*status)?",
                r"employment\s*(?:type|status)?", r"profession",
                r"job\s*type", r"nature\s*of\s*work"],
     "options": {
         "Private":       [r"private\s*(?:sector|company|firm|employee)?", r"salaried"],
         "Govt_job":      [r"govt\.?(?:\s*job)?", r"government", r"public\s*sector",
                          r"civil\s*servant"],
         "Self-employed": [r"self[- ]?employed", r"business(?:man|woman|person)?",
                          r"own\s*business", r"entrepreneur", r"freelanc"],
         "children":      [r"student", r"child(?:ren)?", r"\bschool\b", r"\bminor\b"],
         "Never_worked":  [r"never\s*worked", r"unemployed", r"not\s*working",
                          r"homemaker", r"housewife", r"retired"],
     }},

    {"key": "Residence_type", "type": "cat",
     "labels": [r"residence\s*(?:type)?", r"residential\s*area",
                r"area\s*of\s*residence", r"locality", r"domicile",
                r"address\s*type"],
     "options": {
         "Urban": [r"urban", r"\bcity\b", r"metropolitan", r"\btown\b",
                  r"municipality"],
         "Rural": [r"rural", r"village", r"countryside", r"non[- ]?urban"],
     }},

    # ── Neurological Symptoms ─────────────────────────────────────────────────
    {"key": "Headache", "type": "yesno", "labels": [
        r"head\s*ache", r"headache", r"cephalgi[ae]",
        r"persistent\s*head\s*pain", r"migraine"]},

    {"key": "Seizure", "type": "yesno", "labels": [
        r"seizure", r"convulsion", r"epileptic?\s*(?:attack|episode)?",
        r"\bfits?\b", r"\bictal\b", r"tonic[- ]clonic"]},

    {"key": "MemoryLoss", "type": "yesno", "labels": [
        r"memory\s*loss", r"memory\s*impair(?:ment)?",
        r"amnes(?:ia)?", r"forgetful(?:ness)?",
        r"cognitive\s*decline", r"difficulty\s*remembering"]},

    {"key": "SpeechDifficulty", "type": "yesno", "labels": [
        r"speech\s*(?:difficulty|impair|problem|defect)",
        r"slurred\s*speech", r"\baphasia\b", r"\bdysphasia\b",
        r"\bdysarthria\b", r"difficulty\s*(?:in\s*)?speaking",
        r"communication\s*(?:problem|difficulty)"]},

    {"key": "VisionProblems", "type": "yesno", "labels": [
        r"vision\s*(?:problem|impair|disturbance|loss|change)",
        r"visual\s*(?:problem|impair|defect|field)",
        r"blurred?\s*vision", r"diplopia", r"poor\s*vision",
        r"sight\s*(?:problem|loss)", r"optic\s*(?:problem|defect)"]},

    {"key": "BalanceIssues", "type": "yesno", "labels": [
        r"balance\s*(?:issue|problem|difficulty|disturb)",
        r"gait\s*(?:issue|difficulty|disturb|problem|abnormal)",
        r"unsteady\s*(?:gait|walk)?", r"\bataxia\b",
        r"difficulty\s*walk", r"poor\s*balance"]},

    {"key": "Fatigue", "type": "yesno", "labels": [
        r"fatigue", r"tiredness", r"\bletharg", r"\bweakness\b",
        r"chronic\s*tiredness", r"excessive\s*tiredness",
        r"easy\s*fatigability", r"exhaustion"]},

    {"key": "Nausea", "type": "yesno", "labels": [
        r"nausea", r"vomit(?:ing)?", r"\bemesis\b",
        r"feeling\s*sick", r"motion\s*sickness"]},

    # ── Alzheimer's cognitive + clinical ─────────────────────────────────────
    {"key": "MMSE", "type": "num", "labels": [
        r"mmse\s*(?:score)?", r"mini[- ]?mental\s*state\s*(?:exam|examination|score)",
        r"mini[- ]?mental\s*(?:test|assessment)",
        r"m\.?m\.?s\.?e\.?"]},

    {"key": "FunctionalAssessment", "type": "num", "labels": [
        r"functional\s*assessment(?:\s*score)?",
        r"functional\s*activities\s*(?:score|questionnaire)?",
        r"\bfaq\b", r"functional\s*ability\s*score",
        r"activities\s*(?:assessment|score)"]},

    {"key": "ADL", "type": "num", "labels": [
        r"\badl\b(?:\s*score)?",
        r"activities\s*of\s*daily\s*living(?:\s*score)?",
        r"daily\s*living\s*(?:score|assessment)",
        r"barthel\s*index", r"\bbadl\b", r"\biadl\b"]},

    {"key": "MemoryComplaints", "type": "yesno", "labels": [
        r"memory\s*complaint", r"subjective\s*memory\s*complaint",
        r"self[- ]?reported\s*memory"]},

    {"key": "BehavioralProblems", "type": "yesno", "labels": [
        r"behavi(?:ou?ral?)\s*problem",
        r"psychological\s*(?:symptom|problem|issue)",
        r"agitation", r"\baggression\b", r"mood\s*(?:change|problem|disturbance)"]},

    {"key": "Confusion", "type": "yesno", "labels": [
        r"\bconfusion\b", r"\bconfused\b",
        r"acute\s*confusion", r"disoriented\s*and\s*confused",
        r"altered\s*sensorium"]},

    {"key": "Disorientation", "type": "yesno", "labels": [
        r"disorientation", r"disoriented",
        r"disoriented\s*(?:to\s*)?(?:time|place|person)"]},

    {"key": "PersonalityChanges", "type": "yesno", "labels": [
        r"personality\s*change", r"character\s*change",
        r"behavi(?:ou?r)\s*change", r"mood\s*change",
        r"change\s*in\s*personality"]},

    {"key": "DifficultyCompletingTasks", "type": "yesno", "labels": [
        r"difficulty\s*completing\s*tasks?",
        r"difficulty\s*with\s*tasks?",
        r"unable\s*to\s*complete\s*tasks?",
        r"task\s*completion\s*(?:difficulty|problem)"]},

    {"key": "Forgetfulness", "type": "yesno", "labels": [
        r"forgetful(?:ness)?",
        r"forgetting\s*things",
        r"memory\s*(?:gap|lapse|complaint)"]},

    {"key": "FamilyHistoryAlzheimers", "type": "yesno", "labels": [
        r"family\s*history\s*(?:of\s*)?alzheimer",
        r"f\.?h\.?\s*(?:of\s*)?alzheimer",
        r"family\s*history\s*(?:of\s*)?dementia",
        r"hereditary\s*(?:alzheimer|dementia)",
        r"family\s*history"]},

    {"key": "Diabetes", "type": "yesno", "labels": [
        r"\bdiabetes\s*(?:mellitus)?\b",
        r"\bdiabetic\b",
        r"h/o\s*(?:dm|diabetes)",
        r"history\s*of\s*(?:dm|diabetes)",
        r"\bdm\s*(?:type\s*[12])?\b",
        r"type\s*[12]\s*(?:dm|diabetes)"]},

    {"key": "Depression", "type": "yesno", "labels": [
        r"\bdepression\b", r"\bdepressed\b",
        r"depressive\s*(?:disorder|episode|illness)",
        r"psychiatric\s*history.*depress",
        r"low\s*mood", r"h/o\s*depression"]},

    {"key": "HeadInjury", "type": "yesno", "labels": [
        r"head\s*injury",
        r"traumatic\s*brain\s*injury", r"\btbi\b",
        r"cranial\s*(?:trauma|injury)",
        r"brain\s*trauma",
        r"past\s*head\s*(?:injury|trauma)"]},

    {"key": "BMI", "type": "num", "labels": [
        r"body\s*mass\s*index", r"\bbmi\b",
        r"wt.*ht\^?2"]},

    {"key": "Smoking", "type": "yesno", "labels": [
        r"\bsmoking\b(?!\s*status)",
        r"tobacco\s*(?:use|habit|consumption)",
        r"smoker\s*[:=]",
        r"cigarette\s*(?:habit|use)"]},

    {"key": "AlcoholConsumption", "type": "num", "labels": [
        r"alcohol\s*consumption(?:\s*(?:units?|ml))?",
        r"alcohol\s*intake",
        r"weekly\s*alcohol",
        r"units?\s*of\s*alcohol",
        r"\balcohol\b"]},

    {"key": "PhysicalActivity", "type": "num", "labels": [
        r"physical\s*activity(?:\s*hours?)?",
        r"exercise\s*(?:level|hours?|per\s*week|per\s*day)",
        r"weekly\s*exercise",
        r"activity\s*level"]},

    {"key": "DietQuality", "type": "num", "labels": [
        r"diet\s*quality(?:\s*score)?",
        r"dietary\s*(?:quality|score|assessment)",
        r"nutrition\s*score"]},

    {"key": "SleepQuality", "type": "num", "labels": [
        r"sleep\s*quality(?:\s*score)?",
        r"sleep\s*(?:score|assessment|pattern)",
        r"quality\s*of\s*sleep"]},

    {"key": "SystolicBP", "type": "num", "labels": [
        r"systolic\s*(?:bp|blood\s*pressure|pressure)",
        r"\bsbp\b", r"\bsystolic\b",
        r"bp\s*systolic",
        r"blood\s*pressure\s*systolic"]},

    {"key": "DiastolicBP", "type": "num", "labels": [
        r"diastolic\s*(?:bp|blood\s*pressure|pressure)",
        r"\bdbp\b", r"\bdiastolic\b",
        r"bp\s*diastolic",
        r"blood\s*pressure\s*diastolic"]},

    {"key": "CholesterolTotal", "type": "num", "labels": [
        r"total\s*cholesterol",
        r"serum\s*cholesterol",
        r"cholesterol\s*total",
        r"cholesterol\s*(?:level|value)(?!\s*(?:ldl|hdl|triglyc))"]},

    {"key": "CholesterolLDL", "type": "num", "labels": [
        r"ldl\s*cholesterol",
        r"low[- ]density\s*lipoprotein",
        r"cholesterol\s*ldl",
        r"\bldl\b"]},

    {"key": "CholesterolHDL", "type": "num", "labels": [
        r"hdl\s*cholesterol",
        r"high[- ]density\s*lipoprotein",
        r"cholesterol\s*hdl",
        r"\bhdl\b"]},

    {"key": "CholesterolTriglycerides", "type": "num", "labels": [
        r"triglycerides?",
        r"serum\s*triglycerides?",
        r"tg\s*level",
        r"\bvldl\b",           # VLDL ≈ TG/5 — common proxy in reports
        r"cholesterol\s*triglycerides?"]},

    {"key": "EducationLevel", "type": "cat",
     "labels": [r"education(?:al)?\s*(?:level|qualification|status)?",
                r"highest\s*education", r"years?\s*of\s*education",
                r"schooling"],
     "options": {
         "0": [r"no\s*(?:formal\s*)?education", r"illiterate", r"\bnone\b"],
         "1": [r"high\s*school", r"secondary", r"ssc", r"hsc",
               r"10(?:th|\.?)(?:\s*std)?", r"12(?:th|\.?)(?:\s*std)?",
               r"matriculat"],
         "2": [r"bachelor", r"undergrad", r"b\.?(?:sc|a|tech|e|com)",
               r"degree"],
         "3": [r"master", r"postgrad", r"phd", r"m\.?(?:sc|a|tech|e|com|b\.?b\.?s)",
               r"doctorate", r"higher\s*education"],
     }},

    {"key": "Ethnicity", "type": "cat",
     "labels": [r"ethnicity", r"ethnic\s*(?:background|group|origin)",
                r"\brace\b", r"nationality"],
     "options": {
         "0": [r"caucasian", r"white", r"european"],
         "1": [r"african[- ]american", r"black\s*(?:african)?",
               r"\bafrican\b"],
         "2": [r"asian", r"east\s*asian", r"south\s*asian",
               r"chinese", r"indian", r"japanese"],
         "3": [r"hispanic", r"latino", r"mixed", r"other"],
     }},

    {"key": "Hypertension", "type": "yesno", "labels": [
        r"hypertension",
        r"high\s*blood\s*pressure",
        r"\bhtn\b", r"\bhbp\b",
        r"elevated\s*bp"]},

    {"key": "CardiovascularDisease", "type": "yesno", "labels": [
        r"cardiovascular\s*disease",
        r"cvd\b", r"heart\s*disease",
        r"coronary\s*(?:artery\s*)?disease",
        r"cardiac\s*(?:disease|condition|history)",
        r"ischaemic\s*heart", r"\bchd\b", r"\bcad\b"]},

    # ── Parkinson's voice measurements ────────────────────────────────────────
    {"key": "MDVP_Fo", "type": "num", "labels": [
        r"mdvp[:\s]*fo\b", r"mdvp[:\s]*fo\s*\(hz\)",
        r"average\s*vocal\s*(?:fundamental\s*)?freq",
        r"fundamental\s*freq", r"avg\s*vocal\s*freq",
        r"f0\s*(?:hz)?", r"fo\s*\(hz\)"]},

    {"key": "MDVP_Fhi", "type": "num", "labels": [
        r"mdvp[:\s]*fhi", r"max(?:imum)?\s*vocal\s*freq",
        r"highest\s*vocal\s*freq", r"fhi\s*\(hz\)"]},

    {"key": "MDVP_Flo", "type": "num", "labels": [
        r"mdvp[:\s]*flo", r"min(?:imum)?\s*vocal\s*freq",
        r"lowest\s*vocal\s*freq", r"flo\s*\(hz\)"]},

    {"key": "MDVP_Jitter", "type": "num", "labels": [
        r"mdvp[:\s]*jitter\s*(?:\(?\s*%\s*\)?)?(?!\s*\(abs\))",
        r"jitter\s*percent(?:age)?", r"jitter\s*\(%\)"]},

    {"key": "MDVP_Jitter_Abs", "type": "num", "labels": [
        r"mdvp[:\s]*jitter\s*\(abs\)",
        r"absolute\s*jitter", r"jitter\s*abs(?:olute)?"]},

    {"key": "MDVP_RAP", "type": "num", "labels": [
        r"mdvp[:\s]*rap\b",
        r"relative\s*average\s*perturbation",
        r"\brap\b"]},

    {"key": "MDVP_PPQ", "type": "num", "labels": [
        r"mdvp[:\s]*ppq\b",
        r"five[- ]point\s*period\s*perturb",
        r"\bppq\b"]},

    {"key": "Jitter_DDP", "type": "num", "labels": [
        r"jitter[:\s]*ddp\b",
        r"average\s*absolute\s*diff(?:erence)?\s*of\s*jitter",
        r"\bddp\b"]},

    {"key": "MDVP_Shimmer", "type": "num", "labels": [
        r"mdvp[:\s]*shimmer(?!\s*(?:\(db\)|apq|dda))",
        r"shimmer(?!\s*(?:apq|dda|db))",
        r"amplitude\s*variation"]},

    {"key": "MDVP_Shimmer_dB", "type": "num", "labels": [
        r"mdvp[:\s]*shimmer\s*\(db\)",
        r"shimmer\s*\(db\)", r"shimmer\s*db"]},

    {"key": "Shimmer_APQ3", "type": "num", "labels": [
        r"shimmer[:\s]*apq3\b", r"\bapq3\b"]},

    {"key": "Shimmer_APQ5", "type": "num", "labels": [
        r"shimmer[:\s]*apq5\b", r"\bapq5\b"]},

    {"key": "MDVP_APQ", "type": "num", "labels": [
        r"mdvp[:\s]*apq\b(?!3|5)",
        r"amplitude\s*perturbation\s*quotient(?!\s*(?:3|5))",
        r"\bapq\b(?!3|5)"]},

    {"key": "Shimmer_DDA", "type": "num", "labels": [
        r"shimmer[:\s]*dda\b",
        r"average\s*absolute\s*diff.*amplitude",
        r"\bdda\b"]},

    {"key": "NHR", "type": "num", "labels": [
        r"noise[- ]to[- ]harmonic[s]?\s*ratio",
        r"\bnhr\b"]},

    {"key": "HNR", "type": "num", "labels": [
        r"harmonic[s]?[- ]to[- ]noise\s*ratio",
        r"\bhnr\b"]},

    {"key": "RPDE", "type": "num", "labels": [
        r"recurrence\s*period\s*density\s*entropy",
        r"\brpde\b"]},

    {"key": "DFA", "type": "num", "labels": [
        r"detrended\s*fluctuation\s*analysis",
        r"signal\s*fractal\s*scaling",
        r"\bdfa\b"]},

    {"key": "Spread1", "type": "num", "labels": [
        r"spread\s*1\b", r"\bspread1\b",
        r"nonlinear\s*(?:measure|feature)\s*1"]},

    {"key": "Spread2", "type": "num", "labels": [
        r"spread\s*2\b", r"\bspread2\b",
        r"nonlinear\s*(?:measure|feature)\s*2"]},

    {"key": "D2", "type": "num", "labels": [
        r"correlation\s*dimension",
        r"\bd2\b(?!\d)"]},

    {"key": "PPE", "type": "num", "labels": [
        r"pitch\s*period\s*entropy",
        r"\bppe\b"]},

    # ── Brain tumor indicators ─────────────────────────────────────────────────
    {"key": "TumorSize", "type": "num", "labels": [
        r"tumou?r\s*(?:size|dimension|diameter|measurement)",
        r"lesion\s*(?:size|dimension|diameter)",
        r"size\s*of\s*(?:tumou?r|lesion|mass)",
        r"mass\s*(?:size|dimension|diameter)",
        r"tumou?r\s*(?:measures?|cm)"]},

    {"key": "Location", "type": "cat",
     "labels": [r"tumou?r\s*location", r"lesion\s*location",
                r"location\s*of\s*(?:tumou?r|lesion|mass)",
                r"affected\s*lobe", r"brain\s*region",
                r"tumou?r\s*(?:site|lobe|area)", r"location"],
     "options": {
         "1": [r"frontal\s*lobe", r"\bfrontal\b"],
         "2": [r"temporal\s*lobe", r"\btemporal\b"],
         "3": [r"parietal", r"occipital", r"cerebellar",
               r"cerebellum", r"brainstem", r"brain\s*stem",
               r"posterior", r"\bother\b"],
         "0": [r"not\s*known", r"unknown", r"not\s*(?:specified|identified)",
               r"\bnil\b", r"n/?a"],
     }},
]

# ── Display labels ─────────────────────────────────────────────────────────────
DISPLAY_LABELS: dict[str, str] = {
    spec["key"]: spec["key"].replace("_", " ") for spec in SPECS
}
DISPLAY_LABELS.update({
    "patient_name": "Patient Name", "patient_id": "Patient ID",
    "patient_age": "Age", "patient_gender": "Gender",
    "avg_glucose_level": "Avg Glucose Level (mg/dL)", "bmi": "BMI",
    "hypertension": "Hypertension", "heart_disease": "Heart Disease",
    "smoking_status": "Smoking Status", "ever_married": "Marital Status",
    "work_type": "Work Type", "Residence_type": "Residence Type",
    "Headache": "Headache", "Seizure": "Seizure", "MemoryLoss": "Memory Loss",
    "SpeechDifficulty": "Speech Difficulty", "VisionProblems": "Vision Problems",
    "BalanceIssues": "Balance Issues", "Fatigue": "Fatigue", "Nausea": "Nausea",
    "MMSE": "MMSE Score", "FunctionalAssessment": "Functional Assessment",
    "ADL": "ADL Score", "MemoryComplaints": "Memory Complaints",
    "BehavioralProblems": "Behavioral Problems",
    "FamilyHistoryAlzheimers": "Family History (Alzheimer's)",
    "CardiovascularDisease": "Cardiovascular Disease",
    "SystolicBP": "Systolic BP (mmHg)", "DiastolicBP": "Diastolic BP (mmHg)",
    "CholesterolTotal": "Total Cholesterol", "CholesterolLDL": "LDL Cholesterol",
    "CholesterolHDL": "HDL Cholesterol", "CholesterolTriglycerides": "Triglycerides",
    "EducationLevel": "Education Level", "Ethnicity": "Ethnicity",
    "MDVP_Fo": "MDVP:Fo(Hz)", "MDVP_Fhi": "MDVP:Fhi(Hz)", "MDVP_Flo": "MDVP:Flo(Hz)",
    "MDVP_Jitter": "MDVP:Jitter(%)", "MDVP_Jitter_Abs": "MDVP:Jitter(Abs)",
    "MDVP_RAP": "MDVP:RAP", "MDVP_PPQ": "MDVP:PPQ", "Jitter_DDP": "Jitter:DDP",
    "MDVP_Shimmer": "MDVP:Shimmer", "MDVP_Shimmer_dB": "MDVP:Shimmer(dB)",
    "Shimmer_APQ3": "Shimmer:APQ3", "Shimmer_APQ5": "Shimmer:APQ5",
    "MDVP_APQ": "MDVP:APQ", "Shimmer_DDA": "Shimmer:DDA",
    "NHR": "NHR", "HNR": "HNR", "RPDE": "RPDE", "DFA": "DFA",
    "Spread1": "Spread1", "Spread2": "Spread2", "D2": "D2", "PPE": "PPE",
    "TumorSize": "Tumor Size (cm)", "Location": "Tumor Location",
})

# ── ExtractionResult ───────────────────────────────────────────────────────────
@dataclass
class ExtractionResult:
    patient_data: dict = field(default_factory=dict)
    found: list  = field(default_factory=list)   # [{key, label, value}]
    missing: list = field(default_factory=list)  # [display-label strings]
    page_count: int = 0
    text_length: int = 0
    text_preview: str = ""

# ── Regex helpers ──────────────────────────────────────────────────────────────
_NUM_RE  = re.compile(r"[-+]?\d{1,4}(?:[.,]\d+)?")
_YES_RE  = re.compile(
    r"\b(yes|present|positive|true|confirmed|detected|seen|noted|"
    r"reactive|abnormal|elevated|increased|raised|high)\b", re.I)
_NO_RE   = re.compile(
    r"\b(no\b|absent|negative|false|nil|none|denied|normal|"
    r"within\s*(?:normal\s*)?limits?|wnl|not\s*(?:present|seen|detected|found))\b", re.I)
_BIN_RE  = re.compile(r"(?<!\d)([01])(?!\d)")
_BP_RE   = re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})")


def _norm_text(raw: str) -> str:
    """Collapse runs of whitespace to a single space; strip control chars."""
    return re.sub(r"[^\S\n]+", " ", raw).strip()


def _num_after(text: str, pos: int, window: int = 200) -> str | None:
    """
    Find the first number after pos within window characters.
    Skip pure unit tokens like mg/dL, mmHg, cm, etc. that appear right
    before the actual number (e.g. "Glucose mg/dL: 142").
    """
    segment = text[pos: pos + window]
    # Try to skip over unit tokens that precede the value
    segment = re.sub(r"(?:mg/dl|mmhg|kg/m.2|cm|mm|l|ml|%|units?|h/o|n/a)",
                     " ", segment, flags=re.I)
    m = _NUM_RE.search(segment)
    if not m:
        return None
    val = m.group(0).replace(",", ".")
    # Sanity: must be a valid float
    try:
        float(val)
    except ValueError:
        return None
    return val


def _yesno_after(text: str, pos: int, window: int = 150, key: str = "") -> str | None:
    seg = text[pos: pos + window]
    # BP reading shortcut for hypertension fields
    if key in ("hypertension", "Hypertension"):
        bp = _BP_RE.search(seg)
        if bp:
            sys_b, dia_b = int(bp.group(1)), int(bp.group(2))
            return "1" if (sys_b >= 140 or dia_b >= 90) else "0"
    yes = _YES_RE.search(seg)
    no  = _NO_RE.search(seg)
    if yes and no:
        return "1" if yes.start() < no.start() else "0"
    if yes:
        return "1"
    if no:
        return "0"
    # bare 0/1
    bm = _BIN_RE.search(seg)
    if bm:
        return bm.group(1)
    return None


def _gender_after(text: str, pos: int, window: int = 60) -> str | None:
    seg = text[pos: pos + window]
    if re.search(r"\bfemale\b|\bf\b(?!\w)", seg, re.I):
        return "Female"
    if re.search(r"\bmale\b|\bm\b(?!\w)", seg, re.I):
        return "Male"
    if re.search(r"\bother\b|trans(?:gender)?", seg, re.I):
        return "Other"
    return None


def _text_after(text: str, pos: int, window: int = 120) -> str | None:
    seg = text[pos: pos + window]
    # take up to the first newline or punctuation boundary
    rest = seg.split("\n")[0].lstrip(": \t-—").strip()
    if not rest or len(rest) > 100:
        return None
    return rest


def _cat_after(text: str, pos: int, options: dict, window: int = 150) -> str | None:
    seg = text[pos: pos + window]
    best_value, best_pos = None, None
    for value, keywords in options.items():
        for kw in keywords:
            m = re.search(kw, seg, re.I)
            if m and (best_pos is None or m.start() < best_pos):
                best_value, best_pos = value, m.start()
    return best_value


def _extract_field(spec: dict, text: str) -> tuple[str | None, str]:
    """
    Try every label pattern for this spec.  On the first match that also
    yields a non-empty value, return (value, matched_label_text).

    Enhancement over v1: also tries the pattern on the NEXT LINE in case the
    value is on a separate line from the heading.
    """
    ftype   = spec["type"]
    window  = spec.get("window", 200)
    lines   = text.split("\n")

    for pat in spec["labels"]:
        # 1) search in the full normalised text
        m = re.search(pat, text, re.I)
        if not m:
            continue
        label = m.group(0)
        pos   = m.end()

        val = _dispatch(ftype, text, pos, window, spec.get("key", ""),
                        spec.get("options", {}))
        if val is not None and val != "":
            return val, label

        # 2) if value not found after the match, look at the NEXT line
        # (handles "Heading\nValue" layouts)
        match_line_idx = text[:m.start()].count("\n")
        if match_line_idx + 1 < len(lines):
            next_line = lines[match_line_idx + 1].strip()
            val = _dispatch(ftype, next_line + " ", 0, len(next_line) + 1,
                            spec.get("key", ""), spec.get("options", {}))
            if val is not None and val != "":
                return val, label

    return None, ""


def _dispatch(ftype, text, pos, window, key, options):
    if ftype == "num":
        return _num_after(text, pos, window)
    if ftype == "yesno":
        return _yesno_after(text, pos, window, key)
    if ftype == "gender":
        return _gender_after(text, pos, window)
    if ftype == "text":
        return _text_after(text, pos, window)
    if ftype == "cat":
        return _cat_after(text, pos, options, window)
    return None


def _sync_fields(data: dict) -> dict:
    """Propagate canonical patient fields into ML-mirror columns."""
    age = data.get("patient_age") or data.get("age")
    if age not in (None, ""):
        data["age"] = age
        data["Age"] = age

    gender = data.get("patient_gender") or data.get("gender")
    if gender:
        data["gender"] = gender
        data["Gender"] = "1" if str(gender).lower().startswith("f") else "0"

    for src, dst in [("bmi", "BMI"), ("hypertension", "Hypertension"),
                     ("heart_disease", "CardiovascularDisease")]:
        v = data.get(src)
        if v not in (None, ""):
            data[dst] = v

    smoke = data.get("smoking_status")
    if smoke:
        data["Smoking"] = "0" if smoke in ("never smoked", "Unknown") else "1"

    return data


def _build_table_text(page) -> str:
    """Convert pdfplumber tables into flat label: value pairs."""
    parts = []
    for table in (page.extract_tables() or []):
        for row in table:
            cells = [str(c or "").strip() for c in row]
            # treat 2-column rows as label: value
            non_empty = [c for c in cells if c]
            if len(non_empty) == 2:
                parts.append(f"{non_empty[0]}: {non_empty[1]}")
            elif non_empty:
                parts.append("  ".join(non_empty))
    return "\n".join(parts)


# ── Smart key-value fallback ──────────────────────────────────────────────────
# Maps normalised word stems → field key + extraction type
# Used when regex label patterns don't match (e.g. non-standard headings)
_KV_FALLBACK: list[tuple[str, str, str]] = [
    # (stem_pattern,          field_key,              type)
    # Identity
    ("name",                  "patient_name",          "text"),
    ("patient id",            "patient_id",            "text"),
    ("reg no",                "patient_id",            "text"),
    ("mrn",                   "patient_id",            "text"),
    ("uhid",                  "patient_id",            "text"),
    ("chart no",              "patient_id",            "text"),
    # Vitals
    ("age",                   "patient_age",           "num"),
    ("sex",                   "patient_gender",        "gender"),
    ("gender",                "patient_gender",        "gender"),
    ("glucose",               "avg_glucose_level",     "num"),
    ("blood sugar",           "avg_glucose_level",     "num"),
    ("fbs",                   "avg_glucose_level",     "num"),
    ("rbs",                   "avg_glucose_level",     "num"),
    ("hba1c",                 "avg_glucose_level",     "num"),
    ("bmi",                   "bmi",                   "num"),
    ("body mass",             "bmi",                   "num"),
    ("weight",                "bmi",                   "num"),  # fallback
    ("hypertension",          "hypertension",          "yesno"),
    ("htn",                   "hypertension",          "yesno"),
    ("high bp",               "hypertension",          "yesno"),
    ("heart disease",         "heart_disease",         "yesno"),
    ("cardiac",               "heart_disease",         "yesno"),
    ("cad",                   "heart_disease",         "yesno"),
    ("chd",                   "heart_disease",         "yesno"),
    ("smoking",               "smoking_status",        "cat_smoke"),
    ("tobacco",               "smoking_status",        "cat_smoke"),
    ("marital",               "ever_married",          "cat_married"),
    ("married",               "ever_married",          "cat_married"),
    ("occupation",            "work_type",             "cat_work"),
    ("residence",             "Residence_type",        "cat_res"),
    ("locality",              "Residence_type",        "cat_res"),
    # Symptoms
    ("headache",              "Headache",              "yesno"),
    ("seizure",               "Seizure",               "yesno"),
    ("convulsion",            "Seizure",               "yesno"),
    ("fits",                  "Seizure",               "yesno"),
    ("memory loss",           "MemoryLoss",            "yesno"),
    ("speech",                "SpeechDifficulty",      "yesno"),
    ("vision",                "VisionProblems",        "yesno"),
    ("balance",               "BalanceIssues",         "yesno"),
    ("fatigue",               "Fatigue",               "yesno"),
    ("nausea",                "Nausea",                "yesno"),
    ("vomit",                 "Nausea",                "yesno"),
    # Alzheimer's cognitive
    ("mmse",                  "MMSE",                  "num"),
    ("mini mental",           "MMSE",                  "num"),
    ("moca",                  "MMSE",                  "num"),  # common alias
    ("adl",                   "ADL",                   "num"),
    ("activities of daily",   "ADL",                   "num"),
    ("functional assessment", "FunctionalAssessment",  "num"),
    ("functional activities", "FunctionalAssessment",  "num"),
    ("functional ability",    "FunctionalAssessment",  "num"),
    ("memory complaint",      "MemoryComplaints",      "yesno"),
    ("behavioural",           "BehavioralProblems",    "yesno"),
    ("behavioral",            "BehavioralProblems",    "yesno"),
    ("confusion",             "Confusion",             "yesno"),
    ("disorient",             "Disorientation",        "yesno"),
    ("personality change",    "PersonalityChanges",    "yesno"),
    ("completing task",       "DifficultyCompletingTasks", "yesno"),
    ("forgetful",             "Forgetfulness",         "yesno"),
    ("family history",        "FamilyHistoryAlzheimers","yesno"),
    ("alzheimer",             "FamilyHistoryAlzheimers","yesno"),
    ("dementia",              "FamilyHistoryAlzheimers","yesno"),
    ("depression",            "Depression",            "yesno"),
    ("diabetes",              "Diabetes",              "yesno"),
    ("dm",                    "Diabetes",              "yesno"),
    ("head injury",           "HeadInjury",            "yesno"),
    ("tbi",                   "HeadInjury",            "yesno"),
    ("systolic",              "SystolicBP",            "num"),
    ("sbp",                   "SystolicBP",            "num"),
    ("diastolic",             "DiastolicBP",           "num"),
    ("dbp",                   "DiastolicBP",           "num"),
    ("total cholesterol",     "CholesterolTotal",      "num"),
    ("ldl",                   "CholesterolLDL",        "num"),
    ("hdl",                   "CholesterolHDL",        "num"),
    ("triglyceride",          "CholesterolTriglycerides","num"),
    ("alcohol",               "AlcoholConsumption",    "num"),
    ("physical activity",     "PhysicalActivity",      "num"),
    ("exercise",              "PhysicalActivity",      "num"),
    ("diet quality",          "DietQuality",           "num"),
    ("sleep quality",         "SleepQuality",          "num"),
    ("sleep",                 "SleepQuality",          "num"),
    ("education",             "EducationLevel",        "cat_edu"),
    ("ethnicity",             "Ethnicity",             "cat_eth"),
    ("cardiovascular",        "CardiovascularDisease", "yesno"),
    # Parkinson's
    ("jitter",                "MDVP_Jitter",           "num"),
    ("shimmer",               "MDVP_Shimmer",          "num"),
    ("nhr",                   "NHR",                   "num"),
    ("hnr",                   "HNR",                   "num"),
    ("rpde",                  "RPDE",                  "num"),
    ("dfa",                   "DFA",                   "num"),
    ("spread1",               "Spread1",               "num"),
    ("spread2",               "Spread2",               "num"),
    ("ppe",                   "PPE",                   "num"),
    # Tumor
    ("tumor size",            "TumorSize",             "num"),
    ("tumour size",           "TumorSize",             "num"),
    ("lesion size",           "TumorSize",             "num"),
    ("tumor location",        "Location",              "cat_loc"),
    ("tumour location",       "Location",              "cat_loc"),
    ("location",              "Location",              "cat_loc"),
]

_SMOKE_MAP = {
    "never smoked":    [r"never", r"non.?smoker", r"no.*smok"],
    "formerly smoked": [r"former", r"ex.?smok", r"quit", r"past smok"],
    "smokes":          [r"smokes?", r"current.*smok", r"active smok"],
    "Unknown":         [r"unknown", r"not stated", r"nil", r"n/a"],
}
_MARRIED_MAP = {
    "Yes": [r"married", r"\byes\b", r"spouse"],
    "No":  [r"single", r"unmarried", r"divorced", r"widow", r"\bno\b"],
}
_WORK_MAP = {
    "Private":       [r"private", r"salaried"],
    "Govt_job":      [r"govt", r"government", r"public sector"],
    "Self-employed": [r"self.?employed", r"business"],
    "children":      [r"student", r"child"],
    "Never_worked":  [r"unemployed", r"not working", r"homemaker", r"retired"],
}
_RES_MAP = {
    "Urban": [r"urban", r"\bcity\b", r"town", r"metro"],
    "Rural": [r"rural", r"village"],
}
_LOC_MAP = {
    "1": [r"frontal"], "2": [r"temporal"],
    "3": [r"parietal", r"occipital", r"cerebel", r"brainstem", r"other"],
    "0": [r"unknown", r"not known", r"nil"],
}
_EDU_MAP = {
    "0": [r"none", r"illiterate", r"no formal"],
    "1": [r"high school", r"secondary", r"matric", r"10th", r"12th", r"hsc", r"ssc"],
    "2": [r"bachelor", r"undergrad", r"degree", r"b\.sc", r"b\.a"],
    "3": [r"master", r"postgrad", r"phd", r"doctorate", r"higher"],
}
_ETH_MAP = {
    "0": [r"caucasian", r"white", r"european"],
    "1": [r"african", r"black"],
    "2": [r"asian", r"indian", r"chinese"],
    "3": [r"hispanic", r"latino", r"mixed", r"other"],
}

def _cat_map_match(seg: str, cmap: dict) -> str | None:
    for val, pats in cmap.items():
        for p in pats:
            if re.search(p, seg, re.I):
                return val
    return None

def _fallback_kv_extract(text: str, already_found: set) -> dict:
    """
    Split the PDF text into lines; for each line that looks like
    'label : value' or 'label value', try to map the label stem to a
    known field and extract the value. Only fills keys not already found.
    """
    result = {}
    # Split into candidate pairs — each line is a potential key:value
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) > 200:
            continue

        # Try colon or dash separator first
        if ":" in line or "—" in line or "-" in line:
            sep = ":" if ":" in line else ("—" if "—" in line else "-")
            parts = line.split(sep, 1)
            if len(parts) == 2:
                raw_key = parts[0].strip().lower()
                raw_val = parts[1].strip()
            else:
                continue
        else:
            # No separator — treat whole line as possible label (next line = value)
            continue

        if not raw_key or not raw_val:
            continue

        # Match raw_key against KV_FALLBACK stems
        for stem, field_key, ftype in _KV_FALLBACK:
            if field_key in already_found or field_key in result:
                continue
            if stem in raw_key:
                val = _apply_fallback_type(ftype, raw_val)
                if val is not None and val != "":
                    result[field_key] = val
                break

    return result

def _apply_fallback_type(ftype: str, raw_val: str) -> str | None:
    """Convert a raw string value based on the field type."""
    rv = raw_val.strip()
    if ftype == "num":
        m = _NUM_RE.search(re.sub(r"(?:mg/dl|mmhg|cm|mm|%|units?)", " ", rv, flags=re.I))
        if m:
            try:
                float(m.group(0).replace(",", "."))
                return m.group(0).replace(",", ".")
            except ValueError:
                pass
        return None
    if ftype == "yesno":
        if _YES_RE.search(rv):
            return "1"
        if _NO_RE.search(rv):
            return "0"
        bm = _BIN_RE.search(rv)
        return bm.group(1) if bm else None
    if ftype == "gender":
        if re.search(r"female|woman|\bf\b", rv, re.I):
            return "Female"
        if re.search(r"\bmale\b|man|\bm\b", rv, re.I):
            return "Male"
        return None
    if ftype == "text":
        return rv[:80] if rv else None
    if ftype == "cat_smoke":
        return _cat_map_match(rv, _SMOKE_MAP)
    if ftype == "cat_married":
        return _cat_map_match(rv, _MARRIED_MAP)
    if ftype == "cat_work":
        return _cat_map_match(rv, _WORK_MAP)
    if ftype == "cat_res":
        return _cat_map_match(rv, _RES_MAP)
    if ftype == "cat_loc":
        return _cat_map_match(rv, _LOC_MAP)
    if ftype == "cat_edu":
        return _cat_map_match(rv, _EDU_MAP)
    if ftype == "cat_eth":
        return _cat_map_match(rv, _ETH_MAP)
    return None
def extract_pdf(file_bytes: bytes) -> ExtractionResult:
    """
    Parse a text-based PDF and extract all patient fields.

    Strategy:
      1. Extract raw page text + tables via pdfplumber (two tolerance passes).
      2. Normalise whitespace while preserving newlines.
      3. Pass 1 — run every SPEC regex against the full normalised text.
      4. Pass 2 — KV fallback: split into colon-separated lines and match
         label stems to fields that regex didn't catch.
      5. Fill remaining gaps from DEFAULTS.
      6. Sync cross-disease mirror columns.
    """
    text_parts: list[str] = []
    page_count = 0

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            # Try tight tolerance first, then looser for complex layouts
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            if not page_text.strip():
                page_text = page.extract_text(x_tolerance=8, y_tolerance=8) or ""
            if page_text:
                text_parts.append(page_text)
            table_text = _build_table_text(page)
            if table_text:
                text_parts.append(table_text)

    raw_text = "\n".join(text_parts)
    # Normalise: collapse horizontal whitespace but preserve line breaks
    text = "\n".join(_norm_text(ln) for ln in raw_text.split("\n") if ln.strip())

    found: list[dict] = []
    extracted: dict   = {}

    # ── Pass 1: regex SPEC matching ──────────────────────────────────────────
    for spec in SPECS:
        val, matched_label = _extract_field(spec, text)
        key = spec["key"]
        if val is not None and val != "":
            extracted[key] = val
            found.append({
                "key":    key,
                "label":  DISPLAY_LABELS.get(key, key),
                "value":  val,
                "source": "regex",
            })

    # ── Pass 2: KV fallback for anything still missing ───────────────────────
    fallback = _fallback_kv_extract(text, set(extracted.keys()))
    for key, val in fallback.items():
        if key not in extracted and val is not None and val != "":
            extracted[key] = val
            found.append({
                "key":    key,
                "label":  DISPLAY_LABELS.get(key, key),
                "value":  val,
                "source": "fallback",
            })

    # ── Assemble patient_data ─────────────────────────────────────────────────
    patient_data = dict(DEFAULTS)
    for k, v in extracted.items():
        if k in _SYNCED:
            continue
        patient_data[k] = v
    patient_data = _sync_fields(patient_data)

    found_keys = {f["key"] for f in found}
    missing = [
        DISPLAY_LABELS.get(spec["key"], spec["key"])
        for spec in SPECS
        if spec["key"] not in found_keys
    ]

    preview = re.sub(r"\s+", " ", text)[:500]
    return ExtractionResult(
        patient_data  = patient_data,
        found         = found,
        missing       = missing,
        page_count    = page_count,
        text_length   = len(text),
        text_preview  = preview,
    )
