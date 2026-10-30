import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { predictAll, predictFromPdf } from '../services/api'

/* ── Default values for all form fields ─────────────────────────────────── */
const DEFAULTS = {
  // Personal
  patient_name: '', patient_id: '', patient_age: '', patient_gender: 'Male',
  // Cardiovascular / stroke
  hypertension: '0', heart_disease: '0', avg_glucose_level: '', bmi: '',
  smoking_status: 'never smoked', ever_married: 'Yes',
  work_type: 'Private', Residence_type: 'Urban',
  // ML field mirrors of the personal fields
  age: '', gender: 'Male',
  // Neurological symptoms
  Headache: '0', Seizure: '0', MemoryLoss: '0', SpeechDifficulty: '0',
  VisionProblems: '0', BalanceIssues: '0', Fatigue: '0', Nausea: '0',
  // Alzheimer's (real data schema)
  Age: '', Gender: '0', Ethnicity: '0', EducationLevel: '2',
  BMI: '25', Smoking: '0', AlcoholConsumption: '5',
  PhysicalActivity: '4', DietQuality: '5', SleepQuality: '7',
  FamilyHistoryAlzheimers: '0', CardiovascularDisease: '0', Diabetes: '0',
  Depression: '0', HeadInjury: '0', Hypertension: '0',
  SystolicBP: '120', DiastolicBP: '80',
  CholesterolTotal: '200', CholesterolLDL: '100', CholesterolHDL: '50',
  CholesterolTriglycerides: '150',
  MMSE: '27', FunctionalAssessment: '7', MemoryComplaints: '0',
  BehavioralProblems: '0', ADL: '8',
  Confusion: '0', Disorientation: '0', PersonalityChanges: '0',
  DifficultyCompletingTasks: '0', Forgetfulness: '0',
  // Voice (Parkinson)
  MDVP_Fo: '154', MDVP_Fhi: '197', MDVP_Flo: '116',
  MDVP_Jitter: '0.004', MDVP_Shimmer: '0.025', NHR: '0.015', 'HNR': '22',
  RPDE: '0.47', DFA: '0.72', Spread1: '-5.5', Spread2: '0.22', D2: '2.4',
  PPE: '0.15',
  MDVP_Jitter_Abs: '0.00002', MDVP_RAP: '0.002', MDVP_PPQ: '0.002',
  Jitter_DDP: '0.006', MDVP_Shimmer_dB: '0.25',
  Shimmer_APQ3: '0.012', Shimmer_APQ5: '0.015',
  MDVP_APQ: '0.015', Shimmer_DDA: '0.035',
  // Tumor
  TumorSize: '0', Location: '0',
}

const RISK_BADGE = {
  'High Risk':   'bg-red-100 text-red-700',
  'Medium Risk': 'bg-amber-100 text-amber-700',
  'Low Risk':    'bg-green-100 text-green-700',
}
const RISK_BAR = {
  'High Risk':   'bg-red-500',
  'Medium Risk': 'bg-amber-500',
  'Low Risk':    'bg-green-500',
}

/* ── Small reusable field components ─────────────────────────────────────── */
function TextInput({ label, id, value, onChange, placeholder = '' }) {
  return (
    <div>
      <label htmlFor={id} className="field-label">{label}</label>
      <input id={id} type="text" value={value} placeholder={placeholder}
        onChange={e => onChange(id, e.target.value)} className="field-input" />
    </div>
  )
}

function NumberField({ label, id, value, onChange, min, max, step = 'any', required, hint }) {
  return (
    <div>
      <label htmlFor={id} className="field-label">
        {label}{required && <span className="text-red-500 ml-0.5">*</span>}
      </label>
      <input id={id} type="number" value={value} min={min} max={max} step={step}
        onChange={e => onChange(id, e.target.value)}
        className="field-input" placeholder={required ? 'Required' : ''} />
      {hint && <p className="text-xs text-gray-400 mt-0.5">{hint}</p>}
    </div>
  )
}

function SelectField({ label, id, value, onChange, options, hint }) {
  return (
    <div>
      <label htmlFor={id} className="field-label">{label}</label>
      <select id={id} value={value} onChange={e => onChange(id, e.target.value)} className="field-input">
        {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
      {hint && <p className="text-xs text-gray-400 mt-0.5">{hint}</p>}
    </div>
  )
}

function YesNo({ label, id, value, onChange }) {
  return (
    <SelectField label={label} id={id} value={value} onChange={onChange}
      options={[['0', 'No'], ['1', 'Yes']]} />
  )
}

/* ── Main page ────────────────────────────────────────────────────────────── */
export default function Predict() {
  const [form, setForm]           = useState(DEFAULTS)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState('')
  const [pdfFile, setPdfFile]     = useState(null)
  const [pdfLoading, setPdfLoading] = useState(false)
  const [pdfError, setPdfError]   = useState('')
  const [pdfResult, setPdfResult] = useState(null)   // extraction + risk preview
  const [showManual, setShowManual] = useState(false)
  const pdfRef = useRef(null)
  const navigate = useNavigate()

  const set = (id, val) => setForm(f => ({ ...f, [id]: val }))

  /* Sync patient_age → age, patient_gender → gender for ML */
  const setAge    = (id, val) => setForm(f => ({ ...f, patient_age: val, age: val, Age: val }))
  const setGender = (id, val) => setForm(f => ({ ...f, patient_gender: val, gender: val,
    Gender: val.toLowerCase().startsWith('f') ? '1' : '0' }))

  /* Sync stroke field names → Alzheimer's equivalents before submit */
  const syncForm = (f) => ({
    ...f,
    Age: f.patient_age || f.age || '',
    Gender: f.patient_gender?.toLowerCase().startsWith('f') ? '1' : '0',
    BMI: f.bmi || f.BMI || '',
    Hypertension: f.hypertension || f.Hypertension || '0',
    CardiovascularDisease: f.heart_disease || f.CardiovascularDisease || '0',
    Smoking: f.smoking_status && !['never smoked', 'Unknown'].includes(f.smoking_status) ? '1' : '0',
  })

  const goReport = (result, name, id) => {
    const report = { ...result, _patient_name: name, _patient_id: id }
    sessionStorage.setItem('report', JSON.stringify(report))
    navigate('/report')
  }

  /* PDF selection */
  const handlePdf = (e) => {
    const file = e.target.files[0]
    if (!file) return
    setPdfError(''); setPdfResult(null)
    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
      setPdfError('Please select a valid PDF file.')
      return
    }
    setPdfFile(file)
  }

  /* ── PRIMARY: predict directly from the uploaded PDF ── */
  const handlePdfPredict = async () => {
    if (!pdfFile) { setPdfError('Please upload a patient PDF report first.'); return }
    setPdfError(''); setPdfLoading(true); setPdfResult(null)
    try {
      const res = await predictFromPdf(pdfFile)
      // pre-fill the optional manual form with whatever was read from the PDF
      setForm(f => ({ ...DEFAULTS, ...res.patient_data }))
      setPdfResult(res)
    } catch (err) {
      setPdfError(err.message)
    } finally {
      setPdfLoading(false)
    }
  }

  /* ── OPTIONAL: predict from manually entered / verified data ── */
  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    if (!form.age || !form.avg_glucose_level) {
      setError('Age and Average Blood Glucose Level are required for manual entry.')
      return
    }
    setLoading(true)
    try {
      const result = await predictAll(syncForm(form))
      goReport(result, form.patient_name, form.patient_id)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const sorted = pdfResult?.results
    ? [...pdfResult.results].sort((a, b) => b.risk_score - a.risk_score)
    : []

  return (
    <div className="space-y-5">

      {/* Page header */}
      <div className="card">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h1 className="text-lg font-bold text-gray-900">Patient Data Entry</h1>
            <p className="text-sm text-gray-500 mt-0.5">
              Upload a patient medical report PDF — the system reads it and predicts.
              Manual entry is optional, for verification or when you don't have a PDF.
            </p>
          </div>
        </div>
      </div>

      {/* ── PDF Upload (primary) ─────────────────────────────────────────── */}
      <div className="card border-2 border-blue-200">
        <div className="flex items-center gap-2 mb-1">
          <h2 className="section-head !mb-0">Upload Patient PDF Report</h2>
          <span className="text-xs font-semibold text-blue-700 bg-blue-100 px-2 py-0.5 rounded">Required</span>
        </div>
        <p className="text-sm text-gray-500 mb-3">
          Upload the patient's medical report. The system extracts the data directly
          from the PDF and predicts risk for all four brain diseases.
        </p>

        <div
          onClick={() => pdfRef.current?.click()}
          onDragOver={e => e.preventDefault()}
          onDrop={e => { e.preventDefault(); if (e.dataTransfer.files[0]) handlePdf({ target: { files: e.dataTransfer.files } }) }}
          className="border-2 border-dashed border-blue-300 rounded-lg p-8 text-center cursor-pointer
                     hover:border-blue-500 hover:bg-blue-50 transition-colors">
          <input ref={pdfRef} type="file" accept="application/pdf" className="hidden" onChange={handlePdf} />
          <div className="text-4xl mb-2 text-blue-400">📄</div>
          {pdfFile ? (
            <>
              <p className="text-sm font-semibold text-blue-700">{pdfFile.name}</p>
              <p className="text-xs text-gray-400 mt-1">Click to choose a different file</p>
            </>
          ) : (
            <>
              <p className="text-sm font-medium text-gray-700">Click to upload or drop the patient PDF here</p>
              <p className="text-xs text-gray-400 mt-1">Accepted format: PDF only</p>
            </>
          )}
        </div>

        <div className="mt-4 flex items-center justify-between flex-wrap gap-3">
          <button type="button" onClick={handlePdfPredict} disabled={pdfLoading || !pdfFile}
            className="btn-blue px-8 py-2.5">
            {pdfLoading ? 'Reading PDF & predicting…' : 'Predict from PDF'}
          </button>
        </div>

        {pdfError && (
          <div className="mt-3 rounded border border-red-300 bg-red-50 px-4 py-2.5 text-sm text-red-700">
            {pdfError}
          </div>
        )}
      </div>

      {/* ── PDF analysis result preview ──────────────────────────────────── */}
      {pdfResult && (
        <div className="card border-l-4 border-blue-500">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div>
              <h2 className="text-sm font-semibold text-gray-800">PDF Analysis Result</h2>
              <p className="text-xs text-green-700 mt-0.5">
                ✓ Read {pdfResult.extraction?.fields_read} fields from the PDF
                ({pdfResult.extraction?.fields_missing} not found · {pdfResult.extraction?.page_count} page(s))
              </p>
            </div>
            <button type="button" onClick={() => goReport(pdfResult,
                pdfResult.patient_data?.patient_name, pdfResult.patient_data?.patient_id)}
              className="btn-blue text-sm">
              View Full Report
            </button>
          </div>

          <p className="text-sm text-gray-700 mt-3 bg-gray-50 rounded p-2.5">
            {pdfResult.summary_message}
          </p>

          {/* Risk preview bars */}
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 mt-3">
            {sorted.map(r => (
              <div key={r.disease} className="bg-gray-50 rounded p-3">
                <p className="text-xs font-semibold text-gray-600">{r.disease_label}</p>
                <p className="text-lg font-bold text-gray-900 mt-0.5">{r.risk_score}%</p>
                <div className="h-1.5 rounded bg-gray-200 mt-1.5 overflow-hidden">
                  <div className={`h-full ${RISK_BAR[r.risk_category]}`} style={{ width: `${r.risk_score}%` }} />
                </div>
                <span className={`inline-block mt-1.5 text-xs font-semibold px-2 py-0.5 rounded ${RISK_BADGE[r.risk_category]}`}>
                  {r.risk_category}
                </span>
              </div>
            ))}
          </div>

          {/* Extracted fields list (collapsible) */}
          <details className="mt-3">
            <summary className="text-xs font-medium text-blue-700 cursor-pointer">
              Fields read from the PDF ({pdfResult.extraction?.fields_read})
            </summary>
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-x-4 gap-y-1 mt-2 text-xs">
              {pdfResult.extraction?.found?.map(f => (
                <div key={f.key} className="flex justify-between border-b border-gray-100 py-0.5">
                  <span className="text-gray-500">{f.label}</span>
                  <span className="text-gray-800 font-medium">{String(f.value)}</span>
                </div>
              ))}
            </div>
            {pdfResult.extraction?.missing?.length > 0 && (
              <p className="text-xs text-gray-400 mt-2">
                Not found in PDF (defaults used): {pdfResult.extraction.missing.join(', ')}
              </p>
            )}
          </details>

          <p className="text-xs text-gray-400 mt-3">
            Want to correct something? Expand “Enter / verify data manually” below, adjust the
            pre-filled values, and run the prediction again.
          </p>
        </div>
      )}

      {/* ── Manual entry (optional, collapsible) ──────────────────────────── */}
      <div className="card">
        <button type="button" onClick={() => setShowManual(s => !s)}
          className="w-full flex items-center justify-between text-left">
          <div>
            <h2 className="text-sm font-semibold text-gray-800">Enter / verify data manually</h2>
            <p className="text-xs text-gray-400 mt-0.5">
              Optional. Use this if you don't have a PDF, or to correct values read from one.
              {pdfResult ? ' Pre-filled from your PDF — adjust as needed.' : ''}
            </p>
          </div>
          <span className="text-gray-400 text-lg">{showManual ? '▲' : '▼'}</span>
        </button>
      </div>

      {showManual && (
        <form onSubmit={handleSubmit} noValidate className="space-y-5">

          {/* 1. Patient Information */}
          <div className="card">
            <h2 className="section-head">1. Patient Information</h2>
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <TextInput   label="Patient Name"     id="patient_name" value={form.patient_name} onChange={set} placeholder="Full name" />
              <TextInput   label="Patient ID / Reg. No." id="patient_id" value={form.patient_id} onChange={set} placeholder="e.g. P-2024-001" />
              <NumberField label="Age (years)" id="patient_age" value={form.patient_age} onChange={setAge} min={0} max={120} required />
              <SelectField label="Gender" id="patient_gender" value={form.patient_gender} onChange={setGender}
                options={[['Male','Male'],['Female','Female'],['Other','Other']]} />
              <SelectField label="Marital Status" id="ever_married" value={form.ever_married} onChange={set}
                options={[['Yes','Married'],['No','Unmarried']]} />
              <SelectField label="Occupation Type" id="work_type" value={form.work_type} onChange={set}
                options={[['Private','Private Sector'],['Govt_job','Government Job'],['Self-employed','Self-Employed'],['children','Student / Child'],['Never_worked','Never Worked']]} />
              <SelectField label="Area of Residence" id="Residence_type" value={form.Residence_type} onChange={set}
                options={[['Urban','Urban'],['Rural','Rural']]} />
            </div>
          </div>

          {/* 2. Vital Signs & Blood Reports */}
          <div className="card">
            <h2 className="section-head">2. Vital Signs &amp; Blood Reports</h2>
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <NumberField label="Average Blood Glucose (mg/dL)" id="avg_glucose_level" value={form.avg_glucose_level}
                onChange={set} min={55} max={300} required />
              <NumberField label="BMI (Body Mass Index)" id="bmi" value={form.bmi} onChange={set} min={10} max={60}
                hint="Weight(kg) / Height(m)²" />
              <YesNo label="Hypertension (High BP)" id="hypertension" value={form.hypertension} onChange={set} />
              <YesNo label="Heart Disease" id="heart_disease" value={form.heart_disease} onChange={set} />
              <SelectField label="Smoking Status" id="smoking_status" value={form.smoking_status} onChange={set}
                options={[['never smoked','Never Smoked'],['formerly smoked','Formerly Smoked'],['smokes','Currently Smokes'],['Unknown','Unknown / Not Stated']]} />
            </div>
          </div>

          {/* 3. Neurological Symptoms */}
          <div className="card">
            <h2 className="section-head">3. Neurological Symptoms</h2>
            <p className="text-xs text-gray-400 mb-3">Select Yes if the patient currently has or recently reported the symptom.</p>
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <YesNo label="Persistent Headache"   id="Headache"          value={form.Headache}          onChange={set} />
              <YesNo label="Seizures"              id="Seizure"           value={form.Seizure}           onChange={set} />
              <YesNo label="Memory Loss"           id="MemoryLoss"        value={form.MemoryLoss}        onChange={set} />
              <YesNo label="Speech Difficulty"     id="SpeechDifficulty"  value={form.SpeechDifficulty}  onChange={set} />
              <YesNo label="Vision Problems"       id="VisionProblems"    value={form.VisionProblems}    onChange={set} />
              <YesNo label="Balance / Walking Issues" id="BalanceIssues"  value={form.BalanceIssues}     onChange={set} />
              <YesNo label="Chronic Fatigue"       id="Fatigue"           value={form.Fatigue}           onChange={set} />
              <YesNo label="Nausea / Vomiting"     id="Nausea"            value={form.Nausea}            onChange={set} />
            </div>
          </div>

          {/* 4. Cognitive & Clinical Assessment (Alzheimer's) */}
          <div className="card">
            <h2 className="section-head">4. Cognitive &amp; Clinical Assessment <span className="text-xs font-normal text-gray-400">(For Alzheimer's risk — defaults are healthy values)</span></h2>
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <NumberField label="MMSE Score (0–30)" id="MMSE" value={form.MMSE} onChange={set} min={0} max={30}
                hint="Mini-Mental State Exam. Normal = 27–30. Default: 27" />
              <NumberField label="Functional Assessment (0–10)" id="FunctionalAssessment" value={form.FunctionalAssessment}
                onChange={set} min={0} max={10} hint="Lower = more impaired. Default: 7" />
              <NumberField label="ADL Score (0–10)" id="ADL" value={form.ADL}
                onChange={set} min={0} max={10} hint="Activities of Daily Living. Default: 8" />
              <YesNo label="Memory Complaints" id="MemoryComplaints" value={form.MemoryComplaints} onChange={set} />
              <YesNo label="Behavioral Problems" id="BehavioralProblems" value={form.BehavioralProblems} onChange={set} />
              <YesNo label="Confusion" id="Confusion" value={form.Confusion} onChange={set} />
              <YesNo label="Disorientation" id="Disorientation" value={form.Disorientation} onChange={set} />
              <YesNo label="Personality Changes" id="PersonalityChanges" value={form.PersonalityChanges} onChange={set} />
              <YesNo label="Difficulty Completing Tasks" id="DifficultyCompletingTasks" value={form.DifficultyCompletingTasks} onChange={set} />
              <YesNo label="Forgetfulness" id="Forgetfulness" value={form.Forgetfulness} onChange={set} />
              <YesNo label="Family History (Alzheimer's)" id="FamilyHistoryAlzheimers" value={form.FamilyHistoryAlzheimers} onChange={set} />
              <YesNo label="Cardiovascular Disease" id="CardiovascularDisease" value={form.CardiovascularDisease} onChange={set} />
              <YesNo label="Diabetes" id="Diabetes" value={form.Diabetes} onChange={set} />
              <YesNo label="Depression" id="Depression" value={form.Depression} onChange={set} />
              <YesNo label="Head Injury" id="HeadInjury" value={form.HeadInjury} onChange={set} />
              <SelectField label="Education Level" id="EducationLevel" value={form.EducationLevel} onChange={set}
                options={[['0','None'],['1','High School'],['2','Bachelor\'s'],['3','Higher']]} />
              <NumberField label="Systolic BP (mmHg)" id="SystolicBP" value={form.SystolicBP} onChange={set} min={90} max={180} />
              <NumberField label="Diastolic BP (mmHg)" id="DiastolicBP" value={form.DiastolicBP} onChange={set} min={60} max={120} />
              <NumberField label="Total Cholesterol" id="CholesterolTotal" value={form.CholesterolTotal} onChange={set} min={150} max={300} />
              <NumberField label="LDL Cholesterol" id="CholesterolLDL" value={form.CholesterolLDL} onChange={set} min={50} max={200} />
              <NumberField label="HDL Cholesterol" id="CholesterolHDL" value={form.CholesterolHDL} onChange={set} min={20} max={100} />
              <NumberField label="Triglycerides" id="CholesterolTriglycerides" value={form.CholesterolTriglycerides} onChange={set} min={50} max={400} />
              <NumberField label="Alcohol Consumption (units/wk)" id="AlcoholConsumption" value={form.AlcoholConsumption} onChange={set} min={0} max={20} />
              <NumberField label="Physical Activity (h/wk)" id="PhysicalActivity" value={form.PhysicalActivity} onChange={set} min={0} max={10} />
              <NumberField label="Diet Quality (0–10)" id="DietQuality" value={form.DietQuality} onChange={set} min={0} max={10} />
              <NumberField label="Sleep Quality (4–10)" id="SleepQuality" value={form.SleepQuality} onChange={set} min={4} max={10} />
            </div>
          </div>


          {/* 5. Brain Tumor Indicators */}
          <div className="card">
            <h2 className="section-head">5. Brain Tumor Indicators</h2>
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <NumberField label="Tumor Size (cm)" id="TumorSize" value={form.TumorSize}
                onChange={set} min={0} max={8} hint="Enter 0 if no imaging has been done" />
              <SelectField label="Tumor Location (if known)" id="Location" value={form.Location} onChange={set}
                options={[['0','Not Known'],['1','Frontal Lobe'],['2','Temporal Lobe'],['3','Other Region']]} />
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="rounded border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}

          {/* Submit */}
          <div className="card flex items-center justify-between flex-wrap gap-3">
            <p className="text-xs text-gray-400">
              Fields marked <span className="text-red-500 font-semibold">*</span> are required for manual entry.
              All other fields have default values.
            </p>
            <div className="flex gap-2">
              <button type="button" onClick={() => setForm(DEFAULTS)} className="btn-white text-xs">
                Clear
              </button>
              <button type="submit" disabled={loading} className="btn-blue px-8 py-2.5">
                {loading ? 'Analysing patient data, please wait…' : 'Run Prediction'}
              </button>
            </div>
          </div>

        </form>
      )}

    </div>
  )
}
