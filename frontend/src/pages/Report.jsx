import { useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'

/* ── Colour maps ─────────────────────────────────────────────────────────── */
const RISK_COLOR  = { 'High Risk': '#dc2626', 'Medium Risk': '#d97706', 'Low Risk': '#16a34a' }
const RISK_BG     = { 'High Risk': 'bg-red-50 border-red-300',   'Medium Risk': 'bg-amber-50 border-amber-300', 'Low Risk': 'bg-green-50 border-green-300' }
const RISK_BADGE  = { 'High Risk': 'bg-red-100 text-red-700',    'Medium Risk': 'bg-amber-100 text-amber-700',  'Low Risk': 'bg-green-100 text-green-700' }
const RISK_TRACK  = { 'High Risk': '#fecaca',                     'Medium Risk': '#fde68a',                      'Low Risk': '#bbf7d0' }
const RISK_LABEL  = { 'High Risk': 'HIGH',                        'Medium Risk': 'MEDIUM',                       'Low Risk': 'LOW' }

/* ── SVG semi-circle gauge — percentage is always outside the arc ─────────
   Arc lives in the top half of a 200×120 viewBox.
   The number sits in the clear lower half — never overlaps the bar.        */
function Gauge({ score, category }) {
  const color = RISK_COLOR[category] || '#6b7280'
  const track = RISK_TRACK[category] || '#e5e7eb'

  // semi-circle: centre (100,110), radius 80, 180° sweep
  const cx = 100, cy = 110, r = 80
  const circumference = Math.PI * r          // half circle ≈ 251.3
  const filled = (score / 100) * circumference

  // path: arc from left (20,110) to right (180,110)
  const arcPath = `M ${cx - r},${cy} A ${r},${r} 0 0,1 ${cx + r},${cy}`

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 200 120" className="w-36 h-24 overflow-visible">
        {/* track (background arc) */}
        <path d={arcPath} fill="none" stroke={track} strokeWidth="14" strokeLinecap="round" />
        {/* filled arc */}
        <path d={arcPath} fill="none" stroke={color} strokeWidth="14" strokeLinecap="round"
          strokeDasharray={`${filled} ${circumference}`} />
        {/* tick marks at 0%, 50%, 100% */}
        <text x="16" y="116" fontSize="9" fill="#9ca3af" textAnchor="middle">0</text>
        <text x="100" y="25"  fontSize="9" fill="#9ca3af" textAnchor="middle">50</text>
        <text x="184" y="116" fontSize="9" fill="#9ca3af" textAnchor="middle">100</text>
      </svg>

      {/* Percentage — large, bold, clearly below the arc */}
      <div className="mt-0 text-center">
        <span className="text-3xl font-extrabold leading-none" style={{ color }}>
          {score}%
        </span>
        <p className="text-xs font-bold tracking-widest mt-0.5"
          style={{ color }}>
          {RISK_LABEL[category]}
        </p>
      </div>
    </div>
  )
}

/* ── Progress bar risk score card ───────────────────────────────────────── */
function RiskCard({ r }) {
  const color = RISK_COLOR[r.risk_category] || '#6b7280'
  const track = RISK_TRACK[r.risk_category] || '#e5e7eb'
  return (
    <div className={`rounded-xl border-2 p-5 text-center ${RISK_BG[r.risk_category]}`}>
      <p className="text-sm font-bold text-gray-700 mb-3">{r.disease_label}</p>

      {/* SVG Gauge */}
      <Gauge score={r.risk_score} category={r.risk_category} />

      {/* Progress bar below gauge for extra clarity */}
      <div className="mt-3">
        <div className="h-3 rounded-full overflow-hidden" style={{ background: track }}>
          <div className="h-full rounded-full transition-all duration-700"
            style={{ width: `${r.risk_score}%`, background: color }} />
        </div>
        <div className="flex justify-between text-xs text-gray-400 mt-0.5">
          <span>0%</span><span>100%</span>
        </div>
      </div>

      <span className={`inline-block mt-3 text-xs font-semibold px-3 py-1 rounded-full ${RISK_BADGE[r.risk_category]}`}>
        {r.risk_category}
      </span>
    </div>
  )
}

/* ── Evidence card — why the disease risk is triggered ───────────────────── */
const EV_ICON = {
  high:  { icon: '↑', bg: 'bg-red-50',    border: 'border-red-200',    text: 'text-red-700',    label: 'Above safe limit' },
  low:   { icon: '↓', bg: 'bg-orange-50', border: 'border-orange-200', text: 'text-orange-700', label: 'Below normal range' },
  flag1: { icon: '⚠', bg: 'bg-yellow-50', border: 'border-yellow-200', text: 'text-yellow-700', label: 'Risk factor present' },
}

function EvidenceSection({ results }) {
  const highRisk = results.filter(r => r.evidence && r.evidence.length > 0 && r.risk_category === 'High Risk')
  const withEvidence = highRisk.length > 0 ? [highRisk.reduce((a, b) => a.risk_score > b.risk_score ? a : b)] : []
  if (withEvidence.length === 0) return null

  return (
    <div className="card">
      <h2 className="text-base font-bold text-gray-800 mb-1">
        Why Is This Disease Risk Detected?
      </h2>
      <p className="text-xs text-gray-500 mb-5">
        Showing the highest-risk disease only. The following patient values exceeded or fell
        below clinical reference thresholds, directly contributing to the model's risk assessment.
      </p>

      <div className="space-y-6">
        {withEvidence.map(r => (
          <div key={r.disease}>
            {/* Disease header */}
            <div className={`flex items-center gap-3 px-4 py-2 rounded-lg border-l-4 mb-3
              ${r.risk_category === 'High Risk'   ? 'bg-red-50 border-red-500' :
                r.risk_category === 'Medium Risk' ? 'bg-amber-50 border-amber-500' :
                                                    'bg-green-50 border-green-500'}`}>
              <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${RISK_BADGE[r.risk_category]}`}>
                {r.risk_score}% — {r.risk_category}
              </span>
              <span className="text-sm font-bold text-gray-800">{r.disease_label}</span>
            </div>

            {/* Evidence grid */}
            <div className="grid sm:grid-cols-2 gap-2">
              {r.evidence.map((ev, i) => {
                const style = EV_ICON[ev.direction] || EV_ICON.flag1
                return (
                  <div key={i}
                    className={`flex items-start gap-3 rounded-lg border px-3 py-2.5
                      ${style.bg} ${style.border}`}>
                    {/* Icon */}
                    <div className={`flex-shrink-0 w-7 h-7 rounded-full flex items-center
                      justify-center font-bold text-sm ${style.bg} ${style.text} border ${style.border}`}>
                      {style.icon}
                    </div>
                    {/* Content */}
                    <div className="min-w-0">
                      <p className={`text-xs font-bold ${style.text} mb-0.5`}>{ev.label}</p>
                      <p className="text-xs text-gray-700 leading-snug">{ev.explanation}</p>
                      <span className={`inline-block mt-1 text-xs px-1.5 py-0.5 rounded
                        font-medium ${style.bg} ${style.text} border ${style.border}`}>
                        {style.label}
                      </span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── Main Report Page ────────────────────────────────────────────────────── */
export default function Report() {
  const navigate = useNavigate()

  useEffect(() => {
    const stored = sessionStorage.getItem('report')
    if (!stored) navigate('/predict')
  }, [navigate])

  const stored = sessionStorage.getItem('report')
  if (!stored) return null
  const report = JSON.parse(stored)

  const { results, primary_risk_disease, summary_message, disclaimer,
          _patient_name, _patient_id } = report

  const sorted = [...results].sort((a, b) => b.risk_score - a.risk_score)
  const topRisk = sorted[0]

  return (
    <div className="space-y-5">

      {/* ── Report header ── */}
      <div className="card">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
          <div>
            <h1 className="text-lg font-bold text-gray-900">Patient Prediction Report</h1>
            <p className="text-sm text-gray-500 mt-0.5">
              Generated:&nbsp;
              {new Date().toLocaleDateString('en-IN', { day: '2-digit', month: 'long', year: 'numeric' })}
              &nbsp;&nbsp;|&nbsp;&nbsp;
              {new Date().toLocaleTimeString()}
            </p>
            {(_patient_name || _patient_id) && (
              <p className="text-sm text-gray-700 mt-1">
                {_patient_name && <><strong>Patient:</strong> {_patient_name}&nbsp;&nbsp;</>}
                {_patient_id   && <><strong>ID:</strong> {_patient_id}</>}
              </p>
            )}
          </div>
          <div className="flex gap-2 flex-wrap">
            <Link to="/predict" className="btn-white text-sm">New Patient</Link>
            <button onClick={() => window.print()} className="btn-blue text-sm">Print / Save PDF</button>
          </div>
        </div>
      </div>

      {/* ── Disclaimer ── */}
      <div className="rounded border border-yellow-300 bg-yellow-50 px-4 py-2 text-xs text-yellow-800">
        <strong>Disclaimer:</strong> {disclaimer}
      </div>

      {/* ── Overall summary banner ── */}
      <div className={`card border-l-4 ${RISK_BG[topRisk?.risk_category]}`}>
        <h2 className="font-semibold text-gray-800 mb-1">Overall Assessment</h2>
        <p className="text-sm text-gray-700">{summary_message}</p>
        <p className="text-sm font-semibold text-gray-800 mt-2">
          Primary risk:&nbsp;
          <span style={{ color: RISK_COLOR[topRisk?.risk_category] }}>
            {primary_risk_disease}
          </span>
        </p>
      </div>

      {/* ── Risk gauge cards — percentage is always clearly visible ── */}
      <div className="card">
        <h2 className="text-base font-bold text-gray-800 mb-4">Disease Risk Scores</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {sorted.map(r => <RiskCard key={r.disease} r={r} />)}
        </div>
      </div>

      {/* ── Evidence — why each disease is flagged ── */}
      <EvidenceSection results={sorted} />

      {/* ── Summary table ── */}
      <div className="card">
        <h2 className="text-sm font-bold text-gray-800 mb-3">Risk Score Summary</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm border border-gray-200 rounded">
            <thead className="bg-gray-50 text-xs font-semibold text-gray-600 uppercase">
              <tr>
                <th className="px-3 py-2 text-left">Disease</th>
                <th className="px-3 py-2 text-left">Risk Score</th>
                <th className="px-3 py-2 text-left">Risk Level</th>
                <th className="px-3 py-2 text-left">LR Model</th>
                <th className="px-3 py-2 text-left">ANN Model</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {sorted.map(r => {
                const lr  = r.models?.find(m => m.model.includes('Logistic'))
                const ann = r.models?.find(m => m.model.includes('ANN'))
                return (
                  <tr key={r.disease} className="hover:bg-gray-50">
                    <td className="px-3 py-2 font-semibold text-gray-800">{r.disease_label}</td>
                    <td className="px-3 py-2">
                      <span className="text-2xl font-extrabold"
                        style={{ color: RISK_COLOR[r.risk_category] }}>
                        {r.risk_score}%
                      </span>
                    </td>
                    <td className="px-3 py-2">
                      <span className={`text-xs font-bold px-2 py-1 rounded-full ${RISK_BADGE[r.risk_category]}`}>
                        {r.risk_category}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-gray-600 font-medium">
                      {lr  ? `${(lr.probability  * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="px-3 py-2 text-gray-600 font-medium">
                      {ann ? `${(ann.probability * 100).toFixed(1)}%` : '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      

    </div>
  )
}
