import { useEffect, useState } from 'react'
import { fetchStatus, fetchMetrics, trainAll } from '../services/api'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

export default function Dashboard() {
  const [status, setStatus]   = useState([])
  const [metrics, setMetrics] = useState({})
  const [training, setTrain]  = useState(false)
  const [msg, setMsg]         = useState('')

  const load = () => {
    fetchStatus().then(items => {
      setStatus(items)
      items.filter(s => s.trained).forEach(s => {
        fetchMetrics(s.disease).then(m => setMetrics(prev => ({ ...prev, [s.disease]: m }))).catch(() => {})
      })
    }).catch(() => {})
  }

  useEffect(load, [])

  const handleTrainAll = async () => {
    setTrain(true); setMsg('')
    try {
      const res = await trainAll()
      const ok = Object.entries(res).filter(([,v]) => v.success).length
      setMsg(`✓ Training complete. ${ok}/4 models trained successfully.`)
      load()
    } catch (e) { setMsg(`✗ ${e.message}`) }
    finally { setTrain(false) }
  }

  const LABELS = { stroke:'Stroke', alzheimers:"Alzheimer's", parkinsons:"Parkinson's", brain_tumor:'Brain Tumor' }
  const ICONS  = { stroke:'🩸', alzheimers:'🧩', parkinsons:'🤝', brain_tumor:'🔬' }

  // Build comparison chart data
  const chartData = ['accuracy','f1_weighted','roc_auc'].map(m => {
    const row = { metric: m.replace('_weighted','').replace('_',' ').toUpperCase() }
    status.forEach(s => {
      const dm = metrics[s.disease]
      if (dm) {
        row[`${LABELS[s.disease]} LR`]  = dm.logistic_regression?.[m] != null ? +(dm.logistic_regression[m]*100).toFixed(1) : null
        row[`${LABELS[s.disease]} ANN`] = dm.ann?.[m] != null ? +(dm.ann[m]*100).toFixed(1) : null
      }
    })
    return row
  })

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-extrabold text-gray-900">Model Dashboard</h1>
        </div>
        <button onClick={handleTrainAll} disabled={training} className="btn-primary">
          {training ? '⏳ Training…' : '🚀 Train All Models'}
        </button>
      </div>

      {msg && <div className={`rounded-xl p-3 text-sm ${msg.startsWith('✓') ? 'bg-green-50 text-green-700 border border-green-200' : 'bg-red-50 text-red-700 border border-red-200'}`}>{msg}</div>}

      {/* Status grid */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {status.map(s => {
          const m = metrics[s.disease]
          return (
            <div key={s.disease} className="card">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xl">{ICONS[s.disease]}</span>
                <p className="font-bold text-sm text-gray-800">{s.disease_label}</p>
              </div>
              <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${s.trained ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'}`}>
                {s.trained ? '✓ Trained' : '⚠ Not trained'}
              </span>
              {s.trained && (
                <div className="mt-3 space-y-1 text-xs text-gray-500">
                  <div className="flex justify-between"><span>Train rows:</span><span className="font-semibold">{s.train_rows}</span></div>
                  <div className="flex justify-between"><span>Test rows:</span><span className="font-semibold">{s.test_rows}</span></div>
                  {m && <>
                    <div className="flex justify-between"><span>LR Accuracy:</span><span className="font-semibold text-blue-600">{(m.logistic_regression?.accuracy*100).toFixed(1)}%</span></div>
                    <div className="flex justify-between"><span>ANN Accuracy:</span><span className="font-semibold text-purple-600">{(m.ann?.accuracy*100).toFixed(1)}%</span></div>
                  </>}
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Cross-disease comparison */}
      {Object.keys(metrics).length > 0 && (
        <div className="card">
          <h2 className="font-bold text-gray-800 mb-4">Cross-Disease Model Performance</h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData} margin={{ top:5, right:20, left:0, bottom:5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="metric" tick={{ fontSize:11 }} />
              <YAxis domain={[0,100]} tick={{ fontSize:11 }} unit="%" />
              <Tooltip formatter={v => `${v}%`} />
              <Legend wrapperStyle={{ fontSize:10 }} />
              {status.filter(s => metrics[s.disease]).flatMap(s => [
                <Bar key={`${s.disease}-lr`}  dataKey={`${LABELS[s.disease]} LR`}  fill="#2563eb" radius={[2,2,0,0]} />,
                <Bar key={`${s.disease}-ann`} dataKey={`${LABELS[s.disease]} ANN`} fill="#7c3aed" radius={[2,2,0,0]} />,
              ])}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      
    </div>
  )
}
