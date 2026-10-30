export default function About() {
  return (
    <div className="space-y-5">

      <div className="card">
        <h1 className="text-lg font-bold text-gray-900 mb-1">About This Project</h1>
        <p className="text-sm text-gray-500">Project information and technology details</p>
      </div>

      <div className="grid md:grid-cols-2 gap-5">

        <div className="card">
          <h2 className="section-head">Project Information</h2>
          <table className="w-full text-sm">
            <tbody className="divide-y divide-gray-100">
              {[
                ['Project Title', 'Mathematical Model for Brain Disease Prediction using ANN'],
                ['College', 'Easwari Engineering College'],
                ['Department', 'Information Technology'],
                ['Guide', 'Mrs V. Kavitha, M.Tech., Ph.D.'],
                ['Member 1', 'Sabarish R — 310623205132'],
                ['Member 2', 'Venkatesh Kumar D — 310623205173'],
                ['Review', 'Zeroth Review — 22-06-2026'],
              ].map(([k, v]) => (
                <tr key={k}>
                  <td className="py-2 pr-4 font-medium text-gray-500 align-top w-28">{k}</td>
                  <td className="py-2 text-gray-800">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h2 className="section-head">Abstract</h2>
          <p className="text-sm text-gray-600 leading-relaxed">
            Brain diseases such as Dementia, Alzheimer's Disease, Parkinson's Disease, and Brain
            Tumors can seriously affect a person's health and quality of life. Early detection is
            important for effective treatment and better patient outcomes.
          </p>
          <p className="text-sm text-gray-600 leading-relaxed mt-3">
            This project develops a Mathematical Model using Artificial Neural Networks (ANN) to
            predict brain diseases based on patient medical data. The system generates a Disease Risk
            Score (0–100%) and categorizes patients into Low, Medium, and High Risk groups, providing
            a Doctor Decision Support System for better clinical decisions.
          </p>
        </div>

      </div>

      <div className="card">
        <h2 className="section-head">Technology Stack</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
          {[
            ['Backend',     'Python 3.14, FastAPI'],
            ['ML Models',   'Logistic Regression + ANN (MLPClassifier)'],
            ['Feature Sel.','Mutual Information (SelectKBest)'],
            ['Frontend',    'React 18, Vite, Tailwind CSS, Recharts'],
          ].map(([k, v]) => (
            <div key={k} className="bg-gray-50 rounded p-3 border border-gray-200">
              <p className="text-xs text-gray-400 mb-0.5">{k}</p>
              <p className="font-medium text-gray-800">{v}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h2 className="section-head">Novelty of This Project</h2>
        <div className="grid sm:grid-cols-2 gap-4 text-sm">
          <div>
            <p className="font-medium text-red-700 mb-2">Existing Systems</p>
            <ul className="space-y-1 text-gray-600">
              <li className="flex gap-2"><span className="text-red-400">—</span> Focus only on prediction accuracy</li>
              <li className="flex gap-2"><span className="text-red-400">—</span> Provide only disease classification results</li>
            </ul>
          </div>
          <div>
            <p className="font-medium text-green-700 mb-2">Proposed System</p>
            <ul className="space-y-1 text-gray-600">
              <li className="flex gap-2"><span className="text-green-600">+</span> Generates Disease Risk Score (0–100%)</li>
              <li className="flex gap-2"><span className="text-green-600">+</span> Categorizes into Low / Medium / High Risk</li>
              <li className="flex gap-2"><span className="text-green-600">+</span> Identifies important influencing factors</li>
              <li className="flex gap-2"><span className="text-green-600">+</span> Supports early disease detection</li>
              <li className="flex gap-2"><span className="text-green-600">+</span> Doctor Decision Support System</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="rounded border border-red-200 bg-red-50 p-4 text-sm text-red-800">
        <p className="font-semibold mb-1">Medical Disclaimer</p>
        <p>
          This application is an educational research prototype and is not a medical device.
          Its predictions are not a diagnosis and must not be used to make treatment decisions.
          Consult a qualified healthcare professional for any medical advice.
        </p>
      </div>

    </div>
  )
}
