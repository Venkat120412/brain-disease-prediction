import { Routes, Route, Navigate, NavLink } from 'react-router-dom'
import { useState } from 'react'
import Predict from './pages/Predict.jsx'
import Report  from './pages/Report.jsx'
import Dashboard from './pages/Dashboard.jsx'

const NAV = [
  { to:'/predict',   label:'New Patient' },
  { to:'/dashboard', label:'Dashboard'   },
]

export default function App() {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <div className="min-h-screen flex flex-col bg-gray-100">

      {/* ── Header ── */}
      <header className="bg-blue-800 text-white shadow-md">
        <div className="max-w-6xl mx-auto px-4 flex items-center justify-between h-14">

          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-white rounded flex items-center justify-center">
              <span className="text-blue-800 text-lg font-bold">B</span>
            </div>
            <div className="leading-tight">
              <p className="text-sm font-semibold">Brain Disease Prediction System</p>
            </div>
          </div>

          {/* Desktop nav */}
          <nav className="hidden md:flex items-center gap-1 text-sm">
            {NAV.map(({ to, label }) => (
              <NavLink key={to} to={to}
                className={({ isActive }) =>
                  `px-4 py-1.5 rounded font-medium transition-colors
                   ${isActive ? 'bg-blue-600 text-white' : 'text-blue-100 hover:bg-blue-700'}`}>
                {label}
              </NavLink>
            ))}
          </nav>

          {/* Mobile */}
          <button className="md:hidden text-white text-xl px-2"
            onClick={() => setMobileOpen(o => !o)} aria-label="Menu">
            &#9776;
          </button>
        </div>

        {/* Mobile dropdown */}
        {mobileOpen && (
          <div className="md:hidden bg-blue-900 px-4 py-2 space-y-1">
            {NAV.map(({ to, label }) => (
              <NavLink key={to} to={to}
                onClick={() => setMobileOpen(false)}
                className={({ isActive }) =>
                  `block px-3 py-2 rounded text-sm font-medium
                   ${isActive ? 'bg-blue-700 text-white' : 'text-blue-100 hover:bg-blue-700'}`}>
                {label}
              </NavLink>
            ))}
          </div>
        )}
      </header>

      {/* ── Page ── */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 py-6">
        <Routes>
          <Route path="/"          element={<Navigate to="/predict" replace />} />
          <Route path="/predict"   element={<Predict />} />
          <Route path="/report"    element={<Report />}  />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="*"          element={<Navigate to="/predict" replace />} />
        </Routes>
      </main>

      {/* ── Footer ── */}
      <footer className="bg-blue-800 text-blue-200 text-xs text-center py-2">
        Brain Disease Prediction System &copy; 2026
      </footer>

    </div>
  )
}
