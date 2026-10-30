import axios from 'axios'
const api = axios.create({ baseURL: '/api', timeout: 60000 })
api.interceptors.response.use(r => r, e => {
  const msg = e.response?.data?.detail || e.message || 'Error'
  return Promise.reject(new Error(typeof msg === 'object' ? JSON.stringify(msg) : msg))
})
export const predictAll      = (data)    => api.post('/predict', data).then(r => r.data)
export const predictFromPdf  = (file)   => {
  const fd = new FormData()
  fd.append('file', file)
  return api.post('/predict-pdf', fd,
    { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 120000 })
    .then(r => r.data)
}
export const fetchStatus     = ()        => api.get('/status').then(r => r.data)
export const fetchMetrics    = (disease) => api.get(`/metrics/${disease}`).then(r => r.data)
export const fetchFeatures   = (disease) => api.get(`/features/${disease}`).then(r => r.data)
export const fetchFormFields = ()        => api.get('/form-fields').then(r => r.data)
export const trainAll        = ()        => api.post('/train/all/run').then(r => r.data)
