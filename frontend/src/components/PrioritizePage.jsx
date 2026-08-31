import React, { useState } from 'react'
import { auth } from '../lib/firebase.js'
import { apiUrl } from '../lib/api.js'

export default function PrioritizePage({ user }) {
  const [disease, setDisease] = useState('')
  const [seedGenes, setSeedGenes] = useState('')
  const [hpoTerms, setHpoTerms] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const onSubmit = async (e) => {
    e.preventDefault()
    // Client-side validation: require at least one of disease or seed genes
    const hasDisease = disease.trim().length > 0
    const hasSeeds = seedGenes.split(',').map(s => s.trim()).filter(Boolean).length > 0
    if (!hasDisease && !hasSeeds) {
      setError('Please provide either a disease name or at least one seed gene (HGNC symbol).')
      return
    }
    setLoading(true); setError(null); setResult(null)
    try {
      const token = user && auth?.currentUser ? await auth.currentUser.getIdToken() : null
      const body = {
        disease_name: disease.trim() || null,
        seed_genes: seedGenes ? seedGenes.split(',').map(s => s.trim()).filter(Boolean) : null,
        hpo_terms: hpoTerms ? hpoTerms.split(',').map(s => s.trim()).filter(Boolean) : null,
        top_k: 50,
      }
      const headers = { 'Content-Type': 'application/json' }
      if (token) headers['Authorization'] = `Bearer ${token}`
      const res = await fetch(apiUrl('/api/v1/prioritize'), { method: 'POST', headers, body: JSON.stringify(body) })
      let data
      try { data = await res.json() } catch { throw new Error(`Server returned ${res.status} ${res.statusText}`) }
      if (!res.ok) {
        // Handle FastAPI validation errors (422) which return {detail: ...}
        const detail = data.detail
        const msg = Array.isArray(detail) ? detail.map(d => d.msg || JSON.stringify(d)).join('; ') : (detail || data.message || JSON.stringify(data))
        throw new Error(msg)
      }
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally { setLoading(false) }
  }

  return (
    <div style={{ border: '1px solid #e5e7eb', padding: '1rem', borderRadius: 8, maxWidth: '100%', boxSizing: 'border-box' }}>
      <h2>Prioritize genes</h2>
      <p style={{ color: '#374151', fontSize: '0.9em' }}>Enter a disease name or seed gene list (HGNC symbols). Requires approved model — otherwise shows abstention.</p>
      <form onSubmit={onSubmit} noValidate aria-label="Gene prioritization form">
        <div style={{ marginBottom: 8 }}>
          <label htmlFor="disease-input" style={{ display: 'block', fontWeight: 600, marginBottom: 4 }}>Disease name</label>
          <input id="disease-input" name="disease" type="text" autoComplete="off" aria-describedby="disease-help"
            value={disease} onChange={e => setDisease(e.target.value)} placeholder="e.g. Marfan syndrome"
            style={{ width: '100%', padding: '8px 10px', borderRadius: 6, border: '1px solid #9ca3af', fontSize: '1rem', boxSizing: 'border-box' }} />
          <span id="disease-help" style={{ fontSize: '0.8em', color: '#6b7280' }}>Or provide seed genes below.</span>
        </div>
        <div style={{ marginBottom: 8 }}>
          <label htmlFor="seedgenes-input" style={{ display: 'block', fontWeight: 600, marginBottom: 4 }}>Seed genes (comma-separated HGNC symbols)</label>
          <input id="seedgenes-input" name="seed_genes" type="text" autoComplete="off" aria-describedby="seedgenes-help"
            value={seedGenes} onChange={e => setSeedGenes(e.target.value)} placeholder="e.g. FBN1,TGFBR1,TGFBR2"
            style={{ width: '100%', padding: '8px 10px', borderRadius: 6, border: '1px solid #9ca3af', fontSize: '1rem', boxSizing: 'border-box' }} />
          <span id="seedgenes-help" style={{ fontSize: '0.8em', color: '#6b7280' }}>At least one required if disease name is empty.</span>
        </div>
        <div style={{ marginBottom: 12 }}>
          <label htmlFor="hpo-input" style={{ display: 'block', fontWeight: 600, marginBottom: 4 }}>HPO terms (comma-separated, optional)</label>
          <input id="hpo-input" name="hpo_terms" type="text" autoComplete="off"
            value={hpoTerms} onChange={e => setHpoTerms(e.target.value)} placeholder="e.g. HP:0001377, HP:0001166"
            style={{ width: '100%', padding: '8px 10px', borderRadius: 6, border: '1px solid #9ca3af', fontSize: '1rem', boxSizing: 'border-box' }} />
        </div>
        <button type="submit" disabled={loading} aria-busy={loading}
          style={{ padding: '10px 18px', borderRadius: 6, border: 'none', background: loading ? '#9ca3af' : '#111827', color: '#fff', fontWeight: 600, cursor: loading ? 'not-allowed' : 'pointer', minWidth: 120 }}>
          {loading ? 'Running…' : 'Prioritize'}
        </button>
      </form>

      <div aria-live="polite" aria-atomic="true">
        {loading && <div style={{ marginTop: 12, padding: 10, background: '#eff6ff', borderRadius: 8, color: '#1e40af' }} role="status">Running prioritization…</div>}
        {error && <div style={{ color: '#991b1b', background: '#fef2f2', border: '1px solid #fecaca', padding: 10, borderRadius: 6, marginTop: 12 }} role="alert">Error: {error}</div>}
        {result && (
          <div style={{ marginTop: 12, padding: 12, background: result.status === 'abstained' ? '#fef3c7' : '#ecfdf5', borderRadius: 8, border: result.status === 'abstained' ? '1px solid #fde68a' : '1px solid #a7f3d0' }} role="region" aria-label="Prioritization result">
            <strong>Status:</strong> {result.status}
            {result.message && <p style={{ margin: '8px 0', color: '#374151' }}>{result.message}</p>}
            {result.results && result.results.length > 0 && (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 8, minWidth: 320 }}>
                  <thead><tr style={{ textAlign: 'left', borderBottom: '2px solid #d1d5db' }}><th scope="col" style={{ padding: '6px 8px' }}>Rank</th><th scope="col" style={{ padding: '6px 8px' }}>Gene</th><th scope="col" style={{ padding: '6px 8px' }}>Score</th></tr></thead>
                  <tbody>{result.results.map(r => <tr key={r.gene_symbol} style={{ borderBottom: '1px solid #e5e7eb' }}><td style={{ padding: '6px 8px' }}>{r.rank}</td><td style={{ padding: '6px 8px' }}>{r.gene_symbol}</td><td style={{ padding: '6px 8px' }}>{r.score.toFixed(4)}</td></tr>)}</tbody>
                </table>
              </div>
            )}
            {!result.results && result.status === 'abstained' && (
              <p style={{ fontStyle: 'italic', color: '#92400e', margin: '8px 0 0' }}>No approved model artifact — this is the honest abstention state. Real inference will be enabled after training on Kaggle/Modal and release approval.</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
