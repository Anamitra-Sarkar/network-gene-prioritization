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
    setLoading(true); setError(null); setResult(null)
    try {
      const token = user && auth?.currentUser ? await auth.currentUser.getIdToken() : null
      const body = {
        disease_name: disease || null,
        seed_genes: seedGenes ? seedGenes.split(',').map(s => s.trim()).filter(Boolean) : null,
        hpo_terms: hpoTerms ? hpoTerms.split(',').map(s => s.trim()).filter(Boolean) : null,
        top_k: 50,
      }
      const headers = { 'Content-Type': 'application/json' }
      if (token) headers['Authorization'] = `Bearer ${token}`
      // Also try service token if available (internal endpoint)
      const res = await fetch(apiUrl('/api/v1/prioritize'), { method: 'POST', headers, body: JSON.stringify(body) })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || JSON.stringify(data))
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally { setLoading(false) }
  }

  return (
    <div style={{ border: '1px solid #e5e7eb', padding: '1rem', borderRadius: 8 }}>
      <h2>Prioritize genes</h2>
      <p style={{ color: '#666', fontSize: '0.9em' }}>Enter a disease name or seed gene list (HGNC symbols). Requires approved model — otherwise shows abstention.</p>
      <form onSubmit={onSubmit}>
        <label>Disease name <input value={disease} onChange={e => setDisease(e.target.value)} placeholder="e.g. Marfan syndrome" style={{ width: '100%', padding: 6, marginBottom: 8 }} /></label>
        <label>Seed genes (comma-separated HGNC symbols) <input value={seedGenes} onChange={e => setSeedGenes(e.target.value)} placeholder="e.g. FBN1,TGFBR1,TGFBR2" style={{ width: '100%', padding: 6, marginBottom: 8 }} /></label>
        <label>HPO terms (comma-separated, optional) <input value={hpoTerms} onChange={e => setHpoTerms(e.target.value)} placeholder="e.g. HP:0001377, HP:0001166" style={{ width: '100%', padding: 6, marginBottom: 8 }} /></label>
        <button type="submit" disabled={loading}>{loading ? 'Running...' : 'Prioritize'}</button>
      </form>

      {error && <div style={{ color: 'red', marginTop: 12 }}>Error: {error}</div>}

      {result && (
        <div style={{ marginTop: 12, padding: 12, background: result.status === 'abstained' ? '#fef3c7' : '#ecfdf5', borderRadius: 8 }}>
          <strong>Status:</strong> {result.status}
          {result.message && <p>{result.message}</p>}
          {result.results && (
            <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 8 }}>
              <thead><tr><th>Rank</th><th>Gene</th><th>Score</th></tr></thead>
              <tbody>{result.results.map(r => <tr key={r.gene_symbol}><td>{r.rank}</td><td>{r.gene_symbol}</td><td>{r.score.toFixed(4)}</td></tr>)}</tbody>
            </table>
          )}
          {!result.results && result.status === 'abstained' && (
            <p style={{ fontStyle: 'italic', color: '#92400e' }}>No approved model artifact — this is the honest abstention state. Real inference will be enabled after training on Kaggle/Modal and release approval.</p>
          )}
        </div>
      )}
    </div>
  )
}
