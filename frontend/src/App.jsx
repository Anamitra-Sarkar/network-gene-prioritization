import React, { useEffect, useState } from 'react'
import { auth } from './lib/firebase.js'
import { apiUrl } from './lib/api.js'
import { onAuthStateChanged, signInWithEmailAndPassword, signOut } from 'firebase/auth'
import PrioritizePage from './components/PrioritizePage.jsx'

export default function App() {
  const [user, setUser] = useState(null)
  const [health, setHealth] = useState(null)

  useEffect(() => {
    fetch(apiUrl('/api/v1/healthz')).then(r => r.json()).then(setHealth).catch(() => setHealth({ status: 'unreachable' }))
    if (!auth) return
    const unsub = onAuthStateChanged(auth, setUser)
    return () => unsub()
  }, [])

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 840, margin: '2rem auto', padding: '0 1rem', boxSizing: 'border-box' }}>
      <header>
        <h1 style={{ fontSize: 'clamp(1.4rem, 4vw, 2rem)', lineHeight: 1.2, marginBottom: 4 }}>Network Gene Prioritization</h1>
        <p style={{ color: '#374151', marginTop: 0 }}>Network-propagated multi-omics disease-gene prioritization (RWR + learned fusion)</p>
      </header>

      <div style={{ marginBottom: '1.25rem', borderRadius: 12, overflow: 'hidden', border: '1px solid #e5e7eb', background: '#f9fafb' }}>
        <img
          src="/hero.png"
          alt="Abstract illustration of a gene interaction network with interconnected nodes and edges representing biological pathways and multi-omics data flow"
          style={{ display: 'block', width: '100%', maxHeight: 320, objectFit: 'cover', objectPosition: 'center' }}
          loading="eager"
        />
      </div>

      <div style={{ background: '#f5f5f5', padding: '0.75rem 1rem', borderRadius: 8, marginBottom: '1rem', overflowX: 'auto' }} role="status" aria-live="polite">
        <strong>Service status:</strong>{' '}
        {health ? (
          <code style={{ wordBreak: 'break-all', fontSize: '0.85em' }}>{JSON.stringify(health)}</code>
        ) : 'loading...'}
        {health && !health.model_approved && (
          <div style={{ color: '#92400e', background: '#fef3c7', border: '1px solid #fde68a', padding: '6px 10px', borderRadius: 6, marginTop: 8 }}>
            Research service — no approved model yet. Results will show abstention state until <code>MODEL_RELEASE_APPROVED=true</code>.
          </div>
        )}
        {health && health.status === 'unreachable' && (
          <div style={{ color: '#991b1b', background: '#fef2f2', border: '1px solid #fecaca', padding: '6px 10px', borderRadius: 6, marginTop: 8 }} role="alert">
            API unreachable — check that the backend is running at the configured API base URL.
          </div>
        )}
      </div>

      {!auth ? (
        <div style={{ border: '1px solid #e5e7eb', padding: '1rem', borderRadius: 8, marginBottom: '1rem' }}>
          <p><em>Firebase not configured (set VITE_FIREBASE_* env vars to enable auth).</em></p>
          <p>Showing prioritization page in unauthenticated preview mode — real deployment requires sign-in.</p>
          <PrioritizePage user={null} />
        </div>
      ) : !user ? (
        <LoginForm />
      ) : (
        <div>
          <p>Signed in as {user.email} <button onClick={() => signOut(auth)}>Sign out</button></p>
          <PrioritizePage user={user} />
        </div>
      )}
    </div>
  )
}

function LoginForm() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [err, setErr] = useState(null)
  const onSubmit = async (e) => {
    e.preventDefault()
    setErr(null)
    try { await signInWithEmailAndPassword(auth, email, password) } catch (e) { setErr(e.message) }
  }
  return (
    <form onSubmit={onSubmit} noValidate aria-label="Sign in" style={{ border: '1px solid #e5e7eb', padding: '1rem', borderRadius: 8, maxWidth: '100%', boxSizing: 'border-box' }}>
      <h3>Researcher sign-in</h3>
      <div style={{ marginBottom: 8 }}>
        <label htmlFor="login-email" style={{ display: 'block', fontWeight: 600, marginBottom: 4 }}>Email</label>
        <input id="login-email" name="email" type="email" autoComplete="email" required aria-required="true"
          placeholder="researcher@example.com" value={email} onChange={e => setEmail(e.target.value)}
          style={{ display: 'block', marginBottom: 4, padding: '8px 10px', width: '100%', borderRadius: 6, border: '1px solid #9ca3af', fontSize: '1rem', boxSizing: 'border-box' }} />
      </div>
      <div style={{ marginBottom: 12 }}>
        <label htmlFor="login-password" style={{ display: 'block', fontWeight: 600, marginBottom: 4 }}>Password</label>
        <input id="login-password" name="password" type="password" autoComplete="current-password" required aria-required="true"
          placeholder="password" value={password} onChange={e => setPassword(e.target.value)}
          style={{ display: 'block', marginBottom: 4, padding: '8px 10px', width: '100%', borderRadius: 6, border: '1px solid #9ca3af', fontSize: '1rem', boxSizing: 'border-box' }} />
      </div>
      <button type="submit" style={{ padding: '8px 16px', borderRadius: 6, border: 'none', background: '#111827', color: '#fff', fontWeight: 600, cursor: 'pointer' }}>Sign in</button>
      {err && <p style={{ color: '#991b1b', background: '#fef2f2', border: '1px solid #fecaca', padding: 8, borderRadius: 6, marginTop: 10 }} role="alert">{err}</p>}
    </form>
  )
}
