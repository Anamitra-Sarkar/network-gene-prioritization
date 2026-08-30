import React, { useEffect, useState } from 'react'
import { auth } from './lib/firebase.js'
import { onAuthStateChanged, signInWithEmailAndPassword, signOut } from 'firebase/auth'
import PrioritizePage from './components/PrioritizePage.jsx'

export default function App() {
  const [user, setUser] = useState(null)
  const [health, setHealth] = useState(null)

  useEffect(() => {
    fetch('/api/v1/healthz').then(r => r.json()).then(setHealth).catch(() => setHealth({ status: 'unreachable' }))
    if (!auth) return
    const unsub = onAuthStateChanged(auth, setUser)
    return () => unsub()
  }, [])

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 800, margin: '2rem auto', padding: '0 1rem' }}>
      <h1>Network Gene Prioritization</h1>
      <p style={{ color: '#555' }}>Network-propagated multi-omics disease-gene prioritization (RWR + learned fusion)</p>

      <div style={{ background: '#f5f5f5', padding: '0.75rem', borderRadius: 8, marginBottom: '1rem' }}>
        <strong>Service status:</strong> {health ? JSON.stringify(health) : 'loading...'}
        {health && !health.model_approved && (
          <div style={{ color: '#b45309', marginTop: 6 }}>
            Research service — no approved model yet. Results will show abstention state until MODEL_RELEASE_APPROVED=true.
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
    <form onSubmit={onSubmit} style={{ border: '1px solid #e5e7eb', padding: '1rem', borderRadius: 8 }}>
      <h3>Researcher sign-in</h3>
      <input placeholder="email" value={email} onChange={e => setEmail(e.target.value)} style={{ display: 'block', marginBottom: 8, padding: 6, width: '100%' }} />
      <input type="password" placeholder="password" value={password} onChange={e => setPassword(e.target.value)} style={{ display: 'block', marginBottom: 8, padding: 6, width: '100%' }} />
      <button type="submit">Sign in</button>
      {err && <p style={{ color: 'red' }}>{err}</p>}
    </form>
  )
}
