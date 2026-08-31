/**
 * Central API base URL helper.
 * Reads from Vite env vars (VITE_API_URL preferred, VITE_API_BASE alias for compatibility).
 * Falls back to "" so that relative /api/* paths work with Vite's dev proxy.
 * Strips trailing slash for consistent joining.
 */
export function getApiBase() {
  const raw = (import.meta.env.VITE_API_URL ?? import.meta.env.VITE_API_BASE ?? "").trim()
  if (!raw) return ""
  return raw.replace(/\/+$/, "")
}

/** Join base + path safely */
export function apiUrl(path) {
  const base = getApiBase()
  // path should start with /
  const p = path.startsWith("/") ? path : `/${path}`
  return `${base}${p}`
}
