import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// WebKit only applies :active styles on touch when the document has a
// touchstart listener. Passive and empty; it costs nothing and is what
// makes every button's press feedback work on iOS.
document.addEventListener('touchstart', () => {}, { passive: true })

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// Cache the WASM engine and the ~3.6 MB champion weights so a cold visit
// on cellular does not re-download them. Production only: a service
// worker in front of the Vite dev server just serves stale modules.
if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    const base = import.meta.env.BASE_URL
    navigator.serviceWorker
      .register(`${base}sw.js`, { scope: base })
      .catch((err) => console.warn('service worker registration failed', err))
  })
}
