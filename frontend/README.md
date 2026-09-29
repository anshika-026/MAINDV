# Deco Vision — Frontend

React + Vite + Tailwind single-page app for the Deco Vision admin and client portals.

## Development

```bash
cp .env.example .env      # VITE_API_BASE_URL=http://127.0.0.1:8821/api
npm ci
npm run dev               # http://localhost:5180 (fixed port, see vite.config.js)
```

Log in with an admin account created on the backend
(`python -m app.manage create-admin`), or a client-portal license login.
There is no self-signup and no mock login.

## Production build

```bash
npm ci && npm run build   # uses .env.production: API at same-origin /api
```

nginx serves `dist/` and proxies `/api` and `/ws` to the backend. See
`deploy/nginx-deco-vision.conf` and `DEPLOYMENT.md`.

## Structure

- `src/api/client.js` — every backend call. Errors propagate to the page; there
  is no silent mock-data fallback. `VITE_DEMO_MODE=true` shows sample rows only
  on pages that have no backend yet.
- `src/context/AuthContext.jsx` — admin and client sessions (server-validated tokens).
- `src/hooks/useLiveCameraFeed.js` — live video + overlay websockets, with reconnect/backoff.
- `src/lib/visibleInterval.js` — polling that pauses in hidden tabs.
- Pages are lazy-loaded per route (`src/App.jsx`) behind a per-route error boundary.

## Checks

```bash
npm run lint
npm run build
```
