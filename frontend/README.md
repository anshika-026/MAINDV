# Deco Vision — Frontend

React + Vite + Tailwind frontend for the Deco Vision admin panel, built to match
the Figma file ("Deco vision review").

## Getting started

```bash
npm install
npm run dev
```

Open the printed local URL (usually http://localhost:5173).

Login with any email/password — auth currently runs through the mock API layer
described below, so any credentials will get you in.

## Connecting your backend

Every page fetches data through **`src/api/client.js`** instead of talking to
mock data directly. Each function in that file has the real `fetch()` call
already written and commented out just above the mock line, e.g.:

```js
export async function getAlerts() {
  // return request("/alerts");
  return Promise.resolve(mock.alerts);
}
```

To connect your backend:

1. Copy `.env.example` to `.env` and set `VITE_API_BASE_URL` to your API's base URL.
2. In `src/api/client.js`, for each endpoint you've implemented, delete the
   mock line and uncomment the `request(...)` line above it.
3. Adjust the `request()` helper at the top of that file if your backend uses
   a different auth scheme — it currently sends `Authorization: Bearer <token>`
   using a token stored in `localStorage.deco_token`, set automatically on
   login/signup.

You don't need to touch any page component to do this — every page just calls
`api.getX()` and re-renders when the promise resolves.

## Project structure

```
src/
  api/client.js           <- the ONE file to edit for backend integration
  data/mockData.js        <- placeholder data, mirrors the Figma content
  context/AuthContext.jsx <- login/signup/logout state, persisted to localStorage
  components/             <- StatCard, DataTable, StatusBadge, Modal, Tabs, PageHeader
  layouts/                <- Sidebar, Topbar, AppShell, AuthLayout
  pages/
    Login.jsx, Signup.jsx
    Dashboard.jsx
    LiveFeed.jsx
    Alerts.jsx
    People.jsx
    Attendance.jsx
    Workforce.jsx
    Footfall.jsx
    Intrusion.jsx
    CameraManagement.jsx
    SiteManagement.jsx
    settings/
      SettingsLayout.jsx, Profile.jsx, Notifications.jsx,
      RulesPolicy.jsx, HolidayCalendar.jsx
```

## Design tokens

Colors, radii, and shared component classes (`.btn-primary`, `.card`, `.badge`,
`.input-field`, `.data-table`, etc.) live in `src/index.css` under Tailwind
v4's `@theme` block. Adjust the hex values there to match your brand instead
of hunting through every page — everything references those tokens.

## Notes / things to double check against your real data

- Sidebar badge counts (e.g. the "2" on Alerts & Events) are hardcoded in
  `src/layouts/Sidebar.jsx` — wire to your alerts-summary endpoint once it's live.
- `StatusBadge` (`src/components/StatusBadge.jsx`) maps known status strings
  (Active, Resolved, Enrolled, On time, etc.) to colors — extend `TONE_MAP`
  there if your backend returns different label text.
- Charts (Workforce attendance trend, Footfall peak-traffic) use placeholder
  series defined directly in their page files — swap for real time-series
  data once your analytics endpoints exist.
- The Intrusion, Camera, Site, and People "Add" modals currently just call
  the mock `api.addX()` functions and re-fetch the list — once your backend
  validates and returns the created record, you may want to append it
  directly instead of re-fetching.

## Build for production

```bash
npm run build
```

Outputs to `dist/`. Already verified this builds cleanly with no errors.
