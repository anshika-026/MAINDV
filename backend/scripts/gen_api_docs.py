"""
Generate API_DOCUMENTATION.md from the application's own route table, so it
only ever documents endpoints that exist, with the access rule each one
actually enforces (read from its dependency chain). Run from backend/:

    python -m scripts.gen_api_docs            # writes ../API_DOCUMENTATION.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("APP_ENV", "test")
os.environ["ENABLE_API_DOCS"] = "true"

from fastapi.routing import APIRoute, APIWebSocketRoute  # noqa: E402

from app import main  # noqa: E402

GROUPS = [
    ("Health", ("/health", "/ready", "/api/health")),
    ("Authentication (admin)", ("/api/auth",)),
    ("Licensing and client portal", ("/api/licenses",)),
    ("Cameras and sites", ("/api/cameras", "/api/sites")),
    ("Face training", ("/api/faces/training",)),
    ("Face management (review queue, enrollment, Identity)", ("/api/faces",)),
    ("Attendance, workforce, dashboard", ("/api/attendance", "/api/workforce", "/api/dashboard")),
    ("Desk analytics", ("/api/desk",)),
    ("Footfall", ("/api/footfall",)),
    ("Staff count", ("/api/staff",)),
    ("Intrusion", ("/api/intrusion",)),
    ("Alerts", ("/api/alerts",)),
    ("Analytics switches and settings", ("/api/analytics", "/api/settings", "/api/stats")),
    ("Audit log", ("/api/audit",)),
]

HEADER = """# API Documentation

Generated from the running application by `backend/scripts/gen_api_docs.py`.
Do not edit by hand; regenerate after changing routes.

## Conventions

- Base path: `/api` (served behind nginx on the same origin as the web app).
- Authentication: `Authorization: Bearer <token>`. Tokens come from
  `POST /api/auth/login` (admin: email and password) or
  `POST /api/licenses/client-login` (client portal: license username and
  password). Tokens are opaque, stored server-side, expire after
  `SESSION_TTL_HOURS` (default 12), and are revoked on logout, password
  change, admin disable, and license suspension, credential reset or deletion.
- Access levels:
  - **Public**: no token.
  - **Admin**: an admin session. A client session gets `403`.
  - **Admin or client**: any session. Client results are limited to the
    cameras assigned to their license.
  - **Admin, or client with the licensed feature**: a client also needs the
    feature enabled on their license, otherwise `403`.
- Errors are JSON `{"detail": "..."}`. Common codes: `401` missing, invalid
  or expired token; `403` authenticated but not allowed; `404` not found;
  `413` upload too large; `415` not an accepted image type; `422`
  validation failed; `429` too many failed logins (see `Retry-After`);
  `500` internal error (details only in the server log); `502`/`503` an
  external service (Identity) is unreachable or not configured.
- Uploads: JPEG, PNG or WebP only, at most `MAX_UPLOAD_MB` (default 5).

## Example

```bash
TOKEN=$(curl -s -X POST https://vision.example.com/api/auth/login \\
  -H 'Content-Type: application/json' \\
  -d '{"email":"ops@example.com","password":"..."}' | jq -r .token)
curl -s https://vision.example.com/api/cameras -H "Authorization: Bearer $TOKEN"
```
"""

WS_DOCS = {
    "/ws/live/{camera_id}": "Admin, or client whose license includes the camera. Binary JPEG frames. Query: `token`, "
                            "`plain=1` (no overlay detection), `w` (scale width, 320-3840).",
    "/ws/detections/{camera_id}": "Same access as `/ws/live`. JSON `{people: [{track_id, bbox, employee_id, name, color, "
                                  "confidence, identity_source, expression, expression_confidence}], fire_smoke: [], frame_w, "
                                  "frame_h}` about 3 times a second.",
    "/ws/staff": "Admin, or client with the attendance feature (counts limited to their cameras). JSON staff count every 2 s.",
    "/ws/staff/debug/{camera_id}": "Admin only. Raw tracker state for one entrance camera, 4 times a second.",
}


def _flat_routes():
    """(route, router-level dependencies) for every route, including those of
    included routers (newer FastAPI keeps them nested)."""
    for r in main.app.routes:
        inner = getattr(r, "original_router", None)
        if inner is not None:
            deps = list(getattr(inner, "dependencies", []) or [])
            for sub in inner.routes:
                yield sub, deps
        else:
            yield r, []


def _dep_names(route, router_deps=()) -> set[str]:
    names = {getattr(d.dependency, "__qualname__", "") for d in router_deps}

    def walk(dep):
        for d in dep.dependencies:
            names.add(getattr(d.call, "__qualname__", getattr(d.call, "__name__", "")))
            walk(d)

    walk(route.dependant)
    return names


def _access(route, router_deps=()) -> str:
    names = _dep_names(route, router_deps)
    if any("require_admin" in n for n in names):
        return "Admin"
    if any("require_feature" in n for n in names):
        return "Admin, or client with the licensed feature (limited to their cameras)"
    if any("get_current_client" in n for n in names):
        return "Client session"
    if any("get_principal" in n for n in names):
        return "Admin or client (client limited to their cameras)"
    return "Public"


def _group(path: str) -> str:
    for name, prefixes in GROUPS:
        if any(path == p or path.startswith(p) for p in prefixes):
            return name
    return "Other"


def _fields(schema: dict, components: dict) -> list[str]:
    if "$ref" in schema:
        schema = components[schema["$ref"].split("/")[-1]]
    required = set(schema.get("required", []))
    out = []
    for k, v in (schema.get("properties") or {}).items():
        t = v.get("type") or "|".join(a.get("type", "object") for a in v.get("anyOf", [])) or "object"
        bounds = ", ".join(f"{b} {v[b]}" for b in ("minimum", "maximum", "minLength", "maxLength") if b in v)
        out.append(f"`{k}` ({t}{', required' if k in required else ''}{', ' + bounds if bounds else ''})")
    return out


def build() -> tuple[str, int, int]:
    spec = main.app.openapi()
    components = spec.get("components", {}).get("schemas", {})
    groups: dict[str, list[str]] = {}
    count = 0
    for route, router_deps in _flat_routes():
        if not isinstance(route, APIRoute):
            continue
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            op = spec["paths"].get(route.path, {}).get(method.lower(), {})
            access = _access(route, router_deps)
            lines = [f"#### `{method} {route.path}`", "", f"- **Access:** {access}"]
            doc = " ".join((route.endpoint.__doc__ or "").strip().split("\n\n")[0].split())
            if doc:
                lines.append(f"- **Description:** {doc[:400]}")
            params = [f"`{p['name']}` ({p['in']}{', required' if p.get('required') else ''})" for p in op.get("parameters", [])]
            if params:
                lines.append(f"- **Parameters:** {', '.join(params)}")
            content = op.get("requestBody", {}).get("content", {})
            for ctype in ("application/json", "multipart/form-data"):
                if ctype in content:
                    lines.append(f"- **Request body ({ctype}):** {', '.join(_fields(content[ctype]['schema'], components)) or 'object'}")
            codes = set(op.get("responses", {}).keys())
            if access != "Public":
                codes |= {"401"}
            if access.startswith("Admin") and access != "Admin or client (client limited to their cameras)":
                codes |= {"403"}
            lines.append(f"- **Status codes:** {', '.join(sorted(codes))}")
            groups.setdefault(_group(route.path), []).append("\n".join(lines) + "\n")
            count += 1
    out = [HEADER]
    for name, _ in GROUPS + [("Other", ())]:
        if name in groups:
            out.append(f"\n## {name}\n")
            out.extend(groups[name])
    out.append("\n## WebSockets\n")
    out.append("Browsers can't send an `Authorization` header on a WebSocket handshake, so these take the session token as "
               "`?token=`. The server re-checks the session every 30 s and closes with **4401** (not authenticated, or the "
               "session ended) or **4403** (not allowed: feed switched off, or an admin-only stream).\n")
    ws = [r for r, _ in _flat_routes() if isinstance(r, APIWebSocketRoute)]
    for r in ws:
        out.append(f"#### `WS {r.path}`\n\n- {WS_DOCS.get(r.path, 'See source.')}\n")
    return "\n".join(out), count, len(ws)


if __name__ == "__main__":
    text, n_http, n_ws = build()
    target = Path(__file__).resolve().parent.parent.parent / "API_DOCUMENTATION.md"
    target.write_text(text, encoding="utf-8")
    print(f"wrote {target}: {n_http} HTTP operations, {n_ws} websockets")
