"""License & Camera Access Management — companies, licenses, per-license
camera assignments, and license/camera feature toggles. Reuses the existing
cameras table (camera_db.py) rather than duplicating a camera registry — a
CameraAssignment here just links a license to a row already in that table.

Scope note: companies, licenses (with real hashed-password portal
credentials), camera assignments, license/camera feature toggles, and QR
generation (license_qr.py). Session issuing/validation and per-request
enforcement (status/expiry/ownership checks) live in auth.py, used from
main.py and license_routes.py — this module owns the data, auth.py owns
proving who's asking. Deliberately still NOT here: audit logs, device
binding, and per-user (as opposed to per-license) camera permissions —
this app has one login per license/client, not multiple named users per
company, so a "which user on Client A's team did this" audit trail has no
identity to attach to yet without a bigger multi-user-per-client model.

UUID primary keys throughout (stored as TEXT, generated with uuid4) — a
deliberate departure from the INTEGER AUTOINCREMENT ids used elsewhere in
this app: a license key/id being guessable-in-sequence is a real exposure
this module specifically needs to avoid, unlike e.g. a camera row id.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import secrets
import sqlite3
import time
import uuid as uuid_lib
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"

STATUS_ACTIVE = "active"
STATUS_INACTIVE = "inactive"
STATUS_SUSPENDED = "suspended"
ALL_STATUSES = (STATUS_ACTIVE, STATUS_INACTIVE, STATUS_SUSPENDED)

# "expired" is never SET by an admin (not in ALL_STATUSES, which is the
# settable/stored set) — it's always DERIVED from expires_at vs. the
# current time, computed fresh on every read via effective_status() below.
# This is what makes expiry enforcement real: even if the stored `status`
# column still says 'active', a client whose expires_at has passed gets
# treated as expired on their very next request (see auth.py's
# load_active_client_license), not just when an admin happens to notice
# and flip a status flag.
STATUS_EXPIRED = "expired"
ALL_EFFECTIVE_STATUSES = ALL_STATUSES + (STATUS_EXPIRED,)

# Fixed catalog, not a DB table — a closed, rarely-changing set (like
# ALL_STATUSES above). Every key traces to an existing feature already
# built (or partially built) into this app; adding a new one later is a
# one-line addition here, same as adding a status would be.
FEATURE_FACE_RECOGNITION = "face_recognition"
FEATURE_ATTENDANCE = "attendance"
FEATURE_FOOTFALL_ANALYTICS = "footfall_analytics"
FEATURE_WORKFORCE_ANALYTICS = "workforce_analytics"
FEATURE_INTRUSION_DETECTION = "intrusion_detection"
FEATURE_SMOKE_DETECTION = "smoke_detection"
FEATURE_CROWD_ANALYTICS = "crowd_analytics"
FEATURE_THREAT_DETECTION = "threat_detection"
FEATURE_VEHICLE_DETECTION = "vehicle_detection"
ALL_FEATURES = (
    FEATURE_FACE_RECOGNITION, FEATURE_ATTENDANCE, FEATURE_FOOTFALL_ANALYTICS,
    FEATURE_WORKFORCE_ANALYTICS, FEATURE_INTRUSION_DETECTION, FEATURE_SMOKE_DETECTION,
    FEATURE_CROWD_ANALYTICS, FEATURE_THREAT_DETECTION, FEATURE_VEHICLE_DETECTION,
)
FEATURE_LABELS = {
    FEATURE_FACE_RECOGNITION: "Face Recognition",
    FEATURE_ATTENDANCE: "Attendance",
    FEATURE_FOOTFALL_ANALYTICS: "Footfall Analytics",
    FEATURE_WORKFORCE_ANALYTICS: "Workforce Analytics",
    FEATURE_INTRUSION_DETECTION: "Intrusion Detection",
    FEATURE_SMOKE_DETECTION: "Smoke Detection",
    FEATURE_CROWD_ANALYTICS: "Crowd Analytics",
    FEATURE_THREAT_DETECTION: "Threat Detection",
    FEATURE_VEHICLE_DETECTION: "Vehicle Detection",
}

# Cameras are sold outright, not leased/subscribed — a license never
# expires or needs renewing on its own; the only way out of "active" is an
# explicit admin action (disable or suspend). expires_at still exists as a
# DB column — new licenses just get it set far enough out that it can
# never practically matter.
_NEVER_EXPIRES_SECONDS = 100 * 365 * 86400

# Distinct from _NEVER_EXPIRES_SECONDS above (what NEW licenses are written
# with): this is the threshold the UI uses to decide whether an expires_at
# value is "real" or just the vestigial default — anything further out than
# 10 years is treated as non-expiring for display purposes.
NON_EXPIRING_HORIZON_SECONDS = 10 * 365 * 86400

# Ambiguity-free alphabet (no 0/O, 1/I/L) for license keys people may need
# to type by hand off a printout, not just scan.
_KEY_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_KEY_GROUPS = 4
_KEY_GROUP_LEN = 4


@contextlib.contextmanager
def get_connection():
    """Closed on exit — sqlite3's own `with conn:` never closes the
    connection, which would leak a file descriptor per call."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS companies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                portal_slug TEXT UNIQUE,
                created_at REAL NOT NULL
            )
            """
        )
        _migrate_companies_table(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS licenses (
                id TEXT PRIMARY KEY,
                license_key TEXT NOT NULL UNIQUE,
                company_id TEXT NOT NULL,
                label TEXT DEFAULT '',
                max_cameras INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'active',
                username TEXT UNIQUE,
                password_hash TEXT,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (company_id) REFERENCES companies (id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_licenses_company ON licenses (company_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_licenses_status ON licenses (status)")
        _migrate_licenses_table(conn)

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS camera_assignments (
                id TEXT PRIMARY KEY,
                license_id TEXT NOT NULL,
                camera_id INTEGER NOT NULL,
                assigned_at REAL NOT NULL,
                UNIQUE (license_id, camera_id),
                FOREIGN KEY (license_id) REFERENCES licenses (id),
                FOREIGN KEY (camera_id) REFERENCES cameras (id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_camassign_license ON camera_assignments (license_id)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS license_features (
                id TEXT PRIMARY KEY,
                license_id TEXT NOT NULL,
                feature_key TEXT NOT NULL,
                enabled_at REAL NOT NULL,
                UNIQUE (license_id, feature_key),
                FOREIGN KEY (license_id) REFERENCES licenses (id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_license_features_license ON license_features (license_id)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS camera_features (
                id TEXT PRIMARY KEY,
                license_id TEXT NOT NULL,
                camera_id INTEGER NOT NULL,
                feature_key TEXT NOT NULL,
                enabled_at REAL NOT NULL,
                UNIQUE (camera_id, feature_key),
                FOREIGN KEY (license_id) REFERENCES licenses (id),
                FOREIGN KEY (camera_id) REFERENCES cameras (id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_camera_features_camera ON camera_features (camera_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_camera_features_license ON camera_features (license_id)")


def _migrate_licenses_table(conn: sqlite3.Connection) -> None:
    """CREATE TABLE IF NOT EXISTS doesn't add new columns to an
    already-existing table — backfills username/password_hash for a
    licenses table created before the username/password login feature
    existed (mirrors camera_db.py's _migrate_cameras_table)."""
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(licenses)").fetchall()}
    if "username" not in existing_cols:
        conn.execute("ALTER TABLE licenses ADD COLUMN username TEXT")
    if "password_hash" not in existing_cols:
        conn.execute("ALTER TABLE licenses ADD COLUMN password_hash TEXT")


def _migrate_companies_table(conn: sqlite3.Connection) -> None:
    """Backfills portal_slug for a companies table created before the
    client-portal-URL feature existed, and for any company row inserted
    with no slug for some other reason."""
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(companies)").fetchall()}
    if "portal_slug" not in existing_cols:
        conn.execute("ALTER TABLE companies ADD COLUMN portal_slug TEXT")
    rows = conn.execute("SELECT id, name FROM companies WHERE portal_slug IS NULL OR portal_slug = ''").fetchall()
    for company_id, name in rows:
        conn.execute(
            "UPDATE companies SET portal_slug = ? WHERE id = ?",
            (_unique_slug(conn, name), company_id),
        )


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "client"


def _unique_slug(conn: sqlite3.Connection, name: str) -> str:
    """Used both for a brand-new company and for backfilling old rows —
    this is the client portal's dev-mode identifier (see
    /client/{slug}/login in App.jsx), the local equivalent of
    client-<slug>.decovision.com until real subdomain routing exists in
    production (see the deployment notes in the accompanying report)."""
    base = _slugify(name)
    slug = base
    n = 2
    while conn.execute("SELECT 1 FROM companies WHERE portal_slug = ?", (slug,)).fetchone() is not None:
        slug = f"{base}-{n}"
        n += 1
    return slug


# --- Password hashing ------------------------------------------------------
# PBKDF2-HMAC-SHA256 with a per-license random salt (stdlib only — no new
# dependency). Client-portal login is the first real password check in this
# app (the existing admin /api/auth/login is unrelated and still doesn't
# check a password — see BACKEND_HANDOFF.md), so a plaintext or unsalted
# hash here isn't acceptable even at this app's otherwise-informal security
# level.
_PBKDF2_ITERATIONS = 200_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ITERATIONS).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, digest = stored_hash.split("$", 1)
    except (ValueError, AttributeError):
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ITERATIONS).hex()
    return secrets.compare_digest(candidate, digest)


# --- Companies ---------------------------------------------------------

def create_company(name: str) -> dict:
    company_id = uuid_lib.uuid4().hex
    with get_connection() as conn:
        slug = _unique_slug(conn, name)
        conn.execute(
            "INSERT INTO companies (id, name, portal_slug, created_at) VALUES (?, ?, ?, ?)",
            (company_id, name, slug, time.time()),
        )
    return get_company(company_id)


def get_company(company_id: str) -> dict | None:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM companies WHERE id = ?", (company_id,)).fetchone()
    return dict(row) if row else None


def get_company_by_slug(slug: str) -> dict | None:
    """Public-safe lookup (company name only, via the route layer) for the
    dev-mode client portal route /client/{slug}/login — the slug alone
    never grants access, it just lets that login page show which company
    it belongs to. See license_routes.py's public_company_by_slug."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM companies WHERE portal_slug = ?", (slug,)).fetchone()
    return dict(row) if row else None


def list_companies() -> list[dict]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM companies ORDER BY name").fetchall()
    return [dict(r) for r in rows]


# --- Licenses ------------------------------------------------------------

def generate_license_key() -> str:
    groups = [
        "".join(secrets.choice(_KEY_ALPHABET) for _ in range(_KEY_GROUP_LEN))
        for _ in range(_KEY_GROUPS)
    ]
    return "-".join(groups)


def create_license(
    company_id: str, max_cameras: int, label: str = "",
    username: str | None = None, password: str | None = None,
    expires_at: float | None = None,
) -> dict:
    """username/password are the client portal login credentials for this
    license (see get_license_by_username/verify_license_login below) — the
    portal-access mechanism, replacing "give them the license key and it
    only works on one device" (a QR/license-key activation flow was
    considered but a QR still ties to whatever scans it; a username+
    password works from any browser). Both optional at the DB layer so
    existing rows created before this feature keep working; the API layer
    (license_routes.py) requires them for new licenses.

    expires_at: pass an explicit epoch timestamp for a real expiry date;
    omit (None) for the previous "never practically expires" default —
    see _NEVER_EXPIRES_SECONDS."""
    license_id = uuid_lib.uuid4().hex
    now = time.time()
    key = generate_license_key()
    password_hash = hash_password(password) if password else None
    if expires_at is None:
        expires_at = now + _NEVER_EXPIRES_SECONDS
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO licenses "
            "(id, license_key, company_id, label, max_cameras, status, username, password_hash, "
            " expires_at, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (license_id, key, company_id, label, max_cameras, STATUS_ACTIVE, username, password_hash,
             expires_at, now, now),
        )
    return get_license(license_id)


def effective_status(lic: dict) -> str:
    """The status that actually governs access right now — computed
    fresh every call, never stored. A license explicitly set to
    'inactive'/'suspended' stays that way regardless of expiry; an
    'active' license whose expires_at has passed is treated as
    STATUS_EXPIRED even though the stored `status` column still says
    'active' (see the module-level STATUS_EXPIRED comment for why this
    is never written back — only ever derived)."""
    if lic["status"] != STATUS_ACTIVE:
        return lic["status"]
    if lic["expires_at"] and lic["expires_at"] <= time.time():
        return STATUS_EXPIRED
    return STATUS_ACTIVE


def _license_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d.pop("password_hash", None)  # never leaves this module
    return d


def get_license(license_id: str) -> dict | None:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM licenses WHERE id = ?", (license_id,)).fetchone()
    return _license_row_to_dict(row) if row else None


def get_license_by_key(license_key: str) -> dict | None:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM licenses WHERE license_key = ?", (license_key,)).fetchone()
    return _license_row_to_dict(row) if row else None


def username_taken(username: str, exclude_license_id: str | None = None) -> bool:
    with get_connection() as conn:
        if exclude_license_id:
            row = conn.execute(
                "SELECT 1 FROM licenses WHERE username = ? AND id != ?", (username, exclude_license_id)
            ).fetchone()
        else:
            row = conn.execute("SELECT 1 FROM licenses WHERE username = ?", (username,)).fetchone()
    return row is not None


def set_license_credentials(license_id: str, username: str, password: str) -> dict | None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE licenses SET username = ?, password_hash = ?, updated_at = ? WHERE id = ?",
            (username, hash_password(password), time.time(), license_id),
        )
    return get_license(license_id)


def verify_license_login(username: str, password: str) -> dict | None:
    """Returns the license dict (password_hash stripped) if username/password
    match a license with credentials set, else None. Does NOT check
    status — callers (license_routes.py) decide what a non-active license
    should do at login (currently: reject with a clear reason), keeping
    that policy at the API layer rather than baked into this lookup."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM licenses WHERE username = ?", (username,)).fetchone()
    if row is None or not row["password_hash"] or not verify_password(password, row["password_hash"]):
        return None
    return _license_row_to_dict(row)


def list_licenses(
    company_id: str | None = None, status: str | None = None, search: str = "",
    limit: int = 50, offset: int = 0,
) -> tuple[list[dict], int]:
    """Returns (rows, total_count) for pagination."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        clauses, params = [], []
        if company_id:
            clauses.append("company_id = ?")
            params.append(company_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if search:
            clauses.append("(license_key LIKE ? OR label LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        total = conn.execute(f"SELECT COUNT(*) FROM licenses {where}", params).fetchone()[0]
        rows = [
            _license_row_to_dict(r) for r in conn.execute(
                f"SELECT * FROM licenses {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        ]
    return rows, total


def update_license(license_id: str, **fields) -> dict | None:
    allowed = {"label", "max_cameras", "expires_at"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return get_license(license_id)
    updates["updated_at"] = time.time()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    with get_connection() as conn:
        conn.execute(f"UPDATE licenses SET {set_clause} WHERE id = ?", (*updates.values(), license_id))
    return get_license(license_id)


def set_license_status(license_id: str, status: str) -> dict | None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE licenses SET status = ?, updated_at = ? WHERE id = ?",
            (status, time.time(), license_id),
        )
    return get_license(license_id)


def delete_license(license_id: str) -> None:
    """Permanently removes a license and everything that only exists
    because of it (camera assignments, per-camera and per-license feature
    grants) — the cameras/company themselves are untouched, only the
    license's own rows. Unlike set_license_status (suspend/deactivate,
    reversible), this cannot be undone — the route layer is expected to
    have the user confirm first."""
    with get_connection() as conn:
        conn.execute("DELETE FROM camera_features WHERE license_id = ?", (license_id,))
        conn.execute("DELETE FROM camera_assignments WHERE license_id = ?", (license_id,))
        conn.execute("DELETE FROM license_features WHERE license_id = ?", (license_id,))
        conn.execute("DELETE FROM licenses WHERE id = ?", (license_id,))


# --- Camera assignments ----------------------------------------------------

def assign_cameras(license_id: str, camera_ids: list[int]) -> int:
    """Bulk-assigns cameras to a license, skipping ones already assigned.
    Returns how many were newly added (caller enforces the max_cameras cap
    before calling this)."""
    now = time.time()
    added = 0
    with get_connection() as conn:
        for camera_id in camera_ids:
            try:
                conn.execute(
                    "INSERT INTO camera_assignments (id, license_id, camera_id, assigned_at) VALUES (?, ?, ?, ?)",
                    (uuid_lib.uuid4().hex, license_id, camera_id, now),
                )
                added += 1
            except sqlite3.IntegrityError:
                continue  # already assigned — not an error, just a no-op
    return added


def remove_cameras(license_id: str, camera_ids: list[int]) -> None:
    """Also cascades to camera_features — otherwise a camera unassigned
    from a license would leave stale feature grants behind, silently
    outliving the assignment row they belonged to."""
    with get_connection() as conn:
        placeholders = ",".join("?" * len(camera_ids))
        conn.execute(
            f"DELETE FROM camera_assignments WHERE license_id = ? AND camera_id IN ({placeholders})",
            (license_id, *camera_ids),
        )
        conn.execute(
            f"DELETE FROM camera_features WHERE license_id = ? AND camera_id IN ({placeholders})",
            (license_id, *camera_ids),
        )


def list_cameras_for_license(license_id: str) -> list[dict]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT c.id, c.name, c.site, c.status AS camera_status, ca.assigned_at
            FROM camera_assignments ca
            JOIN cameras c ON c.id = ca.camera_id
            WHERE ca.license_id = ?
            ORDER BY ca.assigned_at
            """,
            (license_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def count_cameras_for_license(license_id: str) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM camera_assignments WHERE license_id = ?", (license_id,)
        ).fetchone()
    return row[0]


def is_camera_assigned(license_id: str, camera_id: int) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM camera_assignments WHERE license_id = ? AND camera_id = ?",
            (license_id, camera_id),
        ).fetchone()
    return row is not None


def count_assigned_cameras() -> int:
    """How many distinct cameras have been handed out via ANY license."""
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(DISTINCT camera_id) FROM camera_assignments").fetchone()
    return row[0]


def get_license_for_camera(camera_id: int) -> dict | None:
    """Which license (if any) a camera is assigned to."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT license_id FROM camera_assignments WHERE camera_id = ? LIMIT 1", (camera_id,)
        ).fetchone()
    return get_license(row["license_id"]) if row else None


# --- License-level features -------------------------------------------

def set_license_features(license_id: str, feature_keys: list[str]) -> list[str]:
    """Full-replace, not assign/remove — the UI saves from a single
    checklist, so "this is now the enabled set" is simpler than a two-call
    add/remove pair. Silently drops any key not in ALL_FEATURES (defensive;
    the route validates first)."""
    keys = [k for k in feature_keys if k in ALL_FEATURES]
    now = time.time()
    with get_connection() as conn:
        conn.execute("DELETE FROM license_features WHERE license_id = ?", (license_id,))
        for key in keys:
            conn.execute(
                "INSERT INTO license_features (id, license_id, feature_key, enabled_at) VALUES (?, ?, ?, ?)",
                (uuid_lib.uuid4().hex, license_id, key, now),
            )
    return keys


def list_license_features(license_id: str) -> list[str]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT feature_key FROM license_features WHERE license_id = ?", (license_id,)
        ).fetchall()
    return [r[0] for r in rows]


# --- Per-camera feature overrides ---------------------------------------
#
# A camera's enabled features must be a subset of its license's enabled
# features (set_license_features above) — enforced here, not just at the
# API layer, so any future caller of this module gets the same guarantee.
# A camera must already be in camera_assignments for this license before
# it can have a features row — see remove_cameras' cascade-delete above.

def set_camera_features(license_id: str, camera_id: int, feature_keys: list[str]) -> list[str]:
    """Returns the subset of feature_keys NOT enabled on the license (i.e.
    invalid). Empty list = all valid and the update was applied; a
    non-empty list means nothing was written and the caller should reject
    with 400 listing these keys."""
    licensed = set(list_license_features(license_id))
    invalid = [k for k in feature_keys if k not in licensed]
    if invalid:
        return invalid
    now = time.time()
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM camera_features WHERE license_id = ? AND camera_id = ?", (license_id, camera_id)
        )
        for key in feature_keys:
            conn.execute(
                "INSERT INTO camera_features (id, license_id, camera_id, feature_key, enabled_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (uuid_lib.uuid4().hex, license_id, camera_id, key, now),
            )
    return []


def get_camera_features(camera_id: int) -> list[str]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT feature_key FROM camera_features WHERE camera_id = ?", (camera_id,)
        ).fetchall()
    return [r[0] for r in rows]


def list_camera_features_bulk(camera_ids: list[int]) -> dict[int, list[str]]:
    """Avoids N+1 queries when rendering the camera list for a license."""
    if not camera_ids:
        return {}
    out: dict[int, list[str]] = {cid: [] for cid in camera_ids}
    placeholders = ",".join("?" * len(camera_ids))
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT camera_id, feature_key FROM camera_features WHERE camera_id IN ({placeholders})",
            camera_ids,
        ).fetchall()
    for camera_id, feature_key in rows:
        out[camera_id].append(feature_key)
    return out


def list_cameras_for_license_detailed(license_id: str) -> list[dict]:
    """Like list_cameras_for_license, but with enabled_features added."""
    cameras = list_cameras_for_license(license_id)
    camera_ids = [c["id"] for c in cameras]
    features_by_camera = list_camera_features_bulk(camera_ids)
    for c in cameras:
        c["enabled_features"] = features_by_camera.get(c["id"], [])
    return cameras


# --- Analytics -----------------------------------------------------------

def get_analytics(online_camera_count: int | None = None) -> dict:
    """online_camera_count can be supplied by the caller (actual live RTSP
    connectivity, if ever available); falls back to counting cameras with
    status='active' in the cameras table (has connection details, not
    necessarily "currently connected") when not supplied."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        all_licenses = [dict(r) for r in conn.execute("SELECT * FROM licenses").fetchall()]
        total_cameras = conn.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
        cameras_assigned = conn.execute("SELECT COUNT(DISTINCT camera_id) FROM camera_assignments").fetchone()[0]
        if online_camera_count is None:
            cameras_online = conn.execute("SELECT COUNT(*) FROM cameras WHERE status = 'active'").fetchone()[0]
        else:
            cameras_online = online_camera_count

    by_status = {s: 0 for s in ALL_STATUSES}
    for lic in all_licenses:
        by_status[lic["status"]] = by_status.get(lic["status"], 0) + 1

    total_license_camera_capacity = sum(lic["max_cameras"] for lic in all_licenses if lic["status"] == STATUS_ACTIVE)

    return {
        "total_licenses": len(all_licenses),
        "active_licenses": by_status[STATUS_ACTIVE],
        "inactive_licenses": by_status[STATUS_INACTIVE],
        "suspended_licenses": by_status[STATUS_SUSPENDED],
        "total_cameras": total_cameras,
        "cameras_assigned": cameras_assigned,
        "cameras_online": cameras_online,
        "cameras_offline": max(0, total_cameras - cameras_online),
        "license_usage_percent": (
            round(100 * cameras_assigned / total_license_camera_capacity, 1)
            if total_license_camera_capacity else 0.0
        ),
        "camera_usage_percent": (
            round(100 * cameras_assigned / total_cameras, 1) if total_cameras else 0.0
        ),
    }
