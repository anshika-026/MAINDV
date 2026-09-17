"""license_routes.py

Core license-management endpoints: companies, licenses, camera
assignments, license/camera feature toggles, a QR code per license, and
client-portal login (username/password — see /client-login below). See
license_db.py's module docstring for what's deliberately NOT here.

Every route except the ones a client (or an anonymous visitor picking a
portal URL) legitimately needs before/without an admin session —
/client-login, /client-logout, /client-me, and the public company-by-slug
lookup — requires an admin session (auth.require_admin). Before this,
literally none of these endpoints checked who was calling them; anyone
who could reach this backend could create/delete licenses, reassign
cameras, or read every client's data. See auth.py for what an "admin
session" actually proves (and its documented limitation).

Mount with: app.include_router(license_routes.router) in main.py
"""

from __future__ import annotations

from datetime import datetime, time as dtime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app import auth, camera_db, license_db, license_qr

router = APIRouter(prefix="/api/licenses", tags=["licenses"])


def _parse_expiry_date(expiry_date: str | None) -> float | None:
    """expiry_date is a plain "YYYY-MM-DD" from an HTML <input type=date>
    — parsed as end-of-that-day (local, naive — matches the rest of this
    app's epoch-float timestamps, none of which are timezone-aware
    either) so a license is valid through its expiry date, not from its
    first moment. None/"" means "use the no-real-expiry default"."""
    if not expiry_date:
        return None
    try:
        d = datetime.fromisoformat(expiry_date).date()
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid expiry_date '{expiry_date}', expected YYYY-MM-DD")
    return datetime.combine(d, dtime(23, 59, 59)).timestamp()


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------

class CompanyIn(BaseModel):
    name: str


@router.get("/companies")
def list_companies(_: dict = Depends(auth.require_admin)):
    return license_db.list_companies()


@router.post("/companies")
def create_company(payload: CompanyIn, _: dict = Depends(auth.require_admin)):
    if not payload.name.strip():
        raise HTTPException(status_code=422, detail="Company name is required")
    return license_db.create_company(payload.name.strip())


@router.get("/companies/slug/{slug}")
def public_company_by_slug(slug: str):
    """Deliberately PUBLIC (no auth) — this is the dev-mode equivalent of
    visiting client-<slug>.decovision.com and seeing which company's
    portal you're at, same as a real subdomain would reveal before any
    login form is even submitted. Returns only the display name; never
    cameras, features, or anything else. The slug never grants access —
    /client-login below still requires the real username/password."""
    company = license_db.get_company_by_slug(slug)
    if company is None:
        raise HTTPException(status_code=404, detail="No client portal at this address")
    return {"name": company["name"], "portal_slug": company["portal_slug"]}


# ---------------------------------------------------------------------------
# Feature catalog (static, for the frontend's checklist UI)
# ---------------------------------------------------------------------------

@router.get("/features")
def list_feature_catalog(_: dict = Depends(auth.require_admin)):
    return [{"key": k, "label": license_db.FEATURE_LABELS[k]} for k in license_db.ALL_FEATURES]


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@router.get("/analytics")
def analytics(_: dict = Depends(auth.require_admin)):
    return license_db.get_analytics()


# ---------------------------------------------------------------------------
# Licenses
# ---------------------------------------------------------------------------

class LicenseIn(BaseModel):
    company_id: str
    max_cameras: int = 1
    label: str = ""
    feature_keys: list[str] = []
    username: str
    password: str
    expiry_date: str | None = None  # "YYYY-MM-DD"; omit for no real expiry


class LicenseUpdate(BaseModel):
    label: str | None = None
    max_cameras: int | None = None
    expiry_date: str | None = None


class LicenseStatusIn(BaseModel):
    status: str


class LicenseCredentialsIn(BaseModel):
    username: str
    password: str


class ClientLoginIn(BaseModel):
    username: str
    password: str


def _license_public(row: dict) -> dict:
    company = license_db.get_company(row["company_id"])
    return {
        **row,
        "cameras_assigned": license_db.count_cameras_for_license(row["id"]),
        "features": license_db.list_license_features(row["id"]),
        "non_expiring": (row["expires_at"] - row["created_at"]) >= license_db.NON_EXPIRING_HORIZON_SECONDS,
        # Computed, not the raw `status` column — see license_db.effective_status
        # for why an active-but-expired license must show as "expired" here
        # regardless of what the stored status says.
        "effective_status": license_db.effective_status(row),
        "company_name": company["name"] if company else None,
        "company_portal_slug": company["portal_slug"] if company else None,
    }


@router.get("")
def list_licenses(company_id: str | None = None, status: str | None = None, search: str = "",
                   limit: int = 50, offset: int = 0, _: dict = Depends(auth.require_admin)):
    if status is not None and status not in license_db.ALL_STATUSES:
        raise HTTPException(status_code=422, detail=f"Invalid status '{status}'")
    rows, total = license_db.list_licenses(company_id, status, search, limit, offset)
    return {"licenses": [_license_public(r) for r in rows], "total": total}


@router.post("")
def create_license(payload: LicenseIn, _: dict = Depends(auth.require_admin)):
    if license_db.get_company(payload.company_id) is None:
        raise HTTPException(status_code=404, detail="Company not found")
    if payload.max_cameras < 1:
        raise HTTPException(status_code=422, detail="max_cameras must be at least 1")
    invalid = [k for k in payload.feature_keys if k not in license_db.ALL_FEATURES]
    if invalid:
        raise HTTPException(status_code=422, detail=f"Unknown feature key(s): {invalid}")
    if not payload.username.strip() or not payload.password:
        raise HTTPException(status_code=422, detail="username and password are required")
    if len(payload.password) < 6:
        raise HTTPException(status_code=422, detail="password must be at least 6 characters")
    if license_db.username_taken(payload.username.strip()):
        raise HTTPException(status_code=409, detail=f"Username '{payload.username}' is already taken")
    expires_at = _parse_expiry_date(payload.expiry_date)

    lic = license_db.create_license(
        payload.company_id, payload.max_cameras, payload.label,
        username=payload.username.strip(), password=payload.password,
        expires_at=expires_at,
    )
    if payload.feature_keys:
        license_db.set_license_features(lic["id"], payload.feature_keys)
    return _license_public(license_db.get_license(lic["id"]))


@router.put("/{license_id}/credentials")
def reset_license_credentials(license_id: str, payload: LicenseCredentialsIn, _: dict = Depends(auth.require_admin)):
    """Admin resets a client's portal username/password (e.g. they forgot
    it, or it needs rotating) — there's no self-service "forgot password"
    flow since there's no email-sending infrastructure in this app."""
    if license_db.get_license(license_id) is None:
        raise HTTPException(status_code=404, detail="License not found")
    if not payload.username.strip() or not payload.password:
        raise HTTPException(status_code=422, detail="username and password are required")
    if len(payload.password) < 6:
        raise HTTPException(status_code=422, detail="password must be at least 6 characters")
    if license_db.username_taken(payload.username.strip(), exclude_license_id=license_id):
        raise HTTPException(status_code=409, detail=f"Username '{payload.username}' is already taken")
    lic = license_db.set_license_credentials(license_id, payload.username.strip(), payload.password)
    return _license_public(lic)


@router.post("/client-login")
def client_login(payload: ClientLoginIn):
    """Client-portal login — separate from the existing admin /api/auth/login.
    A license's username/password lets that client sign in from any
    browser/device, unlike a device-bound QR/key. On success, issues a
    real server-side session token (auth.create_client_session) — every
    subsequent request re-validates this license's status/expiry fresh
    from the DB (see auth.load_active_client_license), so a suspend/
    expiry takes effect on the client's very next request, not just their
    next login."""
    lic = license_db.verify_license_login(payload.username.strip(), payload.password)
    if lic is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    effective = license_db.effective_status(lic)
    if effective != license_db.STATUS_ACTIVE:
        raise HTTPException(
            status_code=403,
            detail=f"This license is {effective} — contact your administrator",
        )
    token = auth.create_client_session(lic["id"])
    return _client_session_payload(lic, session_token=token)


def _client_session_payload(lic: dict, session_token: str | None = None) -> dict:
    company = license_db.get_company(lic["company_id"])
    cameras = license_db.list_cameras_for_license(lic["id"])
    payload = {
        "license_id": lic["id"],
        "username": lic["username"],
        "company_id": lic["company_id"],
        "company_name": company["name"] if company else "",
        "company_portal_slug": company["portal_slug"] if company else None,
        "label": lic["label"],
        "features": license_db.list_license_features(lic["id"]),
        "cameras": [{"id": c["id"], "name": c["name"], "site": c["site"]} for c in cameras],
        "max_cameras": lic["max_cameras"],
    }
    if session_token is not None:
        payload["session_token"] = session_token
    return payload


@router.post("/client-logout")
def client_logout(principal: dict = Depends(auth.get_principal)):
    if principal["principal_type"] == auth.CLIENT:
        auth.revoke_session(principal["token"])
    return {"ok": True}


@router.get("/client-me")
def client_me(client: dict = Depends(auth.get_current_client)):
    """Lets the frontend re-check (and refresh cameras/features for) an
    already-logged-in client session — polled periodically by
    AuthContext so a license change made by an admin (suspended,
    cameras/features changed) reaches an already-open client tab without
    requiring them to log out and back in."""
    return _client_session_payload(client["license"])


@router.get("/{license_id}")
def get_license(license_id: str, _: dict = Depends(auth.require_admin)):
    lic = license_db.get_license(license_id)
    if lic is None:
        raise HTTPException(status_code=404, detail="License not found")
    return _license_public(lic)


@router.put("/{license_id}")
def update_license(license_id: str, payload: LicenseUpdate, _: dict = Depends(auth.require_admin)):
    if license_db.get_license(license_id) is None:
        raise HTTPException(status_code=404, detail="License not found")
    if payload.max_cameras is not None and payload.max_cameras < 1:
        raise HTTPException(status_code=422, detail="max_cameras must be at least 1")
    if payload.max_cameras is not None:
        current = license_db.count_cameras_for_license(license_id)
        if payload.max_cameras < current:
            raise HTTPException(
                status_code=422,
                detail=f"Cannot set max_cameras below {current} cameras already assigned to this license",
            )
    expires_at = _parse_expiry_date(payload.expiry_date) if payload.expiry_date is not None else None
    lic = license_db.update_license(
        license_id, label=payload.label, max_cameras=payload.max_cameras, expires_at=expires_at,
    )
    return _license_public(lic)


@router.post("/{license_id}/status")
def set_license_status(license_id: str, payload: LicenseStatusIn, _: dict = Depends(auth.require_admin)):
    if license_db.get_license(license_id) is None:
        raise HTTPException(status_code=404, detail="License not found")
    if payload.status not in license_db.ALL_STATUSES:
        raise HTTPException(status_code=422, detail=f"Invalid status '{payload.status}'")
    lic = license_db.set_license_status(license_id, payload.status)
    return _license_public(lic)


@router.delete("/{license_id}")
def delete_license(license_id: str, _: dict = Depends(auth.require_admin)):
    if license_db.get_license(license_id) is None:
        raise HTTPException(status_code=404, detail="License not found")
    license_db.delete_license(license_id)
    return {"ok": True}


@router.get("/{license_id}/qr")
def license_qr_code(license_id: str, _: dict = Depends(auth.require_admin)):
    lic = license_db.get_license(license_id)
    if lic is None:
        raise HTTPException(status_code=404, detail="License not found")
    token = license_qr.build_qr_token(lic["license_key"])
    png = license_qr.render_qr_png(token)
    return Response(content=png, media_type="image/png")


# ---------------------------------------------------------------------------
# Camera assignments
# ---------------------------------------------------------------------------

class CameraIdsIn(BaseModel):
    camera_ids: list[int]


@router.get("/{license_id}/cameras")
def list_license_cameras(license_id: str, _: dict = Depends(auth.require_admin)):
    if license_db.get_license(license_id) is None:
        raise HTTPException(status_code=404, detail="License not found")
    return license_db.list_cameras_for_license_detailed(license_id)


@router.post("/{license_id}/cameras")
def assign_cameras(license_id: str, payload: CameraIdsIn, _: dict = Depends(auth.require_admin)):
    lic = license_db.get_license(license_id)
    if lic is None:
        raise HTTPException(status_code=404, detail="License not found")

    unknown = [cid for cid in payload.camera_ids if camera_db.get_camera(cid) is None]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown camera id(s): {unknown}")

    already = {c["id"] for c in license_db.list_cameras_for_license(license_id)}
    new_ids = [cid for cid in payload.camera_ids if cid not in already]
    if len(already) + len(new_ids) > lic["max_cameras"]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Assigning {len(new_ids)} camera(s) would exceed this license's "
                f"max_cameras ({lic['max_cameras']}) — currently {len(already)} assigned"
            ),
        )

    license_db.assign_cameras(license_id, payload.camera_ids)
    return license_db.list_cameras_for_license_detailed(license_id)


@router.delete("/{license_id}/cameras")
def unassign_cameras(license_id: str, payload: CameraIdsIn, _: dict = Depends(auth.require_admin)):
    if license_db.get_license(license_id) is None:
        raise HTTPException(status_code=404, detail="License not found")
    license_db.remove_cameras(license_id, payload.camera_ids)
    return license_db.list_cameras_for_license_detailed(license_id)


# ---------------------------------------------------------------------------
# Feature toggles
# ---------------------------------------------------------------------------

class FeatureKeysIn(BaseModel):
    feature_keys: list[str]


@router.put("/{license_id}/features")
def set_license_features(license_id: str, payload: FeatureKeysIn, _: dict = Depends(auth.require_admin)):
    if license_db.get_license(license_id) is None:
        raise HTTPException(status_code=404, detail="License not found")
    invalid = [k for k in payload.feature_keys if k not in license_db.ALL_FEATURES]
    if invalid:
        raise HTTPException(status_code=422, detail=f"Unknown feature key(s): {invalid}")
    keys = license_db.set_license_features(license_id, payload.feature_keys)
    return {"feature_keys": keys}


@router.put("/{license_id}/cameras/{camera_id}/features")
def set_camera_features(license_id: str, camera_id: int, payload: FeatureKeysIn, _: dict = Depends(auth.require_admin)):
    if license_db.get_license(license_id) is None:
        raise HTTPException(status_code=404, detail="License not found")
    if not license_db.is_camera_assigned(license_id, camera_id):
        raise HTTPException(status_code=422, detail="Camera is not assigned to this license")
    invalid = license_db.set_camera_features(license_id, camera_id, payload.feature_keys)
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Feature(s) not enabled on this license, enable them there first: {invalid}",
        )
    return {"feature_keys": license_db.get_camera_features(camera_id)}
