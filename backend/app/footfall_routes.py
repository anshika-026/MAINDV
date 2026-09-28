"""
footfall_routes.py

Read-only unique-footfall reporting across all entry gates (see footfall.py).

Mount with: app.include_router(footfall_routes.router) in main.py
"""

from fastapi import APIRouter, Depends

from app import auth
from app.footfall import service

router = APIRouter(prefix="/api/footfall", tags=["footfall"])


@router.get("/summary")
def footfall_summary(_: dict = Depends(auth.require_admin)):
    """Today's unique people across every gate (each person once, whichever
    gate(s) they used), plus the per-gate breakdown, hourly arrivals vs
    yesterday, and today's visitor list."""
    return service.summary()
