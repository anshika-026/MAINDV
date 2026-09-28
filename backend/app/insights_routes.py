"""
insights_routes.py

Dashboard and Workforce Insights summaries (see insights.py).

Mount with: app.include_router(insights_routes.router) in main.py
"""

import datetime

from fastapi import APIRouter, Depends, HTTPException

from app import auth, insights

router = APIRouter(prefix="/api", tags=["insights"])


@router.get("/dashboard/summary")
def dashboard_summary(_: dict = Depends(auth.require_admin)):
    return insights.dashboard_summary()


@router.get("/workforce/overview")
def workforce_overview(date: str | None = None, _: dict = Depends(auth.require_admin)):
    if date:
        try:
            datetime.date.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=422, detail="Dates must look like 2026-09-28")
    return insights.workforce_overview(date)


@router.get("/cameras/health")
def cameras_health(_: dict = Depends(auth.require_admin)):
    return insights.camera_health()
