"""
attendance_routes.py

Attendance marked from face recognition (see attendance.py).

Mount with: app.include_router(attendance_routes.router) in main.py
"""

import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app import attendance, auth

router = APIRouter(prefix="/api/attendance", tags=["attendance"])


def _valid_day(day: str) -> str:
    try:
        datetime.date.fromisoformat(day)
    except ValueError:
        raise HTTPException(status_code=422, detail="Dates must look like 2026-09-28")
    return day


@router.get("")
def attendance_for_day(date: str | None = None, _: dict = Depends(auth.require_admin)):
    """Every employee's attendance for `date` (YYYY-MM-DD, default today)."""
    return attendance.day_report(_valid_day(date or datetime.date.today().isoformat()))


@router.get("/{employee_id}/history")
def attendance_history(employee_id: str, days: int = 14, _: dict = Depends(auth.require_admin)):
    return attendance.history(employee_id, days=max(1, min(days, 90)))


class LeaveIn(BaseModel):
    employee_id: str
    day_from: str
    day_to: str
    reason: str = ""


@router.post("/leave")
def mark_leave(payload: LeaveIn, _: dict = Depends(auth.require_admin)):
    try:
        return attendance.add_leave(payload.employee_id, _valid_day(payload.day_from), _valid_day(payload.day_to), payload.reason)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
