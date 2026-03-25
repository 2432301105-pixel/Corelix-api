from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    RiskBatchProfileRequest,
    RiskBatchProfilesResponse,
    RiskDistribution,
    RiskProfileRequest,
    RiskProfileResponse,
)
from app.services.analysis_service import run_risk_profile, run_risk_profiles

router = APIRouter(prefix="/risk", tags=["Risk"])


@router.get("/profile", response_model=RiskProfileResponse)
async def profile(
    symbol: str = Query(..., min_length=1, max_length=24),
    lookback: int = Query(default=252, ge=60, le=1260),
    interval: str = Query(default="1d", min_length=2, max_length=8),
    distribution: RiskDistribution = Query(default=RiskDistribution.student_t),
    refresh: str = Query(default="if_stale", min_length=2, max_length=20),
):
    try:
        request = RiskProfileRequest(
            symbol=symbol,
            lookback=lookback,
            interval=interval,
            distribution=distribution,
            refresh=refresh,
        )
        return run_risk_profile(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "RISK_PROFILE_ERROR", "message": str(exc)},
        ) from exc


@router.post("/profiles", response_model=RiskBatchProfilesResponse)
async def profiles(req: RiskBatchProfileRequest):
    try:
        return run_risk_profiles(req)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "RISK_BATCH_ERROR", "message": str(exc)},
        ) from exc
