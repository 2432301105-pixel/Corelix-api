from fastapi import APIRouter, HTTPException, status
from app.models.schemas import AnalysisRequest, AnalysisResponse
from app.services.analysis_service import run_analysis
router = APIRouter(prefix="/analysis", tags=["Analysis"])
@router.post("/compare", response_model=AnalysisResponse)
async def compare(req: AnalysisRequest):
    try:
        return run_analysis(req)
    except ValueError as exc:
        msg = str(exc)
        code = "UNKNOWN_SYMBOL" if "Unknown symbol" in msg else "INSUFFICIENT_DATA"
        raise HTTPException(status_code=400, detail={"code": code, "message": msg}) from exc
