from fastapi import APIRouter, HTTPException, Query, status
from app.models.schemas import AssetSearchResponse
from app.services.analysis_service import search_assets
router = APIRouter(prefix="/assets", tags=["Assets"])
@router.get("/search", response_model=AssetSearchResponse)
async def asset_search(q: str = Query(..., min_length=1, max_length=50), limit: int = Query(default=10, ge=1, le=50)):
    result = search_assets(q, limit=limit)
    if result.total == 0:
        raise HTTPException(status_code=404, detail={"code": "ASSET_NOT_FOUND", "message": f"No assets found matching '{q}'."})
    return result
