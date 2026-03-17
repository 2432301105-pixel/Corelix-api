from fastapi import APIRouter
from app.core.config import settings
from app.models.schemas import HealthResponse
router = APIRouter(tags=["Health"])
@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version=settings.app_version, environment=settings.app_env)
