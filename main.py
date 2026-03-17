from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api import health, assets, analysis
from app.core.config import settings

def create_app():
    app = FastAPI(title=settings.app_name, version=settings.app_version, docs_url="/docs")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request, exc):
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(x) for x in first.get("loc", [])) or None
        return JSONResponse(status_code=422, content={"error": {"code": "VALIDATION_ERROR", "message": first.get("msg", "Invalid request."), "field": field}})

    @app.exception_handler(Exception)
    async def generic_handler(request, exc):
        return JSONResponse(status_code=500, content={"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred."}})

    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(assets.router, prefix=settings.api_prefix)
    app.include_router(analysis.router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": settings.app_name, "version": settings.app_version, "docs": "/docs"}

    return app

app = create_app()
