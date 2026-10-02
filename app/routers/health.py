from pathlib import Path

from fastapi import APIRouter, Request

from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health/")
async def health_check(request: Request):
    settings = get_settings()
    classifier = getattr(request.app.state, "classifier", None)
    return {
        "status": "healthy",
        "environment": settings.environment,
        "model_available": Path(settings.model_checkpoint_path).exists(),
        "model_loaded": classifier is not None and classifier.is_loaded,
    }
