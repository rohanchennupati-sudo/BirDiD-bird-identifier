from fastapi import APIRouter

from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health/")
async def health_check():
    settings = get_settings()
    return {
        "status": "healthy",
        "environment": settings.environment,
        "model_available": __import__(
            "pathlib"
        ).Path(settings.model_checkpoint_path).exists(),
    }
