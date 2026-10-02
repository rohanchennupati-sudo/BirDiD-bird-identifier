import uuid
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.middleware.rate_limiter import PREDICT_RATE_LIMIT, limiter
from app.models.schemas import BirdPrediction, ConfidenceLevel, PredictionResponse
from app.services.mock_classifier import MockBirdClassifier
from app.services.pytorch_classifier import InvalidImageError

router = APIRouter(tags=["predict"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

_mock_classifier = MockBirdClassifier()


def _choose_classifier(backend: str, request: Request, settings: Settings):
    """Return (classifier, backend_used) for the requested backend."""
    real = getattr(request.app.state, "classifier", None)  # loaded once at startup

    if backend == "mock":
        return _mock_classifier, "mock"

    if real is not None:
        return real, "pytorch"

    # No trained model available.
    if backend == "pytorch" or settings.is_production:
        raise HTTPException(
            status_code=503,
            detail="Trained model checkpoint is not available.",
        )
    return _mock_classifier, "mock"


@router.post("/predict", response_model=PredictionResponse)
@limiter.limit(PREDICT_RATE_LIMIT)
async def predict_species(
    request: Request,
    image: UploadFile = File(...),
    backend: Literal["auto", "pytorch", "mock"] = Form("auto"),
    settings: Settings = Depends(get_settings),
) -> PredictionResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use JPEG, PNG, or WebP.",
        )

    max_bytes = settings.max_image_size_mb * 1024 * 1024
    too_large = HTTPException(
        status_code=413,
        detail=f"File too large. Max {settings.max_image_size_mb} MB.",
    )
    if image.size is not None and image.size > max_bytes:
        raise too_large

    image_bytes = await image.read(max_bytes + 1)
    if len(image_bytes) > max_bytes:
        raise too_large

    classifier, backend_used = _choose_classifier(backend, request, settings)

    # Inference is CPU-bound; run it in a worker thread so the event loop
    # keeps serving other requests.
    try:
        result = await run_in_threadpool(
            classifier.predict, image_bytes, image.content_type
        )
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResponse(
        success=True,
        request_id=request_id,
        backend_used=backend_used,
        data=BirdPrediction(
            common_name=result["common_name"],
            scientific_name=result.get("scientific_name", ""),
            confidence=ConfidenceLevel(result["confidence"]),
            probability=result["probability"],
            top5=result.get("top5", []),
            not_a_bird=result.get("not_a_bird", False),
        ),
    )
