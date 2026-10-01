import uuid
from typing import Literal
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.config import Settings, get_settings
from app.middleware.rate_limiter import limiter
from app.models.schemas import BirdPrediction, ConfidenceLevel, PredictionResponse
from app.services.mock_classifier import MockBirdClassifier
from app.services.pytorch_classifier import PyTorchBirdClassifier

router = APIRouter(tags=["predict"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

_mock_classifier = MockBirdClassifier()


def _get_classifier(settings: Settings):
    real = PyTorchBirdClassifier(settings.model_checkpoint_path)
    if real.is_available:
        return real
    return _mock_classifier


@router.post("/predict", response_model=PredictionResponse)
@limiter.limit("20/minute")
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

    image_bytes = await image.read()
    max_bytes = settings.max_image_size_mb * 1024 * 1024

    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.max_image_size_mb} MB.",
        )

    if backend == "mock":
        result = _mock_classifier.predict(image_bytes, image.content_type)

    elif backend == "pytorch":
        real_classifier = PyTorchBirdClassifier(settings.model_checkpoint_path)

        if not real_classifier.is_available:
            raise HTTPException(
                status_code=503,
                detail="Trained model checkpoint is not available yet.",
            )

        result = real_classifier.predict(image_bytes, image.content_type)

    else:
        classifier = _get_classifier(settings)
        result = classifier.predict(image_bytes, image.content_type)

    return PredictionResponse(
        success=True,
        request_id=request_id,
        data=BirdPrediction(
            common_name=result["common_name"],
            scientific_name=result.get("scientific_name", ""),
            confidence=ConfidenceLevel(result["confidence"]),
            probability=result["probability"],
            top5=result.get("top5", []),
            not_a_bird=result.get("not_a_bird", False),
        ),
    )
