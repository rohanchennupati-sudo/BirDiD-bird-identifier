import uuid
from fastapi import APIRouter, File, UploadFile, Request, Depends, HTTPException
from app.config import Settings, get_settings
from app.models.schemas import PredictionResponse, BirdPrediction, ConfidenceLevel
from app.middleware.rate_limiter import limiter
from app.services.mock_classifier import MockBirdClassifier
router = APIRouter(tags=["predict"])
# Phase 1: use mock. Chapter 10 replaces this with PyTorchBirdClassifier.
classifier = MockBirdClassifier()
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
@router.post("/predict", response_model=PredictionResponse)
@limiter.limit("20/minute")
async def predict_species(
    request: Request,
    image: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> PredictionResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    # Validate file size
    image_bytes = await image.read()
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(413, f"File too large. Max {settings.max_image_size_mb} MB.")
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
