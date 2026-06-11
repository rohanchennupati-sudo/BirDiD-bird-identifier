from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
class BirdPrediction(BaseModel):
    common_name: str = Field(description="Common English name, e.g. Atlantic Puffin")
    scientific_name: str = Field(description="Binomial scientific name, e.g. Fratercula arctica")
    confidence: ConfidenceLevel = Field(description="Confidence level of the prediction")
    probability: float = Field(ge=0.0,le=100.0,description="Top-1 probability %")
    top5: list = Field(default_factory=list,description="Top-5 candidates")
    not_a_bird: bool = Field(default=False)
class PredictionResponse(BaseModel):
    success: bool
    request_id: str
    data: Optional[BirdPrediction] = None
    error: Optional[str] = None