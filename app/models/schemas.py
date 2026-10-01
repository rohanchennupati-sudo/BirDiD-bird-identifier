from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class BirdPrediction(BaseModel):
    common_name: str = Field(description="Common English bird name")
    scientific_name: str = Field(default="", description="Binomial scientific name")
    confidence: ConfidenceLevel
    probability: float = Field(ge=0.0, le=100.0)
    top5: list = Field(default_factory=list)
    not_a_bird: bool = False


class PredictionResponse(BaseModel):
    success: bool
    request_id: str
    data: Optional[BirdPrediction] = None
    error: Optional[str] = None
