# Phase 1 placeholder. Returns deterministic dummy data.
# Replace with PyTorchBirdClassifier in Chapter 10.
from app.models.schemas import BirdPrediction, ConfidenceLevel
class MockBirdClassifier:
    """
    Fake classifier that always returns "Atlantic Puffin".
    Useful for testing backend plumbing before the real model exists.
    Interface is identical to PyTorchBirdClassifier so swapping is trivial.
    """
    @property
    def is_available(self) -> bool:
        return True
    def predict(self, image_bytes: bytes, content_type: str) -> dict:
        return {
            "common_name": "Atlantic Puffin",
            "scientific_name": "Fratercula arctica",
            "confidence": "High",
            "probability": 91.2,
            "top5": [
                {"species": "Atlantic Puffin", "probability": 91.2},
                {"species": "Horned Puffin", "probability": 5.1},
                {"species": "Tufted Puffin", "probability": 2.4},
                {"species": "Rhinoceros Auklet", "probability": 0.8},
                {"species": "Common Murre", "probability": 0.5},
            ],
            "not_a_bird": False,
        }
    