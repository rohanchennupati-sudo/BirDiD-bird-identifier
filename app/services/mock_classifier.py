class MockBirdClassifier:
    """Deterministic classifier used until the trained checkpoint exists."""

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
