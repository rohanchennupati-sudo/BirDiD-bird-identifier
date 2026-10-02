from app.services.pytorch_classifier import open_image


class MockBirdClassifier:
    """Fixed-answer classifier for development and tests.

    Lets the API, UI and tests run without the trained checkpoint. It always
    returns the same prediction, so responses say backend_used="mock".
    """

    @property
    def is_available(self) -> bool:
        return True

    def predict(self, image_bytes: bytes, content_type: str) -> dict:
        open_image(image_bytes)  # reject invalid images the same way as the real model

        return {
            "common_name": "Horned Puffin",
            "scientific_name": "",
            "confidence": "High",
            "probability": 91.2,
            "top5": [
                {"species": "Horned Puffin", "probability": 91.2},
                {"species": "Pigeon Guillemot", "probability": 5.1},
                {"species": "Rhinoceros Auklet", "probability": 2.4},
                {"species": "Parakeet Auklet", "probability": 0.8},
                {"species": "Crested Auklet", "probability": 0.5},
            ],
            "not_a_bird": False,
        }
