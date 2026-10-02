from pathlib import Path

import pytest

from app.services.mock_classifier import MockBirdClassifier
from app.services.pytorch_classifier import InvalidImageError, PyTorchBirdClassifier

CHECKPOINT = Path("models/best_model_finetuned.pth")


def test_mock_classifier_returns_expected_structure(jpeg_bytes):
    result = MockBirdClassifier().predict(jpeg_bytes, "image/jpeg")
    assert {"common_name", "confidence", "probability", "top5"} <= result.keys()
    assert len(result["top5"]) == 5
    assert result["top5"][0]["probability"] >= result["top5"][1]["probability"]


def test_mock_classifier_is_available():
    assert MockBirdClassifier().is_available is True


def test_mock_classifier_rejects_invalid_image():
    with pytest.raises(InvalidImageError):
        MockBirdClassifier().predict(b"not an image", "image/jpeg")


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="trained checkpoint not downloaded")
def test_pytorch_classifier_real_prediction(jpeg_bytes):
    classifier = PyTorchBirdClassifier(str(CHECKPOINT))
    classifier.load()
    result = classifier.predict(jpeg_bytes, "image/jpeg")

    assert classifier.is_loaded
    assert len(result["top5"]) == 5
    probs = [p["probability"] for p in result["top5"]]
    assert probs == sorted(probs, reverse=True)
    assert result["common_name"] == result["top5"][0]["species"]
    assert result["confidence"] in {"High", "Medium", "Low"}
