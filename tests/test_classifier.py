from app.services.mock_classifier import MockBirdClassifier


def test_mock_classifier_returns_expected_structure():
    classifier = MockBirdClassifier()
    result = classifier.predict(b"fake_bytes", "image/jpeg")

    assert "common_name" in result
    assert "confidence" in result
    assert "probability" in result
    assert "top5" in result
    assert len(result["top5"]) == 5
    assert (
        result["top5"][0]["probability"]
        >= result["top5"][1]["probability"]
    )


def test_mock_classifier_is_available():
    assert MockBirdClassifier().is_available is True
