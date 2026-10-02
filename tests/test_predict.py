from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import make_image_bytes

# No `with` block, so the startup hook doesn't run and no model is loaded:
# these tests exercise the API with the mock classifier.
client = TestClient(app)


def post_image(content: bytes, filename="test.jpg", content_type="image/jpeg", backend="mock"):
    return client.post(
        "/api/v1/predict",
        files={"image": (filename, content, content_type)},
        data={"backend": backend},
    )


def test_health():
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_success(jpeg_bytes):
    response = post_image(jpeg_bytes)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["backend_used"] == "mock"
    assert len(data["data"]["top5"]) == 5
    assert "X-Request-ID" in response.headers


def test_predict_accepts_png():
    response = post_image(make_image_bytes("PNG"), "test.png", "image/png")
    assert response.status_code == 200


def test_predict_rejects_unsupported_type():
    response = post_image(b"not an image", "test.txt", "text/plain")
    assert response.status_code == 400


def test_predict_rejects_corrupt_image():
    response = post_image(b"this is not really a jpeg")
    assert response.status_code == 400
    assert "valid image" in response.json()["detail"]


def test_predict_rejects_large_file():
    big = b"0" * (11 * 1024 * 1024)  # default limit is 10 MB
    response = post_image(big, "big.jpg")
    assert response.status_code == 413


def test_pytorch_backend_without_model_returns_503(jpeg_bytes):
    response = post_image(jpeg_bytes, backend="pytorch")
    assert response.status_code == 503


def test_invalid_backend_value_rejected(jpeg_bytes):
    response = post_image(jpeg_bytes, backend="gpu")
    assert response.status_code == 422
