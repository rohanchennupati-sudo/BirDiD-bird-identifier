import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def make_jpeg_bytes() -> bytes:
    image = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health():
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_success():
    response = client.post(
        "/api/v1/predict",
        files={
            "image": (
                "test.jpg",
                make_jpeg_bytes(),
                "image/jpeg",
            )
        },
        data={"backend": "mock"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["data"]["common_name"] == "Atlantic Puffin"


def test_predict_rejects_unsupported_type():
    response = client.post(
        "/api/v1/predict",
        files={
            "image": (
                "test.txt",
                b"not an image",
                "text/plain",
            )
        },
        data={"backend": "mock"},
    )

    assert response.status_code == 400
