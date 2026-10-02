import io

import pytest
from PIL import Image


def make_image_bytes(fmt: str = "JPEG", size=(100, 100)) -> bytes:
    image = Image.new("RGB", size, color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def jpeg_bytes() -> bytes:
    return make_image_bytes()
