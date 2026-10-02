import io
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# Must match EVAL_TRANSFORMS in ml/dataset.py.
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes are not a readable image."""


def open_image(image_bytes: bytes) -> Image.Image:
    """Decode uploaded bytes into an RGB image, or raise InvalidImageError."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("File is not a valid image.") from exc
    return image.convert("RGB")


class PyTorchBirdClassifier:
    """Inference wrapper for the trained EfficientNet-B3 model.

    Create one instance at startup and call load() once; predict() then reuses
    the model for every request.
    """

    HIGH_THRESHOLD = 0.70
    MEDIUM_THRESHOLD = 0.40

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self._model: Optional[nn.Module] = None
        self._class_names: Optional[list[str]] = None
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @property
    def is_available(self) -> bool:
        return self.model_path.exists()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}"
            )

        # weights_only=False because the checkpoint also stores class names;
        # only load checkpoints you trained yourself.
        checkpoint = torch.load(
            self.model_path,
            map_location=self._device,
            weights_only=False,
        )
        self._class_names = checkpoint["class_names"]

        from ml.model import build_model

        model = build_model(
            num_classes=len(self._class_names),
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        self._model = model.to(self._device).eval()

    @torch.no_grad()
    def predict(self, image_bytes: bytes, content_type: str) -> dict:
        self.load()

        image = open_image(image_bytes)
        tensor = INFERENCE_TRANSFORMS(image).unsqueeze(0).to(self._device)

        probs = torch.softmax(self._model(tensor), dim=1)[0]
        top5_probs, top5_idx = probs.topk(5)

        top5 = [
            {
                "species": self._class_names[i.item()].replace("_", " "),
                "probability": round(p.item() * 100, 1),
            }
            for p, i in zip(top5_probs, top5_idx)
        ]

        top1_prob = top5_probs[0].item()
        if top1_prob >= self.HIGH_THRESHOLD:
            confidence = "High"
        elif top1_prob >= self.MEDIUM_THRESHOLD:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "common_name": top5[0]["species"],
            "scientific_name": "",
            "confidence": confidence,
            "probability": round(top1_prob * 100, 1),
            "top5": top5,
            "not_a_bird": False,
        }
