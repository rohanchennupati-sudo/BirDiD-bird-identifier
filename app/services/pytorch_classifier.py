import io
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class PyTorchBirdClassifier:
    """Inference wrapper for the trained EfficientNet-B3 model."""

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

    def _load(self) -> None:
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location=self._device,
            weights_only=False,
        )
        self._class_names = checkpoint["class_names"]

        from ml.model import build_model

        self._model = build_model(
            num_classes=len(self._class_names),
            pretrained=False,
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model = self._model.to(self._device).eval()

    @torch.no_grad()
    def predict(self, image_bytes: bytes, content_type: str) -> dict:
        self._load()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = INFERENCE_TRANSFORMS(image)
        tensor = tensor.unsqueeze(0).to(self._device)

        probs = torch.softmax(self._model(tensor), dim=1)[0]

        top5_probs, top5_idx = probs.topk(5)

        top5 = [
            {
                "species": self._class_names[i.item()].replace("_", " "),
                "probability": round(p.item() * 100, 1),
            }
            for p, i in zip(top5_probs, top5_idx)
        ]

        top1_prob = probs.max().item()
        confidence = (
            "High"
            if top1_prob >= self.HIGH_THRESHOLD
            else "Medium"
            if top1_prob >= self.MEDIUM_THRESHOLD
            else "Low"
        )

        return {
            "common_name": self._class_names[probs.argmax().item()].replace("_", " "),
            "scientific_name": "",
            "confidence": confidence,
            "probability": round(top1_prob * 100, 1),
            "top5": top5,
            "not_a_bird": False,
        }
