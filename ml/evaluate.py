import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ml.dataset import create_dataloaders
from ml.model import build_model


@torch.no_grad()
def evaluate_model(
    checkpoint_path: str,
    data_dir: str,
    batch_size: int = 32,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    class_names = checkpoint["class_names"]

    model = build_model(
        num_classes=len(class_names),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    _, _, test_loader, _ = create_dataloaders(
        data_dir,
        batch_size=batch_size,
    )

    all_labels = []
    all_preds = []
    all_top5 = []

    for images, labels in test_loader:
        probs = torch.softmax(model(images.to(device)), dim=1)
        top5 = probs.topk(5, dim=1).indices

        all_labels.extend(labels.numpy())
        all_preds.extend(probs.argmax(dim=1).cpu().numpy())
        all_top5.extend(top5.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    top1_acc = (all_preds == all_labels).mean()
    top5_acc = sum(
        label in top5
        for label, top5 in zip(all_labels, all_top5)
    ) / len(all_labels)

    correct = defaultdict(int)
    total = defaultdict(int)

    for target, pred in zip(all_labels, all_preds):
        total[target] += 1
        if target == pred:
            correct[target] += 1

    per_class = {
        class_names[i]: correct[i] / total[i]
        for i in range(len(class_names))
        if total[i] > 0
    }

    sorted_classes = sorted(
        per_class.items(),
        key=lambda x: x[1],
    )

    confusion = defaultdict(int)

    for target, pred in zip(all_labels, all_preds):
        if target != pred:
            confusion[
                (class_names[target], class_names[pred])
            ] += 1

    top_errors = sorted(
        confusion.items(),
        key=lambda x: -x[1],
    )[:10]

    print("=" * 60)
    print(f"Top-1 Accuracy: {top1_acc * 100:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc * 100:.2f}%")
    print(f"Test images: {len(all_labels)}")
    print("\nWorst 5 classes:")

    for name, acc in sorted_classes[:5]:
        print(f"  {acc * 100:5.1f}% {name}")

    print("\nTop confusions:")

    for (target, pred), count in top_errors[:5]:
        print(f"  {count}x {target} -> {pred}")

    print("=" * 60)

    results = {
        "top1_accuracy": round(top1_acc * 100, 2),
        "top5_accuracy": round(top5_acc * 100, 2),
        "num_test_images": len(all_labels),
        "worst_5": [
            (name, round(acc * 100, 1))
            for name, acc in sorted_classes[:5]
        ],
        "top_confusions": [
            (f"{target}->{pred}", count)
            for (target, pred), count in top_errors
        ],
    }

    output_path = (
        Path(checkpoint_path).parent / "evaluation_results.json"
    )
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"Saved to {output_path}")
    return results


if __name__ == "__main__":
    evaluate_model(
        "models/best_model_finetuned.pth",
        "data/birds/",
    )
