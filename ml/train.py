import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from ml.dataset import create_dataloaders
from ml.model import build_model, freeze_backbone, unfreeze_backbone


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        total_loss += criterion(logits, labels).item()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def _save_checkpoint(model, class_names, output_dir, filename, val_acc):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "val_accuracy": val_acc,
            "architecture": "efficientnet_b3",
            "num_classes": len(class_names),
        },
        Path(output_dir) / filename,
    )


def train(
    data_dir: str = "data/birds/",
    output_dir: str = "models/",
    batch_size: int = 32,
    num_epochs_phase1: int = 10,
    num_epochs_phase2: int = 25,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 5e-5,
    patience: int = 5,
    num_workers: int = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, class_names = create_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = build_model(num_classes=len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # Phase 1: classifier head only.
    print("\n=== PHASE 1: Training classifier head ===")
    freeze_backbone(model)

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_phase1,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs_phase1,
        eta_min=1e-5,
    )

    for epoch in range(1, num_epochs_phase1 + 1):
        t0 = time.time()

        tr_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        print(
            f"Phase 1 | Ep {epoch:2d}/{num_epochs_phase1} "
            f"tr_loss={tr_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc*100:.1f}% "
            f"({time.time()-t0:.0f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save_checkpoint(
                model,
                class_names,
                output_dir,
                "best_model_phase1.pth",
                val_acc,
            )

    # Phase 2: fine-tune the whole network.
    print("\n=== PHASE 2: Fine-tuning entire network ===")
    unfreeze_backbone(model)

    optimizer = Adam(
        model.parameters(),
        lr=lr_phase2,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs_phase2,
        eta_min=1e-7,
    )

    epochs_no_improve = 0

    for epoch in range(1, num_epochs_phase2 + 1):
        t0 = time.time()

        tr_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        print(
            f"Phase 2 | Ep {epoch:2d}/{num_epochs_phase2} "
            f"tr_loss={tr_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc*100:.1f}% "
            f"({time.time()-t0:.0f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0

            _save_checkpoint(
                model,
                class_names,
                output_dir,
                "best_model_finetuned.pth",
                val_acc,
            )
            print(f"  -> New best: {val_acc*100:.2f}%")
        else:
            epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(
                    f"Early stopping: no improvement for {patience} epochs"
                )
                break

    print(
        f"\nTraining complete. "
        f"Best validation accuracy: {best_val_acc*100:.2f}%"
    )


if __name__ == "__main__":
    train()
