import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize(320),
    transforms.RandomCrop(300),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.05,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_fraction: float = 0.15,
    num_workers: int = 4,
):
    data_dir = str(data_dir).rstrip("/\\")

    train_full = datasets.ImageFolder(
        f"{data_dir}/train",
        transform=TRAIN_TRANSFORMS,
    )
    test_set = datasets.ImageFolder(
        f"{data_dir}/test",
        transform=EVAL_TRANSFORMS,
    )

    class_names = train_full.classes

    n_val = int(len(train_full) * val_fraction)
    n_train = len(train_full) - n_val

    g = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(
        train_full,
        [n_train, n_val],
        generator=g,
    )

    # random_split produces Subsets over the same ImageFolder.
    # Use deterministic evaluation transforms for validation.
    val_set.dataset.transform = EVAL_TRANSFORMS

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(
        f"Classes: {len(class_names)} | "
        f"Train: {n_train} | Val: {n_val} | Test: {len(test_set)}"
    )

    return train_loader, val_loader, test_loader, class_names
