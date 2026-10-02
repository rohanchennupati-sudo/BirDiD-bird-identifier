from PIL import Image

from ml.dataset import EVAL_TRANSFORMS, TRAIN_TRANSFORMS, create_dataloaders


def _make_image_folder(root, classes=("a", "b"), per_class=10):
    for split in ("train", "test"):
        for cls in classes:
            folder = root / split / cls
            folder.mkdir(parents=True)
            for i in range(per_class):
                Image.new("RGB", (40, 40), color=(i * 20, 0, 0)).save(folder / f"{i}.jpg")


def test_train_split_is_augmented_and_val_split_is_not(tmp_path):
    _make_image_folder(tmp_path)
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        str(tmp_path), batch_size=4, val_fraction=0.2, num_workers=0
    )

    assert class_names == ["a", "b"]
    assert train_loader.dataset.dataset.transform is TRAIN_TRANSFORMS
    assert val_loader.dataset.dataset.transform is EVAL_TRANSFORMS
    assert test_loader.dataset.transform is EVAL_TRANSFORMS

    train_idx = set(train_loader.dataset.indices)
    val_idx = set(val_loader.dataset.indices)
    assert len(train_idx) == 16 and len(val_idx) == 4
    assert not train_idx & val_idx
