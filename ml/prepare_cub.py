import shutil
from pathlib import Path


def prepare_cub(cub_root: str, output_dir: str) -> None:
    cub = Path(cub_root)
    out = Path(output_dir)

    id_to_path = {}
    id_to_split = {}

    with open(cub / "images.txt", encoding="utf-8") as f:
        for line in f:
            img_id, path = line.strip().split()
            id_to_path[img_id] = path

    with open(cub / "train_test_split.txt", encoding="utf-8") as f:
        for line in f:
            img_id, is_train = line.strip().split()
            id_to_split[img_id] = "train" if is_train == "1" else "test"

    for img_id, rel_path in id_to_path.items():
        split = id_to_split[img_id]
        class_name = rel_path.split("/")[0].split(".", 1)[1]
        filename = rel_path.split("/")[1]

        dest = out / split / class_name
        dest.mkdir(parents=True, exist_ok=True)

        shutil.copy2(
            cub / "images" / rel_path,
            dest / filename,
        )

    print(f"Done. Output: {out}")


if __name__ == "__main__":
    prepare_cub("CUB_200_2011/", "data/birds/")
