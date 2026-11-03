
from pathlib import Path

import fiftyone as fo

DATASET_DIR = Path(r"D:\dataset")

# Какие части датасета показывать
SPLITS = {
    "train": DATASET_DIR / "train",
    "val": DATASET_DIR / "valid",
    "test": DATASET_DIR / "test",
}

# Принятое название (создаётся в том числе в Roboflow) для файла с аннотациями COCO
ANNOTATION_FILE = "_annotations.coco.json"

def main() -> None:
    """Загружает COCO-датасет и открывает локальный интерфейс FiftyOne."""
    dataset = fo.Dataset("coco_dataset")

    for split, images_dir in SPLITS.items():
        annotations_path = images_dir / ANNOTATION_FILE

        if not images_dir.exists() or not annotations_path.exists():
            print(f"Пропускаю {split}: папка или JSON не найдены")
            continue

        dataset.add_dir(
            dataset_dir=str(images_dir),
            dataset_type=fo.types.COCODetectionDataset,
            data_path=".",
            labels_path=str(annotations_path),
            tags=[split],
        )

    if len(dataset) == 0:
        raise RuntimeError("Датасет пустой. Проверь пути и JSON.")

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
