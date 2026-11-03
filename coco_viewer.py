from pathlib import Path

import fiftyone as fo
import yaml

YAML_PATH = Path(r"D:")

# Какие части датасета показывать
SPLITS = ["train", "val", "test"]


def main() -> None:
    """Загружает YOLO-датасет и открывает локальный интерфейс FiftyOne."""
    config = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))

    dataset = fo.Dataset(name="yolo_dataset")

    for split in SPLITS:
        if split not in config:
            continue

        dataset.add_dir(
            dataset_dir=str(YAML_PATH.parent),
            dataset_type=fo.types.YOLOv5Dataset,
            yaml_path=str(YAML_PATH),
            split=split,
            tags=[split],
            drop_existing_samples=True,
        )

    session = fo.launch_app(dataset)

    # Не закрывать приложение, пока пользователь его не закроет
    session.wait()


if __name__ == "__main__":
    main()
