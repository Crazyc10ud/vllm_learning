from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

ISIC_TRAIN_URL = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip"
ISIC_GROUND_TRUTH_URL = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.csv"


@dataclass
class ISICSample:
    path: Path
    label: int


class ISIC2018Dataset(Dataset):
    """Classification dataset for the ISIC 2018 skin lesion challenge."""

    def __init__(self, root: str, transform: Callable | None = None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples = self._load_samples()

    def _ensure_downloaded(self) -> None:
        data_dir = self.root
        data_dir.mkdir(parents=True, exist_ok=True)
        images_zip = data_dir / "ISIC2018_Task3_Training_Input.zip"
        labels_csv = data_dir / "ISIC2018_Task3_Training_GroundTruth.csv"

        if not images_zip.exists():
            self._download_file(ISIC_TRAIN_URL, images_zip)
        if not labels_csv.exists():
            self._download_file(ISIC_GROUND_TRUTH_URL, labels_csv)

        extracted_dir = data_dir / "ISIC2018_Task3_Training_Input"
        if not extracted_dir.exists():
            with zipfile.ZipFile(images_zip, "r") as zf:
                zf.extractall(data_dir)

    def _download_file(self, url: str, destination: Path) -> None:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def _load_samples(self) -> List[ISICSample]:
        self._ensure_downloaded()
        data_dir = self.root
        images_dir = data_dir / "ISIC2018_Task3_Training_Input"
        labels_path = data_dir / "ISIC2018_Task3_Training_GroundTruth.csv"

        label_map = {}
        with open(labels_path, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            classes = [field for field in reader.fieldnames if field != "image"]
            for row in reader:
                image_id = row["image"]
                for idx, class_name in enumerate(classes):
                    if float(row[class_name]) == 1.0:
                        label_map[image_id] = idx
                        break

        samples = []
        for image_path in images_dir.glob("*.jpg"):
            image_id = image_path.stem
            if image_id in label_map:
                samples.append(ISICSample(path=image_path, label=label_map[image_id]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label

    @property
    def num_classes(self) -> int:
        return 7


def build_isic_dataloaders(
    image_processor: Callable,
    batch_size: int,
    num_workers: int = 4,
    val_split: float = 0.1,
    test_split: float = 0.1,
    root: str = "data/isic2018",
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    dataset = ISIC2018Dataset(root=root, transform=image_processor)
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int((total_size - test_size) * val_split)
    train_size = total_size - val_size - test_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    num_classes = dataset.num_classes
    return train_loader, val_loader, test_loader, num_classes
