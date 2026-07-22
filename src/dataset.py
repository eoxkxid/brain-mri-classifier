from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BrainMRIDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        splits_csv: str | Path,
        split: str,
        transform: Callable | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.transform = transform

        splits = pd.read_csv(splits_csv)

        self.samples = (
            splits[splits["split"] == split]
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self.samples.iloc[index]

        image_path = self.data_root / row["filepath"]
        label = int(row["label"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        # image: [channels=3, height, width]
        return image, label


def build_transforms(
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    return train_transform, evaluation_transform


def create_dataloaders(
    data_root: str | Path,
    splits_csv: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    train_transform, evaluation_transform = build_transforms(
        image_size=image_size,
    )

    train_dataset = BrainMRIDataset(
        data_root=data_root,
        splits_csv=splits_csv,
        split="train",
        transform=train_transform,
    )

    validation_dataset = BrainMRIDataset(
        data_root=data_root,
        splits_csv=splits_csv,
        split="validation",
        transform=evaluation_transform,
    )

    test_dataset = BrainMRIDataset(
        data_root=data_root,
        splits_csv=splits_csv,
        split="test",
        transform=evaluation_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader,
    }