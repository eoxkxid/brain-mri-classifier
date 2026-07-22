from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.dataset import create_dataloaders
from src.models import BaselineCNN
from src.train import train_one_epoch, validate_one_epoch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "brain-tumor-mri-multi-class-dataset"
    / "multi_class_dataset"
)

SPLITS_CSV = PROJECT_ROOT / "data" / "splits" / "splits.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_CLASSES = 4
LEARNING_RATE = 1e-3
SEED = 42

TRAIN_SAMPLES = 128  # 4 batches
VAL_SAMPLES = 64     # 2 batches


def main() -> None:
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    full_dataloaders = create_dataloaders(
        data_root=DATA_ROOT,
        splits_csv=SPLITS_CSV,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    train_dataset = full_dataloaders["train"].dataset
    val_dataset = full_dataloaders["validation"].dataset

    train_subset = Subset(
        train_dataset,
        range(min(TRAIN_SAMPLES, len(train_dataset))),
    )

    val_subset = Subset(
        val_dataset,
        range(min(VAL_SAMPLES, len(val_dataset))),
    )

    train_dataloader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_dataloader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = BaselineCNN(
        num_classes=NUM_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    parameter_before = (
        next(model.parameters())
        .detach()
        .clone()
    )

    start_time = perf_counter()

    train_loss, train_accuracy = train_one_epoch(
        model=model,
        dataloader=train_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    val_loss, val_accuracy = validate_one_epoch(
        model=model,
        dataloader=val_dataloader,
        criterion=criterion,
        device=device,
    )

    elapsed_seconds = perf_counter() - start_time

    parameter_after = next(model.parameters()).detach()

    parameters_changed = not torch.equal(
        parameter_before,
        parameter_after,
    )

    print(f"Device: {device}")
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Parameters changed: {parameters_changed}")
    print(f"Elapsed time: {elapsed_seconds:.1f} seconds")


if __name__ == "__main__":
    main()