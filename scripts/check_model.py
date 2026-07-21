from pathlib import Path

from torch import nn
import torch

from scripts.check_dataloader import PROJECT_ROOT
from src.dataset import create_dataloaders
from src.models import BaselineCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "brain-tumor-mri-multi-class-dataset"
    / "multi_class_dataset"
)

SPLITS_CSV = PROJECT_ROOT / "data" / "splits" / "splits.csv"


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"  
    )

    dataloaders = create_dataloaders(
        data_root=DATA_ROOT,
        splits_csv=SPLITS_CSV,
        image_size=224,
        batch_size=32,
        num_workers=0,
    )

    images, labels = next(iter(dataloaders["train"]))

    images = images.to(device)
    labels = labels.to(device)

    model = BaselineCNN(num_classes=4).to(device)
    model.train()

    logits = model(images)
    # logits: [batch_size, num_classes]

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)

    loss.backward()

    assert logits.shape == (images.size(0), 4)
    assert torch.isfinite(loss).item()

    first_parameter = next(model.parameters())

    print(f"Device: {device}")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"logits sjhape: {logits.shape}")
    print(f"Loss: {loss.item()}")
    print(f"Gradient computed: {first_parameter.grad is not None}")

if __name__ == "__main__":
    main()