from pathlib import Path

from src.dataset import create_dataloaders

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
    dataloaders = create_dataloaders(
        data_root=DATA_ROOT,
        splits_csv=SPLITS_CSV,
        image_size=224,
        batch_size=32,
        num_workers=0,
    )

    images, labels = next(iter(dataloaders["train"]))
    '''
    train_loader = dataloaders["train"]
    train_iterator = iter(train_loader)
    batch = next(train_iterator)

    images = batch[0]
    labels = batch[1]
    '''

    print(f"Images shape: {images.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")

if __name__ == "__main__":
    main()