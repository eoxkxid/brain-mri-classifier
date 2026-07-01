from pathlib import Path

import kagglehub

DATASET_HANDLE = "maxwellbernard/brain-tumor-mri-multi-class-dataset"
OUTPUT_DIR = Path("data/raw/brain-tumor-mri-multi-class-dataset")

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = kagglehub.dataset_download(
        DATASET_HANDLE,
        output_dir=str(OUTPUT_DIR),
    )

    print(f"Dataset hande: {DATASET_HANDLE}") 
    print(f"Dataset path: {dataset_path}")

if __name__ == "__main__":
    main()