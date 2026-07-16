from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_ROOT = Path(
    "data/raw/"
    "brain-tumor-mri-multi-class-dataset/"
    "multi_class_dataset"
)
OUTPUT_PATH = Path("data/splits/splits.csv")

CLASS_TO_IDX = {
    "glioma": 0,
    "healthy": 1,
    "meningioma": 2,
    "pituitary": 3,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SEED = 42


def main() -> None:
    rows = []
                            
    # 클래스 폴더에서 이미지 경로와 라벨 수집
    for class_name, label in CLASS_TO_IDX.items():
        class_dir = DATA_ROOT / class_name

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(
                    {
                        "filepath": image_path.relative_to(DATA_ROOT).as_posix(),
                        "class_name": class_name,
                        "label": label,
                    }
                )

    samples = pd.DataFrame(rows)

    # 전체의 70%를 train으로 분리
    train_df, remaining_df = train_test_split(
        samples,
        test_size=0.30,
        random_state=SEED,
        stratify=samples["label"], # 각 클래스 비율을 유지하면서 분리
    )

    # 남은 30%를 validation 15%, test 15%로 분리
    validation_df, test_df = train_test_split(
        remaining_df,
        test_size=0.50,
        random_state=SEED,
        stratify=remaining_df["label"],
    )

    train_df = train_df.assign(split="train")
    validation_df = validation_df.assign(split="validation")
    test_df = test_df.assign(split="test")

    splits = pd.concat(
        [train_df, validation_df, test_df],
        ignore_index=True,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(OUTPUT_PATH, index=False)

    print(pd.crosstab(splits["split"], splits["class_name"], margins=True))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()