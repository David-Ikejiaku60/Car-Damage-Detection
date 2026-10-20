
import torch
import pandas as pd
from pathlib import Path
import json
import csv
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from dentdetect.prepare import DataCleaning
from dentdetect.image_transform import get_mean_and_std, ImageTransformation


data_dir: Path = Path("car_damage_dataset/CarDD_COCO")
annotations_dir: Path = Path(data_dir, "annotations")

train_img_dir: Path = Path(data_dir, "train2017")
val_img_dir: Path = Path(data_dir, "val2017")
test_img_dir: Path = Path(data_dir, "test2017")

train_annotation_dir: Path = Path(annotations_dir, "instances_train2017.json")
val_annotation_dir: Path = Path(annotations_dir, "instances_val2017.json")
test_annotation_dir: Path = Path(annotations_dir, "instances_test2017.json")

train_label: Path = Path(data_dir, "train_label.csv")
val_label: Path = Path(data_dir, "val_label.csv")
test_label: Path = Path(data_dir, "test_label.csv")


clean_train: DataCleaning = DataCleaning(train_annotation_dir)
clean_train.save_data(train_label)

clean_val: DataCleaning = DataCleaning(val_annotation_dir)
clean_val.save_data(val_label)

clean_test: DataCleaning = DataCleaning(test_annotation_dir)
clean_test.save_data(test_label)

train_df: pd.DataFrame = pd.read_csv(train_label)
val_df: pd.DataFrame = pd.read_csv(val_label)
test_df: pd.DataFrame = pd.read_csv(test_label)


transform_image: transforms.Compose = ImageTransformation().transform(
    128, "mean-std"
)

train_mean: torch.Tensor
train_std: torch.Tensor
train_mean, train_std = get_mean_and_std(transform_image, train_img_dir)


class CarDamageDataset(Dataset):
    """Dataset for loading car images and their multi-label damage targets."""

    def __init__(self, transform: transforms.Compose, subset: str) -> None:
        """
        Initialize the car damage dataset.

        Args:
            transform: Image transformations applied when loading each image.
            subset: Dataset split to use: 'train', 'val', or 'test'.

        Raises:
            ValueError: If subset is not a supported dataset split.
        """
        self.transform: transforms.Compose = transform

        if subset not in ["train", "val", "test"]:
            raise ValueError("subset must be 'train', 'val', or 'test'")

        self.subset: str = subset
        self.df_dict: dict[str, pd.DataFrame] = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }

        self.dir_dict: dict[str, Path] = {
            "train": train_img_dir,
            "val": val_img_dir,
            "test": test_img_dir,
        }

    def __len__(self) -> int:
        """
        Return the number of image samples in the selected subset.

        Returns:
            Number of rows in the subset's label DataFrame.
        """
        return self.df_dict[self.subset].shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load an image and its corresponding multi-label damage targets.

        Args:
            idx: Zero-based positional index of the sample.

        Returns:
            A tuple containing:
                - Image tensor transformed according to the supplied pipeline.
                - Float32 tensor containing the six binary damage labels.

        Raises:
            IndexError: If idx is outside the valid dataset range.
            FileNotFoundError: If the image file does not exist.
        """
        df: pd.DataFrame = self.df_dict[self.subset]
        image_dir: Path = self.dir_dict[self.subset]

        if idx < 0 or idx >= len(df):
            raise IndexError("Invalid dataset index")

        row: pd.Series = df.iloc[idx]
        image_path: Path = image_dir / row["file_name"]

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor: torch.Tensor = self.transform(image)

        label: torch.Tensor = torch.tensor(
            row.drop("file_name").to_numpy(dtype="float32"),
            dtype=torch.float32,
        )

        return image_tensor, label
