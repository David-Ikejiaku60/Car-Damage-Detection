
"""Create separate DataLoaders for DentDetect dataset splits."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dentdetect.dataset import CarDamageDataset
from dentdetect.image_transform import ImageTransformation


@dataclass
class CreateDataloader:
    """Configure and create training, validation, and test DataLoaders."""

    img_size: int | tuple[int, int]
    mean: torch.Tensor
    std: torch.Tensor
    batch_size: int
    num_workers: int = 0

    def __post_init__(self) -> None:
        """Validate the shared DataLoader configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")

        if self.mean.shape != (3,) or self.std.shape != (3,):
            raise ValueError("mean and std must each contain 3 RGB values")

        if torch.any(self.std <= 0):
            raise ValueError("std values must be greater than zero")

        self.image_transformer = ImageTransformation()

    def create_train_dataloader(
        self,
        rotation_angle: int | list[int],
        crop_size: int | tuple[int, int] | None = None,
    ) -> DataLoader:
        """
        Create the training DataLoader.

        Args:
            crop_size: Optional center-crop size. An integer specifies
                a square crop; a tuple specifies (height, width).
            rotation_angle: Maximum rotation angle in degrees, or a
                two-element list specifying the rotation range.

        Returns:
            DataLoader: Shuffled training batches with augmentation.
        """
        train_transform: transforms.Compose = (
            self.image_transformer.transform(
                resize=self.img_size,
                transform_type="train",
                mean=self.mean,
                std=self.std,
                crop_size=crop_size,
                rotation_angle=rotation_angle,
            )
        )

        train_dataset: CarDamageDataset = CarDamageDataset(
            transform=train_transform,
            subset="train",
        )

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def create_val_dataloader(self) -> DataLoader:
        """
        Create the validation DataLoader.

        Returns:
            DataLoader: Validation batches without training augmentation.
        """
        val_transform: transforms.Compose = (
            self.image_transformer.transform(
                resize=self.img_size,
                transform_type="val",
                mean=self.mean,
                std=self.std,
            )
        )

        val_dataset: CarDamageDataset = CarDamageDataset(
            transform=val_transform,
            subset="val",
        )

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def create_test_dataloader(self) -> DataLoader:
        """
        Create the test DataLoader.

        Returns:
            DataLoader: Test batches without training augmentation.
        """
        test_transform: transforms.Compose = (
            self.image_transformer.transform(
                resize=self.img_size,
                transform_type="test",
                mean=self.mean,
                std=self.std,
            )
        )

        test_dataset: CarDamageDataset = CarDamageDataset(
            transform=test_transform,
            subset="test",
        )

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
