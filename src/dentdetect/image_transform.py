
import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
from typing import List, Literal



def get_mean_and_std(
    transform: transforms.Compose,
    image_dir: Path
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the mean and standard deviation for each RGB channel
    across all supported images in a directory.

    Args:
        transform (transforms.Compose): Transformation pipeline applied
            to each image. It should include ToTensor() but not Normalize().
        image_dir (Path): Directory containing the image files.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            A tuple containing:
                - mean: Tensor of shape (3,) with the mean values for
                  the red, green, and blue channels.
                - std: Tensor of shape (3,) with the standard deviation
                  values for the red, green, and blue channels.

    Raises:
        TypeError: If image_dir is not a Path object.
        ValueError: If no supported images are found in the directory.

    Example:
        >>> transform = transforms.Compose([transforms.ToTensor()])
        >>> mean, std = get_mean_and_std(
        ...     transform, Path("data/train")
        ... )
        >>> print(mean.shape, std.shape)
        torch.Size([3]) torch.Size([3])
    """
    if not isinstance(image_dir, Path):
        raise TypeError("image_dir must be a Path object")
    
    
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    total_pixels = 0

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = transform(image)


        channel_sum += image_tensor.sum(dim=(1, 2))
        channel_squared_sum += (image_tensor ** 2).sum(dim=(1, 2))
        total_pixels += image_tensor.shape[1] * image_tensor.shape[2]
        
    if total_pixels == 0:
        raise ValueError("No supported images found in the directory")

    mean = channel_sum / total_pixels
    variance = channel_squared_sum / total_pixels - mean ** 2
    std = torch.sqrt(torch.clamp(variance, min=0))
    

    return mean, std


class ImageTransformation:
    def __init__(self) -> None:
        """Initialize the image transformation utility."""
        ...

    def transform(
        self,
        resize: int | tuple[int, int],
        transform_type: Literal["train", "val", "test", "mean-std"],
        *,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        crop_size: int | tuple[int, int] | None = None,
        rotation_angle: int | List[int] | None = None,
    ) -> transforms.Compose:
        
        """
        Create an image transformation pipeline.

        Args:
            resize (int | tuple[int, int]): Target image size. An integer
                resizes the shorter edge while preserving the aspect ratio.
                A tuple specifies the target (height, width).

            crop_size (int | tuple[int, int] | None): Size of the image crop.
                An integer specifies a square crop, while a tuple specifies
                the target (height, width). Defaults to None, meaning no
                cropping is applied.

            transform_type (Literal["train", "val", "test", "mean-std"]):
                Transformation pipeline to create:
                - "train": Resizes, randomly flips and rotates images,
                  converts them to tensors, and normalizes RGB channels.
                - "val": Resizes, converts images to tensors, and normalizes
                  RGB channels.
                - "test": Resizes, converts images to tensors, and normalizes
                  RGB channels.
                - "mean-std": Resizes and converts images to tensors without
                  normalization, for calculating dataset statistics.

            mean (torch.Tensor | None): Per-channel RGB mean, required for
                normalized transformations. Defaults to None.

            std (torch.Tensor | None): Per-channel RGB standard deviation,
                required for normalized transformations. Defaults to None.

            rotation_angle (int | List[int] | None): Maximum rotation angle
                in degrees, or a two-element list specifying the rotation
                range. Required when transform_type is "train".

        Returns:
            transforms.Compose: The configured image transformation pipeline.

        Raises:
            KeyError: If transform_type is not supported.
            TypeError: If required mean/std values are missing or are not
                tensors, or if rotation_angle is missing for training.
            ValueError: If mean/std do not have three RGB values, or if
                any standard deviation is not greater than zero.
        """

        if transform_type not in ["train", "val", "test", "mean-std"]:
            raise KeyError(f"Invalid transform_type: {transform_type}")

        if transform_type == "mean-std":
            return transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.ToTensor(),
                ]
            )

        if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
            raise TypeError("mean and std must be tensors for normalized transformations")

        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError("mean and std must each contain 3 RGB values")

        if torch.any(std <= 0):
            raise ValueError("std values must be greater than zero")

        if transform_type == "train":
            if rotation_angle is None:
                raise TypeError(
                    "rotation_angle must be provided for training transformations"
                )

            train_transforms = [
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2),
            ]

            if crop_size is not None:
                train_transforms.append(transforms.CenterCrop(crop_size))

            train_transforms.extend(
                [
                    transforms.RandomRotation(rotation_angle),
                    transforms.Resize(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

            return transforms.Compose(train_transforms)

        return transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
