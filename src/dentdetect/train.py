"""
Train and evaluate the DentDetect multi-label car damage classification model.

This module trains a custom CNN using PyTorch, evaluates it on a validation
dataset, records experiment results with MLflow, and saves model checkpoints.
"""

from pathlib import Path
import random

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from dentdetect.dataset import (
    CarDamageDataset,
    train_mean,
    train_std,
)
from dentdetect.dataloader import CreateDataloader
from dentdetect.image_transform import ImageTransformation
from dentdetect.models import DentDetectModel


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED: int = 42
IMAGE_SIZE: tuple[int, int] = (128, 128)
BATCH_SIZE: int = 32
EPOCHS: int = 10
LEARNING_RATE: float = 0.001
THRESHOLD: float = 0.26
NUM_CLASSES: int = 6

CLASS_NAMES: list[str] = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
]

OUTPUT_DIR: Path = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH: Path = OUTPUT_DIR / "dentdetect_best.pth"

# Set random seeds for reproducibility.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Select the available computing device.
device: torch.device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def main() -> None:
    """
    Train, validate, and log the DentDetect CNN model.

    Creates image transformations and data loaders, trains the model,
    calculates validation metrics, saves checkpoints, and records results
    using MLflow.
    """
    print(f"Using device: {device}")

    # ----------------------------------------------
    # 2. Image transformations
    # ----------------------------------------------

    image_transformer: ImageTransformation = ImageTransformation()

    train_transform = image_transformer.transform(
        resize=IMAGE_SIZE,
        transform_type="train",
        mean=train_mean,
        std=train_std,
        rotation_angle=10,
    )

    val_transform = image_transformer.transform(
        resize=IMAGE_SIZE,
        transform_type="val",
        mean=train_mean,
        std=train_std,
    )

    # ----------------------------------------------
    # 3. Create datasets and data loaders
    # ----------------------------------------------

    train_dataset: CarDamageDataset = CarDamageDataset(
        transform=train_transform,
        subset="train",
    )

    val_dataset: CarDamageDataset = CarDamageDataset(
        transform=val_transform,
        subset="val",
    )

    train_loader: DataLoader = CreateDataloader(
        img_size=IMAGE_SIZE,
        mean=train_mean,
        std=train_std,
        batch_size=BATCH_SIZE,
    ).create_train_dataloader(rotation_angle=10)

    val_loader: DataLoader = CreateDataloader(
        img_size=IMAGE_SIZE,
        mean=train_mean,
        std=train_std,
        batch_size=BATCH_SIZE,
    ).create_val_dataloader()

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    # ----------------------------------------------
    # 4. Initialize model, loss, and optimizer
    # ----------------------------------------------

    model: DentDetectModel = DentDetectModel(
        num_classes=NUM_CLASSES
    ).to(device)

    criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ----------------------------------------------
    # 5. MLflow setup
    # ----------------------------------------------

    mlflow.set_tracking_uri(
        "sqlite:///C:/Users/USER/Documents/DentDetect/"
        "src/dentdetect/mlflow.db"
    )
    mlflow.set_experiment("DentDetect")

    with mlflow.start_run(run_name="DentDetect_CNN_Training"):

        # Log the configuration used for this experiment.
        mlflow.log_params({
            "model": "DentDetectModel",
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "loss_function": "BCEWithLogitsLoss",
            "threshold": THRESHOLD,
            "num_classes": NUM_CLASSES,
            "seed": SEED,
            "device": str(device),
            "training_images": len(train_dataset),
            "validation_images": len(val_dataset),
        })

        # ------------------------------------------
        # 6. Training
        # ------------------------------------------

        for epoch in range(EPOCHS):
            model.train()

            running_loss: float = 0.0
            total_samples: int = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Clear gradients from the previous batch.
                optimizer.zero_grad()

                # Forward pass: obtain raw logits.
                outputs = model(images)

                if outputs.shape != labels.shape:
                    raise ValueError(
                        f"Output shape {outputs.shape} does not match "
                        f"label shape {labels.shape}. Check your model "
                        "and the six label columns in dataset.py."
                    )

                # Calculate loss, backpropagate, and update parameters.
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                if (batch_idx + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{EPOCHS}] "
                        f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f}"
                    )

            train_loss: float = running_loss / total_samples

            print(
                f"\nEpoch [{epoch + 1}/{EPOCHS}] "
                f"Training Loss: {train_loss:.4f}"
            )

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch + 1,
            )

            # --------------------------------------
            # 7. Validation
            # --------------------------------------

            model.eval()

            val_running_loss: float = 0.0
            val_total_samples: int = 0

            all_labels: list[np.ndarray] = []
            all_predictions: list[np.ndarray] = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = model(images)
                    loss = criterion(logits, labels)

                    val_running_loss += loss.item() * images.size(0)
                    val_total_samples += images.size(0)

                    probabilities = torch.sigmoid(logits)

                    predictions = (
                        probabilities >= THRESHOLD
                    ).int()

                    all_labels.append(labels.cpu().numpy())
                    all_predictions.append(predictions.cpu().numpy())

            val_loss: float = (
                val_running_loss / val_total_samples
            )

            y_true: np.ndarray = np.concatenate(all_labels, axis=0)
            y_pred: np.ndarray = np.concatenate(
                all_predictions,
                axis=0,
            )

            precision, recall, f1, _ = (
                precision_recall_fscore_support(
                    y_true,
                    y_pred,
                    average=None,
                    zero_division=0,
                )
            )

            (
                macro_precision,
                macro_recall,
                macro_f1,
                _,
            ) = precision_recall_fscore_support(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )

            print(f"Validation Loss: {val_loss:.4f}")
            print("\nPer-class validation results:")

            for index, class_name in enumerate(CLASS_NAMES):
                actual_positive: int = int(
                    y_true[:, index].sum()
                )
                predicted_positive: int = int(
                    y_pred[:, index].sum()
                )

                print(
                    f"{class_name:15s} | "
                    f"Actual: {actual_positive:4d} | "
                    f"Predicted: {predicted_positive:4d} | "
                    f"Precision: {precision[index]:.4f} | "
                    f"Recall: {recall[index]:.4f} | "
                    f"F1: {f1[index]:.4f}"
                )

                mlflow.log_metrics({
                    f"{class_name}_precision": float(
                        precision[index]
                    ),
                    f"{class_name}_recall": float(
                        recall[index]
                    ),
                    f"{class_name}_f1": float(
                        f1[index]
                    ),
                    f"{class_name}_actual_positive": float(
                        actual_positive
                    ),
                    f"{class_name}_predicted_positive": float(
                        predicted_positive
                    ),
                }, step=epoch + 1)

            print(f"\nMacro Precision: {macro_precision:.4f}")
            print(f"Macro Recall:    {macro_recall:.4f}")
            print(f"Macro F1:        {macro_f1:.4f}")

            mlflow.log_metrics({
                "val_loss": val_loss,
                "macro_precision": float(macro_precision),
                "macro_recall": float(macro_recall),
                "macro_f1": float(macro_f1),
            }, step=epoch + 1)

            # --------------------------------------
            # 8. Save the model checkpoint
            # --------------------------------------

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "macro_f1": float(macro_f1),
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "threshold": THRESHOLD,
                    "mean": train_mean.tolist(),
                    "std": train_std.tolist(),
                },
                MODEL_PATH,
            )

            mlflow.log_artifact(str(MODEL_PATH))

            print(f"\nModel saved to: {MODEL_PATH}")

        # ------------------------------------------
        # 9. Log the PyTorch model to MLflow
        # ------------------------------------------

        mlflow.pytorch.log_model(
            model,
            artifact_path="pytorch_model",
            serialization_format="pickle",
        )

        print("\nTraining complete.")
        print("MLflow run finished successfully.")


if __name__ == "__main__":
    main()