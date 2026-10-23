
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

from dentdetect.models import DentDetectModel
from dentdetect.dataset import train_mean, train_std
from dentdetect.dataloader import CreateDataloader


# Device configuration
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Constants
CLASS_NAMES: list[str] = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
]

MODEL_PATH = Path("artifacts/dentdetect_best.pth")
METRICS_PATH = Path("artifacts/test_metrics.csv")
PREDICTIONS_PATH = Path("artifacts/test_predictions.csv")

THRESHOLD = 0.26
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 64

MLFLOW_TRACKING_URI = (
    "sqlite:///C:/Users/USER/Documents/DentDetect/"
    "src/dentdetect/mlflow.db"
)


def save_test_predictions(
    filepath: Path,
    image_names: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Save test-image labels, predictions, and probabilities to a CSV file.

    Args:
        filepath: Destination path for the predictions CSV file.
        image_names: Filenames corresponding to the test samples.
        y_true: Actual binary labels with shape (n_samples, 6).
        y_prob: Predicted probabilities with shape (n_samples, 6).
        y_pred: Predicted binary labels with shape (n_samples, 6).

    Raises:
        ValueError: If sample counts or array shapes do not match.
    """
    prediction_columns = [
        "image_name",
        "true_dent", "pred_dent", "prob_dent",
        "true_scratch", "pred_scratch", "prob_scratch",
        "true_crack", "pred_crack", "prob_crack",
        "true_glass_shatter", "pred_glass_shatter",
        "prob_glass_shatter",
        "true_lamp_broken", "pred_lamp_broken", "prob_lamp_broken",
        "true_tire_flat", "pred_tire_flat", "prob_tire_flat",
    ]

    if not (
        len(image_names)
        == len(y_true)
        == len(y_prob)
        == len(y_pred)
    ):
        raise ValueError(
            "Image names and prediction arrays must have equal sample counts."
        )

    if not (
        y_true.ndim == y_prob.ndim == y_pred.ndim == 2
        and y_true.shape == y_prob.shape == y_pred.shape
        and y_true.shape[1] == len(CLASS_NAMES)
    ):
        raise ValueError(
            "All label, probability, and prediction arrays must have "
            "shape (n_samples, 6)."
        )

    test_predictions_list = []

    for name, true, pred, prob in zip(
        image_names, y_true, y_pred, y_prob
    ):
        row = {"image_name": name}

        for idx, class_name in enumerate(CLASS_NAMES):
            column_name = class_name.replace(" ", "_")

            row[f"true_{column_name}"] = int(true[idx])
            row[f"pred_{column_name}"] = int(pred[idx])
            row[f"prob_{column_name}"] = float(prob[idx])

        test_predictions_list.append(row)

    df = pd.DataFrame(
        test_predictions_list,
        columns=prediction_columns,
    )

    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_metrics(
    filepath: Path,
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    macro_precision: float,
    macro_recall: float,
    macro_f1: float,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> None:
    """Save per-class evaluation metrics and macro averages to a CSV file.

    Args:
        filepath: Destination path for the metrics CSV file.
        precision: Precision score for each class.
        recall: Recall score for each class.
        f1: F1 score for each class.
        macro_precision: Macro-averaged precision across classes.
        macro_recall: Macro-averaged recall across classes.
        macro_f1: Macro-averaged F1 score across classes.
        actual: Number of actual positive samples per class.
        predicted: Number of predicted positive samples per class.
    """
    metrics_columns = [
        "class_name",
        "precision",
        "recall",
        "f1",
        "actual_positives",
        "predicted_positives",
    ]

    metric_list = []

    for idx, name in enumerate(CLASS_NAMES):
        metric_list.append({
            "class_name": name,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "actual_positives": int(actual[idx]),
            "predicted_positives": int(predicted[idx]),
        })

    metric_list.append({
        "class_name": "Macro average",
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1": float(macro_f1),
        "actual_positives": None,
        "predicted_positives": None,
    })

    df = pd.DataFrame(metric_list, columns=metrics_columns)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def main() -> None:
    """Evaluate the trained model on the test dataset and log results."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH.resolve()}"
        )

    # Load the trained checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model = DentDetectModel(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    # Create the test dataloader
    test_loader = CreateDataloader(
        img_size=IMAGE_SIZE,
        mean=train_mean,
        std=train_std,
        batch_size=BATCH_SIZE,
    ).create_test_dataloader()

    test_loss = 0.0
    test_samples = 0

    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("DentDetect")

    with mlflow.start_run(run_name="DentDetect_CNN_Testing"):
        mlflow.log_params({
            "model": "DentDetectModel",
            "threshold": THRESHOLD,
            "image_size": str(IMAGE_SIZE),
            "batch_size": BATCH_SIZE,
            "device": str(device),
            "num_classes": len(CLASS_NAMES),
        })

        # Run evaluation without calculating gradients
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                test_loss += loss.item() * images.size(0)
                test_samples += images.size(0)

                probabilities = torch.sigmoid(logits)

                # Convert tensors to CPU NumPy arrays
                all_labels.append(labels.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())

        if test_samples == 0:
            raise ValueError("The test dataloader returned no samples.")

        # Combine batches into complete arrays
        y_true = np.concatenate(all_labels, axis=0)
        y_prob = np.concatenate(all_probabilities, axis=0)

        if y_true.shape != y_prob.shape:
            raise ValueError(
                f"Label shape {y_true.shape} does not match "
                f"probability shape {y_prob.shape}."
            )

        if y_true.ndim != 2 or y_true.shape[1] != len(CLASS_NAMES):
            raise ValueError(
                f"Expected arrays with six class columns; got {y_true.shape}."
            )

        # Convert probabilities to binary predictions
        y_pred = (y_prob >= THRESHOLD).astype(int)

        # Calculate per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            zero_division=0,
        )

        # Calculate macro-average metrics
        macro_precision, macro_recall, macro_f1, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        )

        avg_loss = test_loss / test_samples

        # Count actual and predicted positives for each class
        actual = y_true.sum(axis=0).astype(int)
        predicted = y_pred.sum(axis=0).astype(int)

        print(f"\nDevice: {device}")
        print(f"Test samples: {test_samples}")
        print(f"Test loss: {avg_loss:.4f}\n")

        # Log and print per-class metrics
        for index, class_name in enumerate(CLASS_NAMES):
            mlflow.log_metrics({
                f"{class_name}_precision": float(precision[index]),
                f"{class_name}_recall": float(recall[index]),
                f"{class_name}_f1": float(f1[index]),
                f"{class_name}_actual_positive": float(actual[index]),
                f"{class_name}_predicted_positive": float(
                    predicted[index]
                ),
            })

            print(
                f"{class_name:15s} | "
                f"Actual: {actual[index]:4d} | "
                f"Predicted: {predicted[index]:4d} | "
                f"Precision: {precision[index]:.4f} | "
                f"Recall: {recall[index]:.4f} | "
                f"F1: {f1[index]:.4f}"
            )

        # Log overall test metrics
        mlflow.log_metrics({
            "test_loss": float(avg_loss),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
        })

        # Save evaluation metrics
        save_metrics(
            filepath=METRICS_PATH,
            precision=precision,
            recall=recall,
            f1=f1,
            macro_precision=float(macro_precision),
            macro_recall=float(macro_recall),
            macro_f1=float(macro_f1),
            actual=actual,
            predicted=predicted,
        )

        # Save predictions here once the dataloader supplies image filenames.
        # Example:
        # save_test_predictions(
        #     filepath=PREDICTIONS_PATH,
        #     image_names=image_names,
        #     y_true=y_true,
        #     y_prob=y_prob,
        #     y_pred=y_pred,
        # )

    # The MLflow context manager finishes the run automatically.
    print(f"\nMacro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"\nMetrics saved to: {METRICS_PATH.resolve()}")


if __name__ == "__main__":
    main()
