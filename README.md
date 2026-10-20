# DentDetect

**DentDetect** is a computer vision project that uses deep learning to identify visible vehicle damage from car images. The system predicts the presence of six types of damage using a multi-label image classification model.

The project is being developed as part of a hands-on learning task at GoMyCode, providing practical experience in machine learning, computer vision, model evaluation, experiment tracking, and end-to-end application development.

## Project Objective

The goal of DentDetect is to develop an image classification system that analyzes a car image and predicts which of the supported damage types are present.

Unlike a simple damaged-versus-undamaged classifier, DentDetect supports multiple damage labels for a single image.

### Supported Damage Classes

- Dent
- Scratch
- Crack
- Glass shatter
- Lamp broken
- Tire flat

## Technologies

- **Python:** Core programming language.
- **PyTorch:** Building and training the convolutional neural network (CNN).
- **Torchvision:** Image transformations and dataset handling.
- **Pandas and NumPy:** Data handling and numerical operations.
- **Pillow:** Image processing.
- **Scikit-learn:** Model evaluation and classification metrics.
- **MLflow:** Experiment tracking and model performance logging.
- **Jupyter Notebook:** Interactive experimentation and development.

### Planned Application Technologies

- **FastAPI:** Backend API development.
- **SQLAlchemy:** Database interaction through an ORM, if required.
- **PostgreSQL:** Database management, if selected for the final application.
- **Frontend framework:** To be selected during application development.

The final application stack may change as development progresses.

## Model Development

DentDetect currently uses a custom convolutional neural network (CNN) built and trained from scratch using PyTorch.

The model produces predictions for six damage classes. Since an image may contain more than one damage type, the project uses a multi-label classification approach.

### Training Setup

- Training images: 2,816
- Validation images: 810
- Image size: 128 × 128 pixels
- Batch size: 32
- Training epochs: 10
- Optimizer: Adam
- Loss function: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)
- Prediction threshold: 0.26

## Model Evaluation

The model has been trained and evaluated using a validation dataset. Prediction thresholds were compared to understand the trade-off between precision and recall.

At the selected global threshold of **0.26**, the validation results are:

| Metric | Score |
|---|---:|
| Macro precision | 60.37% |
| Macro recall | 79.30% |
| Macro F1-score | 67.64% |

These are macro-averaged validation metrics across the six damage classes. They summarize the model's current performance but do not guarantee the same results on unseen real-world images.

A separate test-set evaluation is still needed to assess generalization.

## Experiment Tracking

MLflow is used to record model training experiments and evaluation metrics. This helps compare experiments and track how changes to model settings affect performance.

## Project Status

**Status:** In development.

The initial model training and validation stage is complete. Work will now focus on preparing the trained model for use in an application.

## Development Roadmap

- [x] Dataset exploration and preparation.
- [x] Image preprocessing.
- [x] CNN model development and training.
- [x] Validation and model evaluation.
- [x] Prediction threshold comparison and selection.
- [x] MLflow experiment tracking.
- [ ] Final test-set evaluation.
- [ ] Backend API development with FastAPI.
- [ ] Frontend integration.
- [ ] End-to-end application testing.
- [ ] Application deployment.

## Scope and Limitations

DentDetect is designed to identify visible vehicle damage from images. Its predictions may be affected by image quality, lighting, viewing angle, occlusion, and the distribution of damage types represented in the training dataset.

The model may produce false positives or miss actual damage, particularly for classes with weaker validation performance. Its predictions should therefore be treated as automated estimates rather than definitive assessments.

DentDetect is a learning project and is not intended to replace a professional vehicle inspection.

## Author

**David Ikejiaku**

Developed as a hands-on machine learning and computer vision learning project at GoMyCode.
