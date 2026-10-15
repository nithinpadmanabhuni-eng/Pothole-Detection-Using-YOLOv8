# Pothole Detection Using YOLOv8

## Overview

This project implements a deep learning-based object detection system to identify potholes in road images using **YOLOv8 (You Only Look Once)**.

Unlike image classification, which only predicts whether an image contains a pothole, this system performs object detection by identifying the pothole's location and drawing a bounding box around it.

The project covers dataset preparation, annotation conversion, model training, validation, evaluation, and inference on new road images.

## Objectives

- Detect potholes automatically in road images.
- Identify the location of each detected pothole.
- Draw bounding boxes around potholes.
- Estimate the confidence score for each detection.
- Evaluate model performance using object detection metrics.
- Build a foundation for automated road-condition monitoring.

## Dataset

The project uses the **Annotated Potholes Image Dataset**, originally created by Atikur Rahman Chitholian.

- **Dataset:** [Annotated Potholes Image Dataset on Kaggle](https://www.kaggle.com/datasets/atikurrahmanchitholian/annotated-potholes-dataset)
- **Original dataset size:** 665 annotated road images.
- **Annotation format:** Pascal VOC XML, converted to YOLO format when required.
- **Object class:** `pothole`

The repository's exported YOLO-format dataset is described as having the following distribution:

| Split | Images |
|---|---:|
| Training | 466 |
| Validation | 134 |
| Testing | 68 |
| **Total** | **668*** |

\*The source documentation states 665 images, while the exported split counts sum to 668. The actual dataset files should be checked to confirm the final counts.

### Dataset Structure

The prepared dataset follows the YOLO object detection directory structure:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

Each image has a corresponding label file containing the annotated pothole bounding boxes.

## Technology Stack

- Python
- YOLOv8
- Ultralytics
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Google Colab
- Roboflow YOLO dataset format

## Model Architecture

The project uses **YOLOv8 Nano (`yolov8n`)**, a lightweight, one-stage object detection model.

YOLO predicts object locations and class probabilities in a single forward pass, making it suitable for efficient image inference.

The Nano model is selected as a starting point because it requires relatively fewer computational resources than larger YOLOv8 variants.

## Object Detection Workflow

```text
Road Image Dataset
        |
        v
Pothole Bounding-Box Annotations
        |
        v
Dataset Preparation
        |
        v
YOLO Annotation Conversion
        |
        v
Train / Validation / Test Split
        |
        v
YOLOv8n Model Training
        |
        v
Model Validation and Evaluation
        |
        v
Best Model Weights (best.pt)
        |
        v
Inference on New Road Images
        |
        v
Pothole Bounding Boxes and Confidence Scores
```

## Dataset Preparation

The original annotations may be provided in Pascal VOC XML format. These annotations must be converted into YOLO-compatible text files if the dataset has not already been converted.

Each YOLO label follows this format:

```text
class_id x_center y_center width height
```

The bounding-box coordinates are normalized relative to the image width and height.

For this project, the class ID is:

```text
0 = pothole
```

The dataset configuration is defined in `data.yaml`:

```yaml
path: /path/to/dataset
train: train/images
val: valid/images
test: test/images

names:
  0: pothole
```

The dataset path must be updated to match the actual location of the prepared dataset.

## Installation

Install the required Python libraries:

```bash
pip install ultralytics opencv-python numpy matplotlib pyyaml
```

Verify the installation:

```python
import ultralytics

print(ultralytics.__version__)
```

## Model Training

Load the pretrained YOLOv8 Nano model and fine-tune it on the prepared pothole dataset.

```python
from ultralytics import YOLO

# Load pretrained YOLOv8 Nano weights
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="pothole_detection"
)
```

### Training Configuration

| Parameter | Value |
|---|---|
| Model | YOLOv8n |
| Task | Object Detection |
| Number of classes | 1 |
| Target class | Pothole |
| Input image size | 640 × 640 |
| Epochs | 50 |
| Dataset format | YOLO |
| Training framework | Ultralytics |

The batch size can be reduced if the available GPU memory is insufficient.

## Model Evaluation

The model should be evaluated on validation and test images using standard object detection metrics.

The main evaluation metrics include:

- **Precision:** Measures how many predicted potholes are correct.
- **Recall:** Measures how many actual potholes are detected.
- **mAP@50:** Measures mean average precision at an IoU threshold of 0.50.
- **mAP@50–95:** Measures average precision across multiple IoU thresholds.
- **Training and validation losses:** Help track model learning and potential overfitting.

Example validation code:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/pothole_detection/weights/best.pt")

metrics = model.val(
    data="data.yaml",
    split="test",
    imgsz=640
)

print("mAP@50:", metrics.box.map50)
print("mAP@50-95:", metrics.box.map)
```

The actual metric values should be recorded after training and evaluation. No performance values are assumed in this README.

## Pothole Detection on New Images

After training, the best model weights can be used to detect potholes in new road images.

```python
from ultralytics import YOLO

# Load trained weights
model = YOLO("runs/detect/pothole_detection/weights/best.pt")

# Run inference
results = model.predict(
    source="test_image.jpg",
    imgsz=640,
    conf=0.25,
    save=True
)

# Display detected classes and confidence scores
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        coordinates = box.xyxy[0].tolist()

        print("Class:", model.names[class_id])
        print("Confidence:", round(confidence, 3))
        print("Bounding box:", coordinates)
```

The output includes predicted pothole locations, bounding-box coordinates, and confidence scores. Annotated images can be saved using the `save=True` option.

## Project Structure

```text
Pothole-Detection-YOLOv8/
│
├── pothole_detection.ipynb
├── data.yaml
├── README.md
├── README.dataset.txt
├── README.roboflow.txt
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── runs/
    └── detect/
        └── pothole_detection/
            └── weights/
                ├── best.pt
                └── last.pt
```

The structure above represents the expected project layout. Some directories, including `runs/` and trained weights, are generated only after training.

## Applications

This system can serve as a foundation for:

- Automated road-condition monitoring.
- Smart-city infrastructure inspection.
- Vehicle-mounted road inspection.
- Road-damage mapping.
- Road maintenance prioritization.
- Dashcam-based pothole detection.

## Limitations

- The dataset contains a single detection class.
- The number of images is relatively small for a general-purpose detection system.
- Performance can vary with lighting, weather, camera angle, image quality, pothole size, occlusion, and distance.
- Detection confidence does not directly measure pothole severity.
- Real-world performance must be verified on independent road images and video footage.

## Future Improvements

- Train on a larger and more diverse dataset.
- Add other road-damage classes.
- Compare YOLOv8 Nano, Small, and Medium models.
- Tune hyperparameters and data augmentation.
- Implement real-time detection on dashcam footage.
- Integrate GPS-based pothole mapping.
- Develop a road-condition monitoring dashboard.
- Deploy the model on edge devices.


