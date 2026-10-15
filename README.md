Pothole Detection Using YOLOv8
Overview
This project implements an object detection system for identifying potholes in road images using YOLOv8.

The model is trained to locate potholes using bounding-box annotations and can identify the position of potholes within an image rather than simply classifying whether a pothole is present.

The project covers dataset preparation, annotation conversion, YOLO-format dataset organization, model training, validation, and pothole detection on images.

Dataset
The project uses the Annotated Potholes Image Dataset originally created by Atikur Rahman Chitholian.

Kaggle Dataset
Annotated Potholes Image Dataset - Kaggle

The original dataset contains 665 road images with annotated potholes. The dataset was subsequently exported in YOLOv8 format for object detection.

Dataset Distribution
Split	Images
Training	466
Validation	134
Testing	68
Total	668*
The dataset source documentation states 665 images; the exported directory structure contains 466 training, 134 validation, and 68 test images. The repository therefore preserves the exported dataset structure rather than modifying the source counts.

Classes
The project contains one object-detection class:

pothole
Objective
The primary objective is to develop a computer vision model capable of automatically detecting potholes in road images.

The system predicts:

Whether a pothole is present
The location of the pothole
The bounding box surrounding the detected pothole
The model confidence for each detection
Technology Stack
Python
YOLOv8
Ultralytics
OpenCV
NumPy
Matplotlib
Google Colab
Roboflow YOLOv8 dataset format
Object Detection Workflow
Road Image Dataset
        |
Annotated Bounding Boxes
        |
YOLOv8 Dataset Preparation
        |
Train / Validation / Test Split
        |
YOLOv8 Model Training
        |
Model Validation
        |
Best Model Weights
        |
New Image Inference
        |
Pothole Bounding Box Detection
Dataset Preparation
The original pothole annotations were converted into YOLO-compatible label files.

Each annotation follows the YOLO object-detection format:

class_id x_center y_center width height
The dataset configuration is defined in data.yaml.

nc: 1
names: ['pothole']
The dataset uses separate directories for:

train/
valid/
test/
Each split contains corresponding image and label directories.

Model
The project uses YOLOv8 Nano (yolov8n) for pothole detection.

YOLOv8 is a one-stage object detection architecture designed to perform object localization and classification efficiently.

The Nano variant provides a lightweight model suitable for experimentation and deployment on systems with limited computational resources.

Training Configuration
The model is trained using the YOLOv8 training pipeline.

Key configuration includes:

Parameter	Value
Model	YOLOv8n
Task	Object Detection
Classes	1
Input Image Size	640 × 640
Epochs	50
Dataset Format	YOLOv8
Target Class	Pothole
Model Evaluation
The trained model is evaluated using object-detection metrics such as:

Precision
Recall
mAP@50
mAP@50–95
Training and validation losses
The project also performs inference on validation/test images to visually inspect detected potholes and their bounding boxes.

Inference
After training, the best model weights can be used to detect potholes in new images.

Example workflow:

from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="test_image.jpg",
    imgsz=640,
    conf=0.25
)
The output contains detected pothole bounding boxes and confidence scores.

Project Structure
Pothole-Detection-YOLOv8/
│
├── pothole_detection.ipynb
├── data.yaml
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
├── runs/
│   └── detect/
│       └── ...
│
└── README.md
Key Skills Demonstrated
Computer Vision
Object Detection
YOLOv8
Deep Learning
Image Annotation
YOLO Label Processing
Dataset Preparation
Model Training
Model Validation
Object Localization
Image Inference
Python
Ultralytics
OpenCV
Applications
A pothole detection system can be used as a foundation for:

Automated road-condition monitoring
Road maintenance systems
Smart-city infrastructure monitoring
Vehicle-mounted road inspection
Road-damage mapping
Infrastructure maintenance prioritization
Limitations
The dataset contains a single object class and a relatively small number of images compared with large-scale object-detection datasets.

Performance may also vary depending on:

Lighting conditions
Camera angle
Pothole size
Road surface
Image quality
Weather conditions
Occlusion
Distance from the pothole
The model should therefore be evaluated on independent real-world road images before being considered for practical deployment.

Future Improvements
Train with a larger and more diverse road dataset
Add additional road-damage classes
Perform hyperparameter tuning
Compare YOLOv8 model sizes
Apply controlled data augmentation
Evaluate performance on real-world dashcam footage
Implement real-time video detection
Add GPS-based pothole mapping
Develop a road-condition monitoring dashboard
Deploy the model on an edge device
