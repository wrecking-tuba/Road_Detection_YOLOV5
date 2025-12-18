# Road Detection Using YOLOv5 (Pothole Detection)

## Overview

This repository presents an end-to-end **pothole detection system using YOLOv5**, focused on improving road safety and infrastructure maintenance through real-time object detection. The project covers dataset creation, annotation, transfer learning with YOLOv5, experimental evaluation, and deployment using **Gradio**. In addition, a lightweight **mini-network** is implemented and tuned using **Ray Tune** for comparative experimentation.

The repository is structured to support both research-oriented experimentation and practical deployment.

---

## Key Features

* Custom pothole dataset collected via smartphone cameras
* YOLOv5-based object detection with transfer learning from COCO
* Data augmentation using Albumentations
* Training, validation, and testing pipelines
* Precision–Recall evaluation
* Interactive deployment using Gradio
* Lightweight CNN (Mini-Network) with hyperparameter tuning via Ray Tune

---

## Repository Structure

```
Road_Detection_YOLOV5/
│
├── Deployment_yolov5/        # Gradio-based deployment scripts
├── Mini_network/            # Lightweight CNN implementation + Ray Tune experiments
├── Pothole_Images/          # Sample dataset images
├── Update_1/                # Training / experiment updates
├── Update_2/                # Training / experiment updates
│
├── Pothole_detection_yolov5.ipynb   # End-to-end training & evaluation notebook
├── Report-POTHOLE DETECTION USING YOLOV5.docx
├── .gitignore
└── README.md
```

---

## Dataset

* **Total Images**: 200
* **Source**: Manually collected using smartphone cameras
* **Image Formats**: `.jpg`
* **Image Resolutions**:

  * 3024 × 4032
  * 1200 × 1600
* **Classes**: 1 (Pothole)

### Annotation

* Annotation Tool: **LabelImg**
* Format: **YOLO format**

Each image has a corresponding `.txt` file with the following structure:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized between 0 and 1 relative to image dimensions.

---

## YOLOv5 Training Pipeline

### Model Architecture

* Backbone: YOLOv5 (C3 blocks + SPPF)
* Detection Scales: P3, P4, P5
* Anchors: Default YOLOv5 anchors
* Pretrained Weights: **COCO**
* Number of Classes: 1 (overridden via data config)

### Objective Function

YOLOv5 optimizes a composite loss function consisting of:

* **Classification Loss** (Cross-Entropy)
* **Bounding Box Regression Loss**
* **Objectness Loss**
* **IoU-based Loss**

---

## Data Augmentation

Augmentation is applied using **Albumentations** in addition to YOLOv5’s default augmentations:

* Blur
* Median Blur
* Grayscale conversion

This improves robustness given the relatively small dataset size.

---

## Transfer Learning Strategy

Due to the limited dataset size, transfer learning is employed:

1. Initialize with COCO-pretrained YOLOv5 weights
2. Freeze backbone layers (feature extractor)
3. Train detection head
4. Fine-tune entire model with reduced learning rate

### Key Training Parameters

* Batch size: Auto-scaled based on hardware
* Image size: Configurable via training arguments
* Epochs: Tunable
* Optimizer: SGD

---

## Evaluation

* Dataset Splits: Train / Validation / Test
* Metrics:

  * Precision
  * Recall
  * Precision–Recall Curve

The validation pipeline ensures generalization assessment on unseen data.

---

## Deployment (Gradio)

An interactive web-based interface is built using **Gradio**:

### Deployment Workflow

1. Load trained YOLOv5 model (`best.pt`)
2. Run inference on uploaded images
3. Draw bounding boxes for detected potholes
4. Display results in browser

This enables rapid testing and demonstration without requiring deep technical setup.

---

## Mini-Network (Lightweight CNN)

A custom CNN is implemented for comparison and experimentation:

### Architecture

* Convolution + ReLU
* MaxPooling layers
* Fully Connected output layer

### Optimization & Tuning

* Hyperparameter tuning via **Ray Tune**
* Search over:

  * Learning rate
  * Network depth and width
  * Learning rate decay policies

This module demonstrates systematic experimentation beyond YOLO-based detection.

---

## Experiments & Results

* YOLOv5 transfer learning significantly outperforms training from scratch
* Fine-tuning improves localization accuracy
* Data augmentation enhances robustness
* Mini-network serves as a baseline and research comparison

---

## Conclusion

This project demonstrates the effectiveness of **YOLOv5 for real-time pothole detection**, offering a scalable and deployable solution for road condition monitoring. The combination of transfer learning, data augmentation, and interactive deployment highlights a practical pathway from research to real-world application.

Future work may focus on:

* Expanding dataset size
* Multi-class road defect detection
* Video-based real-time inference
* Edge deployment optimization

---

## References

* YOLOv5 by Ultralytics
* COCO Dataset
* Albumentations
* Gradio
* Ray Tune

---

## Author

**Devnath Reddy Motati**
