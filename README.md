#  Animal Facial Feature Analysis  
### **YOLOv8 · Feature Engineering · Statistical Visualization**  
A Complete AI Pipeline for Multispecies Facial Structure Analysis
*(Author: Lizzie — Data Science & Multimodal Perception)*

---
<div align="center">

End-to-End AI Pipeline
—from detection → feature extraction → statistical analysis → visualization

</div>

##  Overview  

This project demonstrates a fully reproducible AI pipeline integrating:

### Computer Vision Engineering

YOLOv8 training for animal facial landmark detection

Left–right flip disabled to preserve facial asymmetry

Batch inference on multiple species (cat / dog / fox / bear / hamster)

###  Geometric Feature Engineering
Extracted features include:

EFR — Eye-to-Face Ratio

ESI — Eye Shape Index

fWHR — Facial Width-to-Height Ratio

Eye Symmetry Difference (%)

Ear Uprightness (left/right/avg)

### Statistical Visualization

KDE plots

Histograms

Multi-species distribution comparison

Training curves & confusion matrix visualization

### Research-Oriented Workflow
The pipeline is designed with scientific reproducibility in mind, suitable for:

cross-species structural analysis

perceptual studies

dataset exploration

morphological comparison

---

## Project Structure
```
animal-feature-analysis/
├── README.md
├── data/
│   └── animal_features_extracted_sample.csv
├── results/
│   ├── batch_inference_output/      # YOLO annotated outputs
│   └── feature_distribution_plots/  # KDE & histogram visualizations
│   └── training_plots/              # YOLO training curves & confusion matrix
│
├── src/
│   ├── feature_extraction/
│   │   ├── extract_animal_features.py
│   │   └── create_data_yaml.py
│   │
│   ├── visualization/
│   │   ├── plot_feature_distributions.py
│   │   ├── plot_fWHR_histogram.py
│   │   └── plot_training_log.py
│   │
│   ├── inference/
│   │   └── batch_inference.py
│   │
│   └── training/
│       └── train_yolov8_noflip.py
│   
└── test_images/                     # Sample inputs for inference
```

## Example Outputs
## 1. YOLOv8 Detection Result

(You can replace the image file below with one from results/batch_inference_output)

<p align="center"> <img src="results/batch_inference_output/cat4.jpg" width="55%"> </p>

## 2. Feature Distribution Visualization (KDE)
<p align="center"> <img src="results/feature_distribution_plots/EFR_advanced_distribution.png" width="65%"> </p>

## 3.YOLOv8 Training Curves
<p align="center"> <img src="results/training_plots/loss_curve.png" width="60%"> </p>

## How to Use
## 1. Run YOLOv8 Training
(Left-right flip disabled to avoid corrupting asymmetric facial landmarks)
```bash
python src/training/train_yolov8_noflip.py
```

## 2. Batch Inference on Sample Images
```bash
python src/inference/batch_inference.py
```
Outputs saved to:
```bash
results/batch_inference_output/
```

## 3. Extract Geometric Facial Features

```bash
python src/feature_extraction/extract_animal_features.py
```
Output:
```bash
data/animal_features_extracted_sample.csv
```

## 4. Visualize Feature Distributions
KDE + Per-species distributions:
```bash
python src/visualization/plot_feature_distributions.py
```
fWHR Histogram:
```bash
python src/visualization/plot_fWHR_histogram.py
```
Training Curves:
```bash
python src/visualization/plot_training_log.py
```

## 4. YOLOv8 Training

Left-right flip disabled to avoid confusing asymmetric landmarks.
```bash
python src/training/train_yolov8_noflip.py
```
Outputs saved under:
```bash
runs/detect_yolov8s_noflip/
```

---
## Technologies Used

**Programming**

Python

NumPy / pandas

Matplotlib / seaborn

**Computer Vision**

YOLOv8 — training & batch inference

Custom augmentation constraints

Bounding box + landmark-based geometric processing

**Data Science**

Feature engineering

Data normalization

Multi-species comparison

Scientific visualization

**Pipeline Design**

Modular directory structure

Fully reproducible workflow

Suitable for research & portfolio showcasing

Model weights not included due to size. 
You can download YOLOv8n/s from Ultralytics or train with train_yolov8_noflip.py.


---
Author

Lizzie Qing

