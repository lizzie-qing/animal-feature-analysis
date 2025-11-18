#  Animal Facial Feature Analysis  
### **A Complete AI Pipeline: YOLOv8 Training · Batch Inference · Geometric Feature Extraction · Statistical Visualization**  
*(Author: Lizzie — Data Science & Multimodal Perception)*

---

##  Overview  

This repository demonstrates an end-to-end AI pipeline integrating YOLOv8 detection, geometric facial feature engineering,
and scientific visualization for analyzing structural patterns in animal faces.
This project implements a **full end-to-end AI pipeline** for analyzing animal facial features.  
The workflow combines:

###  Computer Vision Engineering
- Custom YOLOv8 training for animal facial landmarks  
- Left-right flip disabled to preserve asymmetry of eyes/ears  
- Batch inference with confidence filtering

###  Data Science & Feature Engineering
- Extraction of geometric features:
  - EFR (Eye-to-Face Ratio)  
  - ESI (Eye Shape Index)  
  - Eye Symmetry Difference (%)  
  - fWHR (Facial Width-to-Height Ratio)  
  - Ear Uprightness (Left / Right / Average)
- KDE plots, histograms, mean–σ visualization  
- Multi-species comparison (cat/dog/fox/hamster)

### Multimodal Research Workflow
A reproducible pipeline suitable for scientific analysis  
(e.g., comparing structural features across animals).

---

## Project Structure
```
animal-feature-analysis/
├── README.md
├── data/
│   └── animal_features_extracted.csv
├── results/
│   ├── fWHR_histogram.png
│   ├── training_plots/
│   │   ├── loss_curves.png
│   │   └── mAP_curves.png
│   └── feature_distribution_plots/
├── src/
│   ├── feature_extraction/
│   │   ├── extract_animal_features.py
│   │   └── create_data_yaml.py
│   ├── visualization/
│   │   ├── plot_feature_distributions.py
│   │   ├── plot_fWHR_histogram.py
│   │   └── plot_training_log.py
│   ├── inference/
│   │   └── batch_inference.py
│   └── training/
│       └── train_yolov8_noflip.py
└── test_images/
```

## 1. Feature Extraction

Run:
```bash
python src/feature_extraction/extract_animal_features.py
```
Output:
```bash
data/animal_features_extracted.csv
```
Features include:
EFR
ESI
fWHR
Eye symmetry difference
Ear uprightness (left, right, avg)

## 2. Visualization

KDE Distribution:
```bash
python src/visualization/plot_feature_distributions.py
```
fWHR Histogram:
```bash
python src/visualization/plot_fWHR_histogram.py
```
YOLO Training Curves:
```bash
python src/visualization/plot_training_log.py
```

## 3. Batch Inference
```bash
python src/inference/batch_inference.py
```
Outputs annotated images in:
```bash
results/batch_inference_output/
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
## Skills Demonstrated

**Python** (NumPy, pandas, matplotlib, seaborn)

**Computer Vision** (YOLOv8 training & inference)

**Data Engineering & Feature Extraction**

**Statistical Visualization**

**End-to-end AI Pipeline Design**

**Research Documentation & Reproducibility**

---
Author

Lizzie Qing

