# ECE 624 Project

## Project Title
**Fuzzy C-Means Clustering and CNN-Based Multiclass Skin Lesion Classification Using Dermoscopic Images**

## Introduction
This project explores an integrated framework for multi-class skin lesion classification. The approach combines Fuzzy C-Means (FCM) clustering-based segmentation with a Convolutional Neural Network (CNN) classifier, leveraging the HAM10000 dataset [1]. By incorporating segmentation, we aim to improve classification performance compared to a CNN-only baseline.

### Objectives
- **Integrate FCM clustering with CNN classification** for multi-class skin lesion classification and analysis.
- **Evaluate performance improvements** over a CNN-only classification model.
- **Identify strengths and limitations** of the proposed FCM-CNN approach to inform future enhancements in automated skin lesion classification and analysis.

## Experiments

### Source Code
The source code is divided into the following modules:

1. **Preprocessing (Proposed Model):** `PreProcessing.py`  
2. **FCM-Based Segmentation:** `fuzzySegmentation.py`  
3. **Segmentation Evaluation:** `segmentation_ground_truth.py`  
4. **CNN Training and Testing:** `model_cnn_holdout.py`

### Results
Result files for both FCM-CNN and CNN-only models are available in the `results` folder.

### Dataset
The dataset and saved models are provided at the following link:  
**[Google Drive: HAM10000 Dataset & Saved Models](https://drive.google.com/drive/folders/10LaJmZuhP1xtmZUqMcDY7FHzbwqVhuB0?usp=sharing)**

**Note:** Please download the dataset and saved models and place them in the parent directory before running the code.

### Documentation
A detailed project report can be found in the `Document` folder.

## References
1. P. Tschandl, *The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions*, Version V4, 2018. DOI: [10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)

## Acknowledgments
We sincerely thank our instructor for guidance and support throughout this project.
