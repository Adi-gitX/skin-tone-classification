# Skin Tone Classification using EfficientNetB0

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-blue?style=for-the-badge&logo=keras)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🎥 Demo Video
<a href="https://youtu.be/fsft_bwV3_Y" target="_blank">
  <img src="https://img.youtube.com/vi/fsft_bwV3_Y/maxresdefault.jpg" 
       alt="Demo Video" 
       style="width:100%; border-radius:12px;">
</a>

This repository contains the Jupyter Notebook and documentation for a deep learning model that classifies images into one of three skin-tone categories: **dark**, **fair**, and **light**.

This model is part of a larger fashion personalization system, as described in the accompanying Phase 1 Report and the academic paper *"Optimizing Skin Tone Classification Using EfficientNet for AI-Powered Fashion Personalization."*

---

## Key Features

* **Model:** Built using the **EfficientNetB0** architecture with transfer learning from ImageNet.
* **Dataset:** Trained on the **SkinTone Dataset** from Kaggle, containing 2,100 images  
  (1,470 training, 420 validation, 210 test).
* **Performance:** Final model `efficientnet_finetuned_v5.keras` achieves **79.05% test accuracy** across 3 classes.

---

## Tech Stack

* Python  
* TensorFlow / Keras  
* KaggleHub (for dataset + model loading)  
* EfficientNetB0  
* scikit-learn (for metrics)

---

## Project Resources (Kaggle Hub)

The pre-trained model and dataset are publicly available:

* **Pre-trained Model:**  
  `https://www.kaggle.com/models/adityakammati/skintone-images-model`

* **Dataset:**  
  `https://www.kaggle.com/datasets/adityakammati/skintone-dataset`

---

## Project Structure

* **`skin_tone_classification_Traning.ipynb`**  
  Full notebook containing:
  - Data loading  
  - Preprocessing  
  - EfficientNetB0 fine-tuning  
  - Training  
  - Saving the model  
  - “Run from Here” testing section

* **`skin_tone_classification_testing_notebook.ipynb`**  
  Notebook for running predictions on new images.

* **`Skin-Tone-ResearchPaper.pdf`**  
  Complete academic paper.

* **`HOW_TO_RUN.md`**  
  Step-by-step instructions.

---

## How to Use

To run the model, follow the instructions in **`HOW_TO_RUN.md`**.

