# Skin Tone Classification using EfficientNetB0

This repository contains the Jupyter Notebook and documentation for a deep learning model that classifies images into one of three skin-tone categories: **dark, fair, or light**.

This model is a component of a larger fashion personalization system, as detailed in the accompanying Phase 1 Report and the academic paper, "Optimizing Skin Tone Classification Using EfficientNet for AI-Powered Fashion Personalization."

### Key Features

* **Model:** Uses the **EfficientNetB0** architecture, leveraging transfer learning from ImageNet for high efficiency.
* **Dataset:** Trained on the **SkinTone Dataset** from Kaggle, which contains 2,100 images (1,470 training, 420 validation, 210 test).
* **Performance:** The final model saved as `efficientnet_finetuned_v5.keras` achieves a **test accuracy of 79.05%** on the 3 classes.

### Tech Stack

* **Python**
* **TensorFlow / Keras**
* **KaggleHub** (for easy loading of the model and dataset)
* **EfficientNetB0**
* **scikit-learn** (for metrics)

### Project Resources (Kaggle Hub)

The pre-trained model and dataset are publicly available on Kaggle Hub:

* **Pre-trained Model:**
    * **URL:** `https://www.kaggle.com/models/adityakammati/skintone-images-model`

* **Dataset:**
    * **URL:** `https://www.kaggle.com/datasets/adityakammati/skintone-dataset`

### Project Structure

* **`skin_tone_classification_(AIML)_(1).ipynb`**: The complete Jupyter notebook containing all code for:
    * Data loading and preprocessing.
    * Model definition and fine-tuning (EfficientNetB0).
    * Model training and saving.
    * A "Run from Here" testing section to predict on new images.
* **`IEEE_Conference_Template (1).pdf`**: The academic paper describing the methodology.
* **`Phase 1 Report.pdf`**: The summary report for the project.

### How to Use

For detailed steps on running the model, please see **`HOW_TO_RUN.md`**.
