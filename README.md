# Skin Tone Classification using EfficientNetB0

[cite_start]This repository contains the Jupyter Notebook and documentation for a deep learning model that classifies images into one of three skin-tone categories: **dark, fair, or light**[cite: 22, 1078].

[cite_start]This model is a component of a larger fashion personalization system, as detailed in the accompanying Phase 1 Report  [cite_start]and the academic paper, "Optimizing Skin Tone Classification Using EfficientNet for AI-Powered Fashion Personalization" [cite: 844-845].

### Key Features

* [cite_start]**Model:** Uses the **EfficientNetB0** architecture [cite: 29-33, 852, 1094], leveraging transfer learning from ImageNet for high efficiency.
* [cite_start]**Dataset:** Trained on the **SkinTone Dataset** from Kaggle, which contains 2,100 images (1,470 training, 420 validation, 210 test) [cite: 1077, 1096-1098].
* [cite_start]**Performance:** The final model saved as `efficientnet_finetuned_v5.keras` [cite: 721] [cite_start]achieves a **test accuracy of 79.05%** on the 3 classes[cite: 723, 1100].

### Tech Stack

* **Python**
* [cite_start]**TensorFlow / Keras** [cite: 14-16]
* [cite_start]**KaggleHub** (for easy loading of the model and dataset) [cite: 17, 725, 730]
* [cite_start]**EfficientNetB0** [cite: 29, 895]
* [cite_start]**scikit-learn** (for metrics) [cite: 724]

### Project Resources (Kaggle Hub)

The pre-trained model and dataset are publicly available on Kaggle Hub:

* **Pre-trained Model:**
    * **URL:** `https://www.kaggle.com/models/adityakammati/skintone-images-model`

* **Dataset:**
    * **URL:** `https://www.kaggle.com/datasets/adityakammati/skintone-dataset`

### Project Structure

* **`skin_tone_classification_(AIML)_(1).ipynb`**: The complete Jupyter notebook containing all code for:
    * [cite_start]Data loading and preprocessing [cite: 17-22, 609-632].
    * [cite_start]Model definition and fine-tuning (EfficientNetB0) [cite: 29-33, 649-688].
    * [cite_start]Model training and saving [cite: 34-47, 689-722].
    * [cite_start]A "Run from Here" testing section to predict on new images [cite: 728-751].
* [cite_start]**`IEEE_Conference_Template (1).pdf`**: The academic paper describing the methodology .
* [cite_start]**`Phase 1 Report.pdf`**: The summary report for the project .

### How to Use

For detailed steps on running the model, please see **`HOW_TO_RUN.md`**.
