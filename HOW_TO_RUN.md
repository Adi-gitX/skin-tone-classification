# Procedure: How to Run the Skin Tone Model

This procedure is based on the `skin_tone_classification_(AIML)_(1).ipynb` notebook.

This single notebook is designed to be run in an environment like Google Colab or Kaggle, as it requires a GPU and uses the `kagglehub` library to fetch the data and model.

### Option 1: Run Prediction Only (Recommended)

This method loads the pre-trained model directly from Kaggle Hub and is the fastest way to test a new image.

1.  Open the notebook in your environment.
2.  Run the initial setup cells to import libraries (like `tensorflow`, `kagglehub`, `cv2`, `numpy`, etc.).
3.  Scroll down to the section titled **"Model Testing — Run from Here"**.
4.  Run the cell that downloads the model from Kaggle Hub:
    ```python
    import kagglehub
    
    # Download latest version
    path = kagglehub.model_download("adityakammati/skintone-images-model/keras/default")
    model_file = os.path.join(path, "efficientnet_finetuned_v5.keras")
    print("Path to model files:", model_file)
    ```
5.  Run the cell that loads the model (e.g., `model = load_model(model_file)`).
6.  Find the cell that defines the `predict_and_show` function and run it.
7.  In the final cell, change the `test_image_path` variable to the path of your new test image and run it. The notebook will display the image and its predicted class (dark, fair, or light).

### Option 2: Re-Train the Model from Scratch

If you want to train the model yourself:

1.  Open the notebook in a **GPU-enabled** environment.
2.  Run all cells in the notebook from top to bottom.
3.  The notebook will automatically:
    * Download the **SkinTone Dataset** from Kaggle Hub:
        ```python
        import kagglehub
        
        # Download latest version
        path = kagglehub.dataset_download("adityakammati/skintone-dataset")
        print("Path to dataset files:", path)
        ```
    * Load and preprocess the 1,470 training images and 420 validation images.
    * Define the **EfficientNetB0** model architecture.
    * Train and fine-tune the model.
    * Save the final model as `efficientnet_finetuned_v5.keras`.
