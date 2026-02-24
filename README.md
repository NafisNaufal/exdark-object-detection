# Low-Light Object Detection - ExDARK with YOLO11

This repository contains experiments leveraging the **ExDARK** (Exclusively Dark Image) dataset to train robust object detection models capable of identifying common objects in extremely low-light and nighttime conditions. The core model used across experiments is **YOLO11m** (Ultralytics).

## 🚀 Overview of Experiments

The project consists of exploratory data analysis and a progression of three increasingly sophisticated training pipelines designed to combat the loss of visual contrast in dark environments.

### 1. `eda_exdark.ipynb`
*   **Purpose**: Exploratory Data Analysis. 
*   **Key Features**: Parses robustly formatted `bbGt`/YOLO annotations, analyzes class distributions across 12 object categories, and visualizes ground truth bounding boxes overlaid on the dark images to ensure coordinate mapping is correct.

### 2. `yolo-baseline.ipynb` 
*   **Purpose**: Establish a baseline performance.
*   **Key Features**: Formats the ExDARK dataset to strict YOLO standards and trains a default YOLO11m model (40 epochs) directly on raw, unmodified dark images using standard augmentations. This provides the performance floor to measure future improvements against.

### 3. `yolo-clahe.ipynb`
*   **Purpose**: Introduce offline contrast enhancement.
*   **Key Features**: Implements **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to boost local contrast before training. Images are converted to LAB color space, where the Lightness channel (`L`) is equalized (`clipLimit=2.0`, `tileGridSize=(8, 8)`), and then converted back. YOLO11m is then trained on these structurally enhanced images.

### 4. `yolo-clahe-llaug.ipynb`
*   **Purpose**: Combine CLAHE with custom Low-Light Photometric Augmentation.
*   **Key Features**: The most advanced pipeline. Takes the CLAHE-enhanced base and statically augments a random 30% of the training split to enforce robustness against lighting shifts. The custom augmentations include:
    *   **Random Brightness Reduction** (down to 30%-70% of original)
    *   **Gamma Adjustments** (0.4 to 0.8 to heavily warp lighting curves)
    *   **Gaussian Noise Injection** (simulating high-ISO sensor noise found in low-light cameras)
    *   **Color Temperature Shifts**
*   **Training Specs**: Trains for an extended **100 epochs** while adjusting dynamic YOLO HSV augmentations (e.g., heavily weighting brightness variation `hsv_v=0.6` while dropping saturation `hsv_s=0.5`).

## 📊 Dataset: ExDARK
The dataset contains 12 classes of objects: `Bicycle`, `Boat`, `Bottle`, `Bus`, `Car`, `Cat`, `Chair`, `Cup`, `Dog`, `Motorbike`, `People`, `Table`. 
*   **Evaluation Split**: The framework divides the annotated training pool into an 85% Training / 15% Validation split.

## 🛠 Setup & Requirements

```bash
pip install ultralytics opencv-python matplotlib seaborn pandas numpy tqdm pillow
```

## ⚙️ Running the Code
1. Place the dataset inside the standard Kaggle or local directory format (e.g., `dataset/train/images`, `dataset/train/annotations`).
2. Update the `KAGGLE_INPUT` or `DATA_DIR` paths in the first cell of the desired notebook. 
3. Run the cells sequentially. Each training notebook handles YOLO formatting, model initiation, training, validation, test inference, and generates a `submission.csv` automatically in the `outputs/results` directory.
