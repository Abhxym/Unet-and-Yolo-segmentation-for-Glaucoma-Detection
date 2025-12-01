# Glaucoma Detection & Segmentation Pipeline

A high-accuracy, coarse-to-fine deep learning pipeline for Glaucoma detection using the Drishti-GS1 dataset. This project integrates **YOLOv8** for Optic Disc detection and a **ResNet34-based UNet++** for precise Optic Disc and Cup segmentation.

## 🚀 Key Features

*   **Coarse-to-Fine Strategy**: Uses YOLOv8 to locate the Optic Disc (OD) and then crops the region for high-resolution segmentation.
*   **Advanced Architecture**: Custom `NestedUNetResNet` model combining UNet++ skip connections with a pre-trained ResNet34 encoder.
*   **Robust Training**: Implements AdamW optimizer, Combined Loss (Dice + BCE), and Learning Rate Scheduling.
*   **Test Time Augmentation (TTA)**: Enhances inference accuracy by averaging predictions from original and flipped images.
*   **High Accuracy**: Achieves **>91% vCDR Accuracy** and **>97% Pixel Accuracy**.

## 📊 Results

Evaluated on the Drishti-GS1 Test Set (51 images):

| Metric | Score | Description |
| :--- | :--- | :--- |
| **vCDR Accuracy** | **91.97%** | Based on vertical Cup-to-Disc Ratio |
| **vCDR MAE** | 0.0483 | Mean Absolute Error |
| **Pixel Accuracy (OD)** | 97.94% | Optic Disc segmentation accuracy |
| **Pixel Accuracy (Cup)** | 95.99% | Optic Cup segmentation accuracy |
| **Mean IoU (OD)** | 0.9586 | Intersection over Union for OD |
| **Mean IoU (Cup)** | 0.8303 | Intersection over Union for Cup |

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Abhxym/Unet-and-Yolo-segmentation-for-Glaucoma-Detection.git
    cd Unet-and-Yolo-segmentation-for-Glaucoma-Detection
    ```

2.  Install dependencies:
    ```bash
    pip install torch torchvision ultralytics numpy pillow matplotlib tqdm
    ```

## 📂 Dataset Structure

Ensure your dataset is organized as follows:

```
drishti/
├── Training/
│   ├── Images/
│   └── GT/
└── Test/
    ├── Images/
    └── Test_GT/
```

## ⚡ Usage

### 1. Train the Model
To train the ResNet34-UNet++ model on cropped images:
```bash
python src/train_crop.py
```
*   This will train for 100 epochs with data augmentation and save the best model to `checkpoints/best_model_crop.pth`.

### 2. Evaluate the Pipeline
To run the full end-to-end evaluation (YOLO detection -> Crop -> Segmentation -> Metrics):
```bash
python src/evaluate_pipeline.py
```

### 3. Single Image Inference
To run inference on a single image:
```bash
python src/pipeline.py --image path/to/image.png
```

## 🧠 Model Architecture

The core segmentation model (`src/model.py`) is a **NestedUNetResNet**:
*   **Encoder**: ResNet34 (pre-trained on ImageNet) layers `conv1`, `layer1`, `layer2`, `layer3`.
*   **Decoder**: UNet++ dense skip connections with Deep Supervision.
*   **Input**: 512x512 RGB images (cropped around Optic Disc).
*   **Output**: 2-channel mask (Optic Disc, Optic Cup).

## 📜 License

This project is open-source. Please cite the Drishti-GS1 dataset if you use it for research.
