# Image Colorizer with U-Net & Perceptual Loss

**An interactive Deep Learning application to colorize black and white images.**

Built with **PyTorch** and **Gradio**, this project features a U-Net architecture trained with VGG16 Perceptual Loss to produce realistic colorization results. It includes a user-friendly web UI for both training custom models and running inference on grayscale images.

## Features
*   **Model**: U-Net with ResNet-like encoder blocks or standard Conv blocks.
*   **Loss**: L1 Pixel Loss + VGG16 Perceptual Loss for sharp, perceptually accurate colors.
*   **Preprocessing**: RGB to LAB color space conversion (L channel input, ab channels target).
*   **Interface**: Interactive Gradio Web UI.

## Hardware Requirements
- **GPU**: NVIDIA GTX 1650 (4GB VRAM) or better recommended.
- **RAM**: 16GB+ recommended.
- **OS**: Windows (tested), Linux.

## Setup

1.  **Install Python 3.10+**
2.  **Create Virtual Environment**
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\activate
    ```
3.  **Install Dependencies**
    ```powershell
    pip install -r requirements.txt
    ```
    *Note: Ensure you have PyTorch installed with CUDA support. If not, see [pytorch.org](https://pytorch.org).*

## Data Preparation

1.  **Download Datasets**
    Run the helper script to attempt automatic download of Places365 (small) and CelebA:
    ```powershell
    python src/download_data.py
    ```
    *   **Places365**: The script tries to download the 256x256 validation set.
    *   **CelebA**: The script tries to download CelebA. If it fails (due to Google Drive quotas), download manually from [Kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset) and extract to `data/celeba_raw`.

2.  **Directory Structure**
    Ensure your data looks like this:
    ```
    data/
      places_raw/
      celeba_raw/
    ```

## Training

To start training:
```powershell
python src/train.py --epochs 20 --batch_size 8 --size 128
```
*   **--batch_size**: Reduce to 4 if you hit Out Of Memory (OOM) errors on GTX 1650.
*   **--size**: Image size (default 128).
*   **--lambda_p**: Weight for perceptual loss (default 0.01).

Checkpoints are saved in `checkpoints/`.
Sample validations are saved in `outputs/samples/`.

## Inference (Demo)

To colorize a grayscale image:
```powershell
python src/inference.py --img path/to/gray_image.jpg --ckpt checkpoints/epoch_20.pth --out result.jpg
```
