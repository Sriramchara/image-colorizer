# Image Colorization with U-Net and Perceptual Loss

This project implements an image colorization pipeline using a Convolutional Autoencoder (U-Net) and VGG16-based Perceptual Loss.

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

## Description
*   **Model**: U-Net with ResNet-like encoder blocks (simplified) or standard Conv blocks.
*   **Loss**: L1 Pixel Loss + VGG16 Perceptual Loss.
*   **Preprocessing**: RGB -> LAB conversion. L channel is input, ab channels are target.
