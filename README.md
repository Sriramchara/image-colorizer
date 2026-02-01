# Image Colorizer with U-Net & Perceptual Loss

**An interactive Deep Learning application to colorize black and white images.**

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)
![Gradio](https://img.shields.io/badge/Interface-Gradior-pink)

This project implements a state-of-the-art image colorization pipeline using a **U-Net Convolutional Autoencoder** trained with **VGG16 Perceptual Loss**. It includes a full-featured web interface for training models and running inference.

---

## 📋 Table of Contents
1. [Requirements](#-requirements)
2. [Libraries Used](#-libraries-used)
3. [Installation & Setup](#-installation--setup)
4. [How to Run](#-how-to-run)
    - [Running the Web App](#1-running-the-web-app-gui)
    - [Command Line Interface](#2-command-line-interface)
5. [Procedures](#-procedures)
    - [Data Preparation](#data-preparation)
    - [Training](#training)
    - [Inference](#inference)
6. [Outputs](#-outputs)
7. [Project Structure](#-project-structure)

---

## 💻 Requirements

To run this application efficiently, the following hardware is recommended:

*   **Operating System**: Windows 10/11 or Linux.
*   **GPU**: NVIDIA GPU with CUDA support (GTX 1650 / 4GB VRAM minimum; RTX 3060+ recommended).
    *   *Note: CPU-only training is possible but extremely slow.*
*   **RAM**: 16GB or more.
*   **Storage**: At least 10GB free space for datasets (CelebA/Places365) and models.

---

## 📚 Libraries Used

The core technology stack includes:

*   **[PyTorch](https://pytorch.org/)**: Core deep learning framework.
*   **[Torchvision](https://pytorch.org/vision/stable/index.html)**: Image transformations and datasets.
*   **[Gradio](https://gradio.app/)**: Web-based user interface for easy interaction.
*   **[OpenCV](https://opencv.org/)** (`opencv-python`): Image processing (LAB<->RGB conversion).
*   **[NumPy](https://numpy.org/)**: Matrix operations.
*   **[Matplotlib](https://matplotlib.org/)**: Visualization.
*   **[Tqdm](https://github.com/tqdm/tqdm)**: Progress bars.
*   **[Albumentations](https://albumentations.ai/)**: Advanced data augmentation.

See `requirements.txt` for specific version pinning.

---

## 🛠 Installation & Setup

Follow these steps to set up the project locally:

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Sriramchara/image-colorizer.git
    cd image-colorizer
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```powershell
    # Windows
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```powershell
    pip install -r requirements.txt
    ```
    *Important: If you have an NVIDIA GPU, ensure you install the CUDA-enabled version of PyTorch. The `requirements.txt` is set up for CUDA 11.8. If you need a different version, visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/).*

---

## 🚀 How to Run

### 1. Running the Web App (GUI)
The easiest way to use the application is via the Gradio interface.

```powershell
python src/app.py
```
*   This will launch a local web server (usually at `http://127.0.0.1:7860`).
*   Open the link in your browser to access the **Inference** and **Training** tabs.

### 2. Command Line Interface
You can also run scripts directly from the terminal.

**Training:**
```powershell
python src/train.py --epochs 20 --batch_size 8 --size 128
```

**Inference:**
```powershell
python src/inference.py --img tests/test_image.jpg --ckpt checkpoints/latest_model.pth --out result.jpg
```

---

## ⚙ Procedures

### Data Preparation
Before training, you need to download the datasets.
Run the helper script:
```powershell
python src/download_data.py
```
*   This attempts to download **Places365** and **CelebA**.
*   **Manual Setup**: If automatic download fails, place your images in:
    *   `data/celeba_raw/img_align_celeba/`
    *   `data/places_raw/val_256/`

### Training
The model learns by splitting images into **L (Lightness)** and **ab (Color)** channels. It takes 'L' as input and tries to predict 'ab'.
1.  Go to the **"Train"** tab in the Web UI.
2.  Select your parameters (Epochs, Batch Size).
3.  Click **"Start Training"**.
4.  Checkpoints are saved automatically to the `checkpoints/` folder.

### Inference
1.  Go to the **"Colorize"** tab.
2.  Select a trained **Checkpoint** from the dropdown.
3.  Upload a **Grayscale** (or color) image.
4.  Click **"Colorize"**. The model will predict colors for the image.

---

## 📂 Outputs

*   **`checkpoints/`**: Stores trained model weights (`.pth` files). *Note: Large files are not included in this repo.*
*   **`outputs/samples/`**: During training, the model saves sample comparisons (Gray Input vs. Model Output vs. Ground Truth) here so you can visualize progress.
*   **`outputs/`**: General inference results are saved here.

---

## 📁 Project Structure

```
image-colorizer/
├── src/
│   ├── app.py              # Main Entry point (Web UI)
│   ├── dataset.py          # PyTorch Dataset class (Places365/CelebA)
│   ├── model.py            # U-Net Architecture definition
│   ├── loss.py             # Custom VGG16 Perceptual Loss
│   ├── train.py            # Training loop script
│   ├── inference.py        # Logic for colorizing single images
│   ├── download_data.py    # Helper script to fetch datasets
│   └── utils.py            # Helper functions for stats/logging
├── data/                   # (Ignored) Raw dataset images
├── checkpoints/            # (Ignored) Saved model weights
├── outputs/                # Generated colorized images
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
