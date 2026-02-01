import torchvision
from torchvision.datasets import Places365, CelebA
import os

def download_datasets():
    print("Downloading Places365 (small, 256x256)...")
    # This will download to data/places_raw using the built-in torchvision support
    try:
        Places365(root="data/places_raw", split="train-standard", small=True, download=True)
        print("Places365 download complete.")
    except Exception as e:
        print(f"Error downloading Places365: {e}")
        print("You may need to download manually from http://places2.csail.mit.edu/download.html")

    print("\nAttempting to check/download CelebA...")
    # CelebA often has daily quota limits on Google Drive for automatic download.
    # If this fails, the user must download manually from Kaggle or official site.
    try:
        CelebA(root="data/celeba_raw", split="train", download=True)
        print("CelebA download complete (or already exists).")
    except Exception as e:
        print(f"Error downloading CelebA: {e}")
        print("Please download CelebA manually (e.g., from Kaggle) and extract to data/celeba_raw/celeba.")

if __name__ == "__main__":
    os.makedirs("data/places_raw", exist_ok=True)
    os.makedirs("data/celeba_raw", exist_ok=True)
    download_datasets()
