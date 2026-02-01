import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from PIL import Image
import torch

class LABColorizationDataset(Dataset):
    def __init__(self, image_paths, size=128, split="train"):
        self.image_paths = image_paths
        self.size = size
        self.split = split
        self.transform = A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5 if split == "train" else 0.0),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = str(self.image_paths[idx])
        # Read image
        img = cv2.imread(path)
        if img is None:
            # Handle bad images by skipping or returning a zero image (simple fallback)
            # In a real pipeline, we might want to filter these out beforehand.
            # For now, let's create a black image of correct size to avoid crashing
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Albumentations expects RGB
        
        # Augment/Resize
        augmented = self.transform(image=img)
        img = augmented["image"]
        
        # Convert to LAB
        # OpenCV expects [0, 255] for 8-bit, 
        # LAB: L [0, 255], a [0, 255], b [0, 255] if converted from uint8
        # But cv2.cvtColor(..., cv2.COLOR_RGB2LAB) with uint8 returns:
        # L: [0, 255] -> scaled from 0-100 to 0-255 (L_cv = L_std * 2.55)
        # a: [0, 255] -> offset by 128 (a_cv = a_std + 128)
        # b: [0, 255] -> offset by 128 (b_cv = b_std + 128)
        
        # However, it is often easier to work with float32 [0,1] input for pytorch
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        L = img_lab[:, :, 0]
        ab = img_lab[:, :, 1:]
        
        # Normalize
        # L from [0, 255] to [0, 1]
        L = L / 255.0
        
        # ab from [0, 255] to [-1, 1] (approximately)
        # In OpenCV uint8: 0 is -128, 128 is 0, 255 is +127
        # So (val - 128) / 128.0 gives [-1, 1] range
        ab = (ab - 128.0) / 128.0
        
        # To Tensor: (H, W, C) -> (C, H, W)
        L = torch.from_numpy(L).unsqueeze(0).float() # (1, H, W)
        ab = torch.from_numpy(ab.transpose(2, 0, 1)).float() # (2, H, W)
        
        return L, ab

def make_dataloaders(batch_size=16, size=128, num_workers=2, pin_memory=True, max_samples=None):
    from glob import glob
    from sklearn.model_selection import train_test_split
    import random
    
    # Gather paths
    places_dir = Path("data/places_raw")
    celeba_dir = Path("data/celeba_raw")
    
    # Recursive search for images
    places_imgs = list(places_dir.rglob("*.jpg"))
    celeba_imgs = list(celeba_dir.rglob("*.jpg"))
    
    all_imgs = places_imgs + celeba_imgs
    
    if len(all_imgs) == 0:
        print("Warning: No images found in data/places_raw or data/celeba_raw.")
        return None, None
        
    
    # Shuffle to ensure diversity
    random.shuffle(all_imgs)
    
    if max_samples is not None and max_samples > 0:
        if len(all_imgs) > max_samples:
            print(f"Capping dataset at {max_samples} images (found {len(all_imgs)}).")
            all_imgs = all_imgs[:max_samples]

    # Split
    train_paths, val_paths = train_test_split(all_imgs, test_size=0.1, random_state=42)
    
    train_ds = LABColorizationDataset(train_paths, size=size, split="train")
    val_ds = LABColorizationDataset(val_paths, size=size, split="val")
    
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader
