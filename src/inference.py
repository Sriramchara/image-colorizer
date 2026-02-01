import torch
import cv2
import numpy as np
import argparse
import os
from unet_colorizer import UNetColorizer

def load_model(model_path, device="cuda"):
    model = UNetColorizer().to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def colorize_image(img, model, device="cuda"):
    """
    img: cv2 image (BGR) or numpy array
    model: loaded UNetColorizer
    Returns: colorized image (BGR)
    """
    if img is None:
        return None
        
    h, w = img.shape[:2]
    
    # Preprocess
    inp_size = 128
    img_resized = cv2.resize(img, (inp_size, inp_size))
    
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    
    # Normalize L
    L_norm = L / 255.0 # [0, 1]
    
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        pred_ab = model(L_tensor)
        
    pred_ab = pred_ab.squeeze(0).cpu().numpy().transpose(1, 2, 0) # (128, 128, 2)
    
    # Un-normalize ab
    # Un-normalize ab happens later
    
    # Combine with L
    pred_ab_orig = cv2.resize(pred_ab, (w, h))
    
    # Get original L
    lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L_orig = lab_orig[:, :, 0]
    
    out_lab = np.zeros((h, w, 3), dtype=np.float32)
    
    # Our L_orig is uint8 0..255. Convert to 0..100 for conversion coherence if needed
    # But for final BGR conversion, we can stick to standard ranges if careful.
    
    # Re-assembling 
    # L [0..255] works for cv2 8-bit
    # ab [-128..127] needs to be adapted for 8-bit LAB in cv2 (a+128, b+128)
    
    # Let's do it manually for 8-bit safety
    out_lab_u8 = np.zeros((h, w, 3), dtype=np.uint8)
    out_lab_u8[:, :, 0] = L_orig # [0, 255]
    
    # Clip and shift ab
    a_u8 = np.clip(pred_ab_orig[:, :, 0] * 128.0 + 128.0, 0, 255).astype(np.uint8)
    b_u8 = np.clip(pred_ab_orig[:, :, 1] * 128.0 + 128.0, 0, 255).astype(np.uint8)
    
    out_lab_u8[:, :, 1] = a_u8
    out_lab_u8[:, :, 2] = b_u8
    
    out_bgr = cv2.cvtColor(out_lab_u8, cv2.COLOR_LAB2BGR)
    return out_bgr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to input grayscale image")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--out", type=str, default="output.jpg")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    try:
        model = load_model(args.ckpt, args.device)
        img = cv2.imread(args.img)
        if img is None:
            print(f"Error reading image {args.img}")
            exit(1)
            
        res = colorize_image(img, model, args.device)
        cv2.imwrite(args.out, res)
        print(f"Saved colorized image to {args.out}")
    except Exception as e:
        print(f"Error: {e}")
