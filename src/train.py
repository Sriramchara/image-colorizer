import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import cv2
import numpy as np

from dataset import make_dataloaders
from unet_colorizer import UNetColorizer
from vgg_perceptual import VGGPerceptualLoss

def lab_to_rgb(L, ab):
    """
    Convert a batch of L, ab images to RGB for perceptual loss.
    L: (N, 1, H, W) in [0, 1]
    ab: (N, 2, H, W) in [-1, 1]
    Returns: (N, 3, H, W) in [0, 1]
    """
    # This function is not efficient for large batches if done in pure python loop + numpy
    # But since we need it for VGG loss, we can do it via torch operations or cv2 if needed.
    # However, VGG expects RGB.
    # We can approximate the conversion or use a differentiable approximation if needed, 
    # but standard practice is often just concatenating and converting.
    # Since cv2.cvtColor is CPU only and non-differentiable, we MUST use a differentiable method 
    # or just assume the model learns to output valid AB that matches the RGB ground truth features.
    
    # Wait, if we use cv2 in the training loop it breaks the gradient graph.
    # So we CANNOT use cv2 for the perceptual loss on the PREDICTED image if we want gradients 
    # to flow back through the colorizer.
    
    # We need a differentiable LAB -> RGB conversion in PyTorch.
    # Or we can just compute perceptual loss on the predicted AB concatenated with Input L vs Target RGB?
    # Actually, the user asked for "Perceptual loss (VGG feature space)".
    # VGG takes RGB.
    # So we need L (input) + AB (pred) -> RGB (pred) in a differentiable way.
    
    # Simplified Differentiable LAB->RGB (approximate or precise)
    # PyTorch doesn't have built-in lab2rgb.
    # But Kornia does. If the user doesn't have kornia, we can write a small helper.
    # Let's write a small torch-based conversion.
    
    # L: [0, 100], a: [-128, 127], b: [-128, 127] standard
    # Our L: [0, 1], ab: [-1, 1]
    
    L_unnorm = L * 100.0
    ab_unnorm = ab * 128.0
    
    # LAB to XYZ
    # ref: https://github.com/richzhang/colorization-pytorch/blob/master/util/util.py or similar
    
    # Check if we can use a library or just implement approximation.
    # For simplicity, and since `kornia` wasn't in the list, 
    # let's assume we can use a simplified linear approximation or we just rely on L1 for now 
    # OR we try to implement the conversion.
    
    # Actually, for this assignment level, maybe we can just use the user's provided snippet?
    # User provided snippet:
    # def lab_to_rgb(L, ab): ... (uses cv2, numpy)
    # AND "loss_perc = perceptual(pred_rgb, target_rgb)"
    # ERROR on user's part: cv2 breaks gradients.
    # User's snippet does L.cpu().numpy() ... this DETACHES gradients.
    # Use of perceptual loss on DETACHED images means the model won't learn 
    # from the perceptual loss! It will only learn from L1 pixel loss.
    
    # I should fix this for the user or warn them. I will fix it by implementing a 
    # differentiable conversion.
    
    # Let's use a known differentiable conversion implementation.
    # Or better yet, maybe just install `kornia` if I can? 
    # But requirements.txt is already written.
    
    # I'll implement a basic differentiable LAB->RGB.
    
    # Constants
    # Conversion logic adapted from Kornia/skimage
    
    # Just implementing a very rough differentiable approximation for now to avoid huge complexity
    # or just skipping perceptual loss backprop if I can't do it quickly?
    # No, the user explicitly asked for it. 
    
    # Let's add kornia to the imports if available, otherwise fallback (but warn).
    # Actually, I'll add the conversion logic.
    return lab_to_rgb_torch(L, ab)

def lab_to_rgb_torch(L, ab):
    """
    Differentiable LAB to RGB conversion in PyTorch.
    L: (N, 1, H, W) [0, 1]
    ab: (N, 2, H, W) [-1, 1]
    Returns: (N, 3, H, W) [0, 1]
    """
    # Scale to standard LAB ranges
    L = L * 100.0
    a = ab[:, 0:1, :, :] * 128.0
    b = ab[:, 1:2, :, :] * 128.0
    
    # LAB to XYZ
    fy = (L + 16.0) / 116.0
    fx = (a / 500.0) + fy
    fz = fy - (b / 200.0)
    
    fx3 = fx ** 3
    fy3 = fy ** 3
    fz3 = fz ** 3
    
    # epsilon = 0.008856
    # kappa = 903.3
    epsilon = 0.008856
    kappa = 903.3
    
    X_n = 95.047
    Y_n = 100.000
    Z_n = 108.883
    
    x_out = torch.where(fx3 > epsilon, fx3, (116.0 * fx - 16.0) / kappa)
    y_out = torch.where(L > (kappa * epsilon), ((L + 16.0) / 116.0) ** 3, L / kappa)
    z_out = torch.where(fz3 > epsilon, fz3, (116.0 * fz - 16.0) / kappa)
    
    X = x_out * X_n
    Y = y_out * Y_n
    Z = z_out * Z_n
    
    # XYZ to RGB
    # sRGB D65
    Rr =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    Gg = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    Bb =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    
    rgb = torch.cat([Rr, Gg, Bb], dim=1) / 255.0
    
    # Clip
    rgb = torch.clamp(rgb, 0.0, 1.0)
    
    return rgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8) # Lower for GTX 1650
    parser.add_argument("--lr", type=float, default=2e-4)    
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--lambda_p", type=float, default=0.01) # Perceptual loss weight
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images (0/None = Full scale)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup Data
    train_loader, val_loader = make_dataloaders(
        batch_size=args.batch_size, 
        size=args.size,
        max_samples=args.limit
    )
    
    if train_loader is None:
        print("No data found. Please download datasets first.")
        return

    # Setup Model
    model = UNetColorizer().to(device)
    
    # Setup Loss
    pixel_loss = nn.L1Loss()
    perceptual_loss = VGGPerceptualLoss(device=device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler = GradScaler()
    
    # Checkpoints dir
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    
    print("Starting training...")
    
    start_epoch = 1
    # Auto-resume logic
    if args.resume or True: # Default to auto-resume for user convenience
        ckpts = list(ckpt_dir.glob("epoch_*.pth"))
        if ckpts:
            # Sort by epoch number
            ckpts.sort(key=lambda x: int(x.stem.split('_')[1]))
            latest_ckpt = ckpts[-1]
            try:
                state = torch.load(latest_ckpt, map_location=device)
                model.load_state_dict(state)
                start_epoch = int(latest_ckpt.stem.split('_')[1]) + 1
                print(f"Resuming from checkpoint: {latest_ckpt} (Starting Epoch {start_epoch})")
            except Exception as e:
                print(f"Failed to load checkpoint {latest_ckpt}: {e}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        train_loss = 0.0
        
        for L, ab in loop:
            L = L.to(device)
            ab = ab.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                pred_ab = model(L) # [-1, 1]
                
                # Pixel Loss
                loss_pix = pixel_loss(pred_ab, ab)
                
                # Perceptual Loss
                # Convert Lab -> RGB (Differentiable)
                pred_rgb = lab_to_rgb(L, pred_ab)
                target_rgb = lab_to_rgb(L, ab) # Ground truth RGB
                
                loss_perc = perceptual_loss(pred_rgb, target_rgb)
                
                loss = loss_pix + (args.lambda_p * loss_perc)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item()) # pix=loss_pix.item(), perc=loss_perc.item())
        
        # Save checkpoint
        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pth")
        print(f"Epoch {epoch} complete. Avg Loss: {train_loss / len(train_loader):.4f}")
        
        # Validation (Visual) - Save one sample
        if epoch % 5 == 0 or epoch == 1:
            validate_and_save(model, val_loader, device, epoch)

def validate_and_save(model, val_loader, device, epoch):
    model.eval()
    with torch.no_grad():
        L, ab = next(iter(val_loader))
        L = L.to(device)
        ab = ab.to(device) # unused for pred but used for GT if we want
        
        pred_ab = model(L)
        
        # Convert first image to RGB
        L_disp = L[0] # (1, H, W)
        ab_disp = pred_ab[0] # (2, H, W)
        
        # We can use the differentiable one or just standard cv2 here safely since no grad
        # Let's stick to numpy/cv2 for visualization correctness confirmation
        L_np = L_disp.cpu().numpy().transpose(1, 2, 0) # (H, W, 1) [0,1]
        ab_np = ab_disp.cpu().numpy().transpose(1, 2, 0) # (H, W, 2) [-1,1]
        
        L_np = (L_np * 100.0).astype(np.float32)
        ab_np = (ab_np * 128.0).astype(np.float32)
        
        lab = np.concatenate([L_np, ab_np], axis=2)
        # Clip
        lab[:,:,0] = np.clip(lab[:,:,0], 0, 100)
        lab[:,:,1:] = np.clip(lab[:,:,1:], -128, 127)
        
        lab = lab.astype(np.float32)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) # outputs float if input float? No, usually needs uint8 or proper scale
        # cv2.cvtColor with float32 input [0..100, -128..127] -> usually it expects 0..1 float for some conversions but LAB is special
        # Safer: convert to uint8
        
        # Scaling for uint8 LAB in opencv:
        # L: 0..100 -> 0..255 (multiply by 2.55)
        # a, b: -128..127 -> 0..255 (add 128)
        
        lab_u8 = np.zeros_like(lab, dtype=np.uint8)
        lab_u8[:,:,0] = (lab[:,:,0] * 2.55).astype(np.uint8)
        lab_u8[:,:,1] = (lab[:,:,1] + 128).astype(np.uint8)
        lab_u8[:,:,2] = (lab[:,:,2] + 128).astype(np.uint8)
        
        bgr = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR)
        
        out_path = f"outputs/samples/val_epoch_{epoch}.jpg"
        cv2.imwrite(out_path, bgr)
        # Also save the Grayscale input for comparison
        # cv2.imwrite(f"outputs/samples/val_epoch_{epoch}_gray.jpg", (L_np * 2.55).astype(np.uint8))

if __name__ == "__main__":
    main()
