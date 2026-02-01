import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as T

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=("features.3", "features.8", "features.15", "features.22"), device="cuda"):
        super().__init__()
        # Load VGG16 with default pre-trained weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        
        # We don't need gradients for VGG
        for p in vgg.parameters():
            p.requires_grad = False
            
        self.vgg = vgg.to(device).eval()
        self.layers = {}
        
        # Parse layer indices
        # features.3 -> relu1_2
        # features.8 -> relu2_2
        # features.15 -> relu3_3
        # features.22 -> relu4_3
        for l in layers:
            idx = int(l.split(".")[1])
            self.layers[idx] = l
            
        self.max_layer_idx = max(self.layers.keys())
        self.criterion = nn.L1Loss()
        
        # ImageNet normalization
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def forward(self, pred_rgb, target_rgb):
        # pred_rgb, target_rgb: [N, 3, H, W] in [0, 1]
        
        loss = 0.0
        x = self.normalize(pred_rgb)
        y = self.normalize(target_rgb)
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            
            if i in self.layers:
                loss += self.criterion(x, y)
            
            if i >= self.max_layer_idx:
                break
                
        return loss
