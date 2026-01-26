import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- PART A: The CNN Architecture (U-Net) ---
class FingerprintUNet(nn.Module):
    def __init__(self):
        super(FingerprintUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(64, 128)
        
        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(64 + 32, 32)
        
        # Output
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        u2 = self.up2(b)
        # Handle slight shape mismatch due to pooling odd dims
        if u2.size() != e2.size():
            u2 = F.interpolate(u2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(d2)
        if u1.size() != e1.size():
            u1 = F.interpolate(u1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        return torch.sigmoid(self.final(d1))

# --- PART B: Inference & Fallback ---
def segment_fingerprint(image_numpy, model_path=None, use_fallback=True):
    """
    Args:
        image_numpy: Grayscale image (H, W)
        model_path: Path to .pth file.
        use_fallback: If True, uses variance segmentation when model fails/is missing.
    Returns:
        segmented_image (image with background set to 0), mask (binary)
    """
    h, w = image_numpy.shape
    
    # 1. Try CNN Inference
    if model_path:
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = FingerprintUNet().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            # Preprocess
            img_tensor = torch.from_numpy(image_numpy).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, H, W)
            
            with torch.no_grad():
                output = model(img_tensor)
                
            mask = output.squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8) * 255
            
            # Post-process mask (remove noise)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return cv2.bitwise_and(image_numpy, image_numpy, mask=mask), mask
            
        except Exception as e:
            print(f"CNN Segmentation failed ({e}). Switching to fallback...")

    # 2. Fallback: Variance-Based Segmentation (Standard non-learning method)
    if use_fallback:
        # Calculate local variance
        img_float = image_numpy.astype(np.float32)
        mu = cv2.blur(img_float, (16, 16))
        mu2 = cv2.blur(img_float * img_float, (16, 16))
        variance = mu2 - mu * mu
        
        # Normalize and threshold
        variance = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX)
        _, mask = cv2.threshold(variance.astype(np.uint8), 25, 255, cv2.THRESH_BINARY)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask
        segmented = cv2.bitwise_and(image_numpy, image_numpy, mask=mask)
        return segmented, mask

    return image_numpy, np.ones_like(image_numpy) * 255