import cv2
import numpy as np
import argparse
from PIL import Image
import os

def get_finger_mask(img_gray):
    """
    Generates a binary mask for the fingerprint region.
    Uses heavy blurring and Otsu thresholding to find the finger shape,
    then fills holes to create a solid mask.
    """
    # 1. Strong Gaussian Blur to remove noise and unify the finger area
    # A large kernel size (e.g., 25x25 for 500ppi) helps merge ridges into a solid block
    blur = cv2.GaussianBlur(img_gray, (25, 25), 0)
    
    # 2. Otsu threshold to separate foreground (finger) from background
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Morphological closing to fill small holes inside the finger region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return mask

def get_intrinsic_components(img_gray):
    # (Same as previous implementation)
    img_float = img_gray.astype(np.float32) / 255.0
    R = cv2.GaussianBlur(img_float, (0, 0), sigmaX=15, sigmaY=15)
    R = np.maximum(R, 0.001)
    S = img_float / R
    return R, S

def normalize_shading(S, mask):
    """
    Paper Section II.C: Shading Normalization using the provided mask.
    """
    # Extract values only from the masked foreground region
    roi_vals = S[mask > 0]
    
    if len(roi_vals) == 0:
        # Fallback if masking failed completely, just return S
        return S 
    
    m = np.mean(roi_vals)
    v = np.var(roi_vals)
    
    m0 = 0.0
    v0 = 1.0
    if v < 1e-6: v = 1e-6
    
    # Apply Normalization globally
    S_prime = m0 + (S - m) * np.sqrt(v0 / v)
    
    return S_prime

def enhance_fingerprint(input_path, output_path):
    # 1. Load Image
    if not os.path.exists(input_path):
         print(f"Error: Input path does not exist: {input_path}")
         return
         
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: failed to load image data from {input_path}")
        return

    print(f"Processing: {input_path}")

    # --- NEW STEP: Generate Finger Mask ---
    print("Generating background mask...")
    finger_mask = get_finger_mask(img)
    # --------------------------------------

    # 2. Intrinsic Image Decomposition
    R, S = get_intrinsic_components(img)
    
    # 3. Shading Normalization (Passing the mask)
    S_prime = normalize_shading(S, finger_mask)

    # 4. Guided Image Filtering
    print("Applying Guided Filtering (Cross-Guided)...")
    img_float = img.astype(np.float32) / 255.0
    radius = 4
    eps = 0.04
    
    try:
        filtered = cv2.ximgproc.guidedFilter(
            guide=S_prime.astype(np.float32), 
            src=img_float, 
            radius=radius, 
            eps=eps
        )
    except AttributeError:
        filtered = cv2.guidedFilter(
            guide=S_prime.astype(np.float32), 
            src=img_float, 
            radius=radius, 
            eps=eps
        )

    # 5. Post-Processing
    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
    filtered_uint8 = filtered.astype(np.uint8)

    # CLAHE for final local contrast boost
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_output = clahe.apply(filtered_uint8)

    # --- NEW STEP: Apply Final Mask to clean background ---
    print("Applying final background mask...")
    # Bitwise AND keeps pixels where mask is 255, sets others to 0
    final_output_masked = cv2.bitwise_and(enhanced_output, enhanced_output, mask=finger_mask)
    # ------------------------------------------------------

    # 6. Save with Metadata
    pil_image = Image.fromarray(final_output_masked)
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_image.save(output_path, dpi=(500, 500))
    print(f"Saved masked enhanced image to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 3: Image Enhancement via Intrinsic Decomposition & Guided Filtering (with Masking)')
    parser.add_argument('--input', type=str, required=True, help='Path to ROI image')
    parser.add_argument('--output', type=str, required=True, help='Path to save enhanced image')
    args = parser.parse_args()

    enhance_fingerprint(args.input, args.output)