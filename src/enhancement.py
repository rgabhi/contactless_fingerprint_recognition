import cv2
import numpy as np

def estimate_illumination(image, kernel_size=31):
    """
    Estimates the illumination component (L) using a low-pass filter (Gaussian).
    The kernel size should be roughly the scale of the lighting variation.
    """
    # Large Gaussian Blur acts as an estimation of the smooth illumination
    illumination = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return illumination

def intrinsic_decomposition(image):
    """
    Decomposes Image (I) into Reflectance (R) and Illumination (L).
    I = R * L  =>  R = I / L
    """
    # Convert to float for division
    I = image.astype(np.float32) + 1.0 # Add 1 to avoid div by zero
    
    # 1. Estimate Illumination
    L = estimate_illumination(image)
    L = L.astype(np.float32) + 1.0
    
    # 2. Recover Reflectance (R)
    # R = I / L
    R = np.divide(I, L)
    
    # Normalize R back to 0-255
    R = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX)
    return R.astype(np.uint8)

def guided_filtering(image, radius=8, eps=100):
    """
    Applies Guided Filtering to smooth noise while preserving edges.
    """
    # In this context, the image guides itself
    # OpenCV requires contrib modules or ximgproc for guidedFilter in some versions,
    # but the standard cv2.ximgproc usually contains it.
    try:
        from cv2.ximgproc import guidedFilter
        enhanced = guidedFilter(guide=image, src=image, radius=radius, eps=eps)
    except ImportError:
        # Fallback if ximgproc is not installed: Bilateral Filter (similar edge-preserving property)
        print("Warning: cv2.ximgproc not found. Using Bilateral Filter as fallback.")
        enhanced = cv2.bilateralFilter(image, d=radius, sigmaColor=eps, sigmaSpace=radius)
        
    return enhanced

def enhance_fingerprint(image):
    """
    Full Enhancement Pipeline:
    1. Intrinsic Decomposition (Get Reflectance)
    2. Guided Filtering (Denoise Reflectance)
    """
    # Step 1: Intrinsic Decomposition
    reflectance = intrinsic_decomposition(image)
    
    # Step 2: Guided Filtering
    # The assignment says "Apply guided filtering to the reflectance image" [cite: 30]
    enhanced = guided_filtering(reflectance)
    
    # Optional: Adaptive Histogram Equalization (CLAHE) to boost local contrast further
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    final_output = clahe.apply(enhanced)
    
    return final_output