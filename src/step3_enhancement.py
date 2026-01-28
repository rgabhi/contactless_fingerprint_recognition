import cv2
import numpy as np
import argparse
from PIL import Image


def intrinsic_image_decomposition(img_gray):
   """
    Decomposes image into Reflectance and Illumination.
    Reflectance = Image / Illumination
    """
   # estimate illumination (L) using gaussian blur.
   illumination = cv2.GaussianBlur(img_gray, (51, 51), 0)
   #avoid div by zero
   illumination = np.maximum(illumination, 1.0)

   #get reflectance
   # convert to float for div
   img_float = img_gray.astype(np.float32)
   illumination_float = illumination.astype(np.float32)

   reflectance = img_float/illumination_float

   reflectance = cv2.normalize(reflectance, None, 0, 255, cv2.NORM_MINMAX)
   return reflectance.astype(np.uint8)

def guided_filter_enhancement(image, radius=5, eps=50):
   """
   Applies Guided Filtering to preserve edges (ridges) while smoothing noise.
   """
   #using the image itself as guidance img
   return cv2.ximgproc.guidedFilter(guide=image, src=image, radius=radius, eps=eps)


def enhance_fingerprint(input_path, output_path):
    #load image (grayscale)
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: couldn't read img {input_path}")
        return
    print("Processing: Intrinsic Image Decomposition...")
    reflectance = intrinsic_image_decomposition(img)

    print("Processing: Guided Filtering...")
    enhanced = guided_filter_enhancement(reflectance)

    # Optional: CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    # to further improve contrast as final touch
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    final_output = clahe.apply(enhanced)

    # 3. SAVE WITH PIL TO PRESERVE DPI (Critical Fix)
    # OpenCV's imwrite ignores DPI. We convert to PIL Image to save with metadata.
    pil_image = Image.fromarray(final_output)
    pil_image.save(output_path, dpi=(500, 500))
    print(f"Saved enhanced image to: {output_path} @ 500 DPI")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 3: Image Enhancement')
    parser.add_argument('--input', type=str, required=True, help='Path to ROI image (from Step 2)')
    parser.add_argument('--output', type=str, required=True, help='Path to save enhanced image')
    args = parser.parse_args()

    enhance_fingerprint(args.input, args.output)