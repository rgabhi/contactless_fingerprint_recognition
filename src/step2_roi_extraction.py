import argparse
import sys
import os
import cv2
import numpy as np

# Import NSDK modules
from pynsdk.media import NImage
from pynsdk.biometrics import NBiometricEngine, NBiometricOperations, NFinger, NSubject, NBiometricStatus, NFPosition
from pynsdk.licensing import NLicense, NLicenseManager

def main(args):
    # 1. Initialize License
    is_trial_mode = True
    NLicenseManager.set_trial_mode(is_trial_mode)
    
    license_name = "FingerClient" # Changed from FingerExtractor to FingerClient for Segmentation
    if not NLicense.obtain("/local", 5000, license_name):
        print(f"Failed to obtain license: {license_name}")
        # Fallback to FingerExtractor if Client not available (often bundle)
        if not NLicense.obtain("/local", 5000, "FingerExtractor"):
            sys.exit(1)

    # 2. Setup the Biometric Engine
    engine = NBiometricEngine()
    # We do NOT set fingers_return_binarized_image = True because we want grayscale for Step 3

    # 3. Load and Pre-process Image (CRITICAL FOR DARK IMAGES)
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        sys.exit(1)

    print(f"Loading image: {args.input}")
    
    # --- PRE-PROCESSING START ---
    # The SDK fails on dark images ('bad_object'). We manually boost contrast first.
    cv_img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if cv_img is None:
        print("Error: OpenCV failed to load image.")
        sys.exit(1)
        
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This reveals the finger in the dark background
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_cv_img = clahe.apply(cv_img)
    
    # Save to temp file for SDK
    temp_path = args.input + ".tmp.png"
    cv2.imwrite(temp_path, enhanced_cv_img)
    # --- PRE-PROCESSING END ---

    try:
        # Load the pre-processed image into NImage
        nimage = NImage(file_name=temp_path)
        
        # Force 500 PPI (Critical for SDK logic)
        nimage.horz_resolution = 500
        nimage.vert_resolution = 500

        # 4. Prepare Subject
        subject = NSubject()
        finger = NFinger()
        finger.image = nimage
        finger.position = NFPosition.nfpUnknown # Let SDK decide, or use nfpRightThumb if known
        subject.fingers.add(finger)

        # 5. Perform Segmentation
        # This isolates the finger but keeps the grayscale data
        print("Performing Segmentation...")
        status = engine.perform_operation(subject, NBiometricOperations.segment)

        if status == NBiometricStatus.ok:
            # According to SDK docs/sample:
            # subject.fingers[0] is the original
            # subject.fingers[1...N] are the segmented fingers
            
            cnt = subject.fingers.count
            print(f"Segmentation successful. Found {cnt - 1} fingers.")
            
            if cnt > 1:
                # Save the first segmented finger found
                # We save to args.output (e.g., ..._roi.jpg)
                # The image inside NFinger is usually PNG/BMP buffer, we save to file
                subject.fingers[1].image.save_to_file(args.output)
                print(f"Saved ROI to: {args.output}")
            else:
                print("Warning: Operation OK but no segments returned. Saving Original.")
                # Fallback: Save the enhanced original if segmentation "passed" but returned nothing
                cv2.imwrite(args.output, enhanced_cv_img)
        else:
            print(f"Segmentation failed with status: '{status}'")
            # If 'bad_object' or 'too_few_features', we should output the cropped/enhanced 
            # original so the pipeline doesn't break, or exit.
            # Given the difficulty of data, let's save the enhanced image as fallback
            # so Step 3 has *something* to work with.
            print("Fallback: Saving pre-enhanced image as ROI.")
            cv2.imwrite(args.output, enhanced_cv_img)
            # Do not exit(1) here to allow batch process to continue
            
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 2: ROI Extraction (Segmentation)')
    parser.add_argument('--input', required=True, help='Input standardized image')
    parser.add_argument('--output', required=True, help='Output ROI image')
    args_ = parser.parse_args()
    main(args_)