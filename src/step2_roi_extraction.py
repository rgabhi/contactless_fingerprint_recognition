import sys
import os
import cv2
import numpy as np
from pynsdk.media import NImage
from pynsdk.biometrics import NBiometricEngine, NBiometricOperations, NFinger, NSubject, NBiometricStatus, NFPosition
from pynsdk.licensing import NLicense, NLicenseManager

def init_sdk():
    """Initializes the SDK and returns the Engine object."""
    # 1. Initialize License
    is_trial_mode = True
    NLicenseManager.set_trial_mode(is_trial_mode)
    
    print("Initializing VeriFinger SDK License...")
    license_name = "FingerClient"
    if not NLicense.obtain("/local", 5000, license_name):
        print(f"Failed to obtain license: {license_name}")
        # Fallback to FingerExtractor if Client not available (often bundle)
        if not NLicense.obtain("/local", 5000, "FingerExtractor"):
            sys.exit(1)

    # 2. Setup the Biometric Engine
    engine = NBiometricEngine()
    # We do NOT set fingers_return_binarized_image = True because we want grayscale for Step 3
    engine.fingers_return_binarized_image = False
    return engine

def process_roi(engine, input_path, output_path):
    """Processes a single image using the existing engine instance."""
    
    # --- Check Input ---
    if not os.path.exists(input_path):
        print(f"Error: Input file does not exist: {input_path}")
        return False

    # --- PRE-PROCESSING (CLAHE) ---
    cv_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if cv_img is None:
        print(f"Error: OpenCV could not read image: {input_path}")
        return False

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_cv_img = clahe.apply(cv_img)
    
    # Save temp for SDK
    temp_path = input_path + ".tmp_roi.png"
    cv2.imwrite(temp_path, enhanced_cv_img)

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
        finger.position = NFPosition.nfpUnknown # Let SDK decide
        subject.fingers.add(finger)

        # 5. Perform Segmentation
        # This isolates the finger but keeps the grayscale data
        status = engine.perform_operation(subject, NBiometricOperations.segment)

        if status == NBiometricStatus.ok:
            # subject.fingers[0] is original, subject.fingers[1] is segmented
            cnt = subject.fingers.count
            
            if cnt > 1:
                subject.fingers[1].image.save_to_file(output_path)
                print(f"Segmentation successful. Saved to {output_path}")
            else:
                print("Warning: Operation OK but no segments returned. Saving Original.")
                cv2.imwrite(output_path, enhanced_cv_img)
        else:
            print(f"Segmentation failed with status: '{status}'. Saving pre-enhanced fallback.")
            cv2.imwrite(output_path, enhanced_cv_img)
            
        return True

    except Exception as e:
        print(f"Step 2 Error: {e}")
        return False
        
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Step 2: ROI Extraction (Segmentation)')
    parser.add_argument('--input', required=True, help='Input standardized image')
    parser.add_argument('--output', required=True, help='Output ROI image')
    args_ = parser.parse_args()
    
    eng = init_sdk()
    try:
        process_roi(eng, args_.input, args_.output)
    finally:
        # CRITICAL FIX: Explicitly delete the engine before script exit
        # This prevents the 'invalid_operation' error during garbage collection
        del eng