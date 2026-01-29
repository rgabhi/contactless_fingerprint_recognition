import argparse
import json
import os
import sys
import math
import cv2
import numpy as np

# Import NSDK modules
from pynsdk.media import NImage
from pynsdk.biometrics import NBiometricEngine, NBiometricOperations, NFinger, NSubject, NBiometricStatus
from pynsdk.licensing import NLicense, NLicenseManager

def get_type_string(type_int):
    if type_int == 1:
        return "ending"
    elif type_int == 2:
        return "bifurcation"
    else:
        return str(type_int)

def main(args):
    # 1. Initialize License
    is_trial_mode = True
    NLicenseManager.set_trial_mode(is_trial_mode)
    
    license_name = "FingerExtractor"
    if not NLicense.obtain("/local", 5000, license_name):
        print(f"Failed to obtain license: {license_name}")
        sys.exit(1)
    
    # 2. Initialize Engine
    engine = NBiometricEngine()
    
    # --- FIX 1: Set Quality Threshold ---
    # Default is often 0 (accept everything). 
    # Setting to 40-50 filters out "weak" minutiae found in noise.
    engine.fingers_quality_threshold = 40
    # ------------------------------------
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)
        
    print(f"Extracting minutiae from: {args.input}")
    
    # 3. Load Image 
    nimage = NImage(args.input)
    
    # 4. Prepare Subject
    subject = NSubject()
    finger = NFinger()
    finger.image = nimage
    subject.fingers.add(finger)

    # 5. Perform Extraction
    status = engine.perform_operation(subject, NBiometricOperations.create_template)
    
    if status != NBiometricStatus.ok:
        print(f"Extraction failed with status: {status}")
        sys.exit(1)

    # 6. Retrieve Minutiae from Template
    try:
        template = subject.template
        if template and template.fingers and template.fingers.records.count > 0:
            nf_record = template.fingers.records[0]
            minutiae_list = nf_record.minutiae
            
            # --- FIX 2: Boundary Filtering Setup ---
            # Load image with OpenCV to create an erosion mask
            # We want to ignore minutiae within X pixels of the black background
            cv_img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
            h, w = cv_img.shape
            
            # Create binary mask (Finger = 255, Background = 0)
            _, binary_mask = cv2.threshold(cv_img, 1, 255, cv2.THRESH_BINARY)
            
            # Erode the mask to create a "Safe Zone"
            # Any minutia found in the eroded part (the edge) is discarded
            erosion_size = 15 # Reject features within 15 pixels of edge
            kernel = np.ones((erosion_size, erosion_size), np.uint8)
            safe_zone = cv2.erode(binary_mask, kernel, iterations=1)
            # ---------------------------------------

            extracted_data = []
            original_count = len(minutiae_list)
            
            for idx, m in enumerate(minutiae_list):
                # Check 1: Is the point inside the Safe Zone?
                # Coordinates might need int casting
                mx, my = int(m.x), int(m.y)
                
                # Bounds check
                if mx < 0 or mx >= w or my < 0 or my >= h:
                    continue
                    
                # If safe_zone is 0 at this pixel, it's too close to the edge -> Skip
                if safe_zone[my, mx] == 0:
                    continue

                # Check 2: Quality (if exposed by wrapper, otherwise Engine handles it)
                # Some wrappers expose m.quality. If not, the engine threshold above handles it.
                if hasattr(m, 'quality') and m.quality < 40:
                    continue

                # Convert VeriFinger angle (0-255) to radians
                theta_radians = (m.angle * 360.0 / 256.0) * (math.pi / 180.0)
                
                type_val = int(m.type)
                type_str = get_type_string(type_val)

                minutia_point = {
                    "id": len(extracted_data), # Re-index
                    "x": m.x,
                    "y": m.y,
                    "theta": theta_radians, 
                    "raw_angle": m.angle,
                    "type": type_str,
                    "type_id": type_val,
                    "quality": m.quality if hasattr(m, 'quality') else 0
                }
                extracted_data.append(minutia_point)
                
            print(f"Extraction Complete. Filtered {original_count} -> {len(extracted_data)} minutiae.")
            
            with open(args.output, 'w') as f:
                json.dump(extracted_data, f, indent=4)
                
            print(f"Saved minutiae set M to: {args.output}")
        else:
            print("Extraction successful, but no fingerprint records found in template.")

    except Exception as e:
        print(f"Error parsing template: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 4: Minutiae Extraction')
    parser.add_argument('--input', type=str, required=True, help='Path to enhanced image (from Step 3)')
    parser.add_argument('--output', type=str, required=True, help='Path to save minutiae JSON')
    args_ = parser.parse_args()
    main(args_)