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
    if type_int == 1: return "ending"
    elif type_int == 2: return "bifurcation"
    else: return str(type_int)

def init_sdk():
    """Standalone init if run individually."""
    is_trial_mode = True
    NLicenseManager.set_trial_mode(is_trial_mode)
    if not NLicense.obtain("/local", 5000, "FingerExtractor"):
        print("Failed to obtain FingerExtractor license")
        sys.exit(1)
    engine = NBiometricEngine()
    engine.fingers_quality_threshold = 40
    return engine

def process_minutiae(engine, input_path, output_path):
    """
    Processes minutiae extraction using a persistent engine.
    Returns True if successful, False otherwise.
    """
    # 1. Set Quality Threshold (Discard weak features in noise)
    engine.fingers_quality_threshold = 40

    if not os.path.exists(input_path):
        print(f"Error: Input not found {input_path}")
        return False
    
    try:
        # 2. Load Image into SDK
        nimage = NImage(input_path)
        subject = NSubject()
        finger = NFinger()
        finger.image = nimage
        subject.fingers.add(finger)

        # 3. Extract
        status = engine.perform_operation(subject, NBiometricOperations.create_template)
        
        # 4. Handle Empty/Failed Extraction gracefully
        if status != NBiometricStatus.ok:
            # Save empty list so pipeline doesn't break
            with open(output_path, 'w') as f: json.dump([], f)
            return True

        # 5. Parse Results
        template = subject.template
        if template and template.fingers and template.fingers.records.count > 0:
            nf_record = template.fingers.records[0]
            minutiae_list = nf_record.minutiae
            
            # --- BOUNDARY FILTERING (Fix for 147+ minutiae) ---
            # Load image to create "Safe Zone" mask
            cv_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            h, w = cv_img.shape
            _, binary_mask = cv2.threshold(cv_img, 1, 255, cv2.THRESH_BINARY)
            
            # Erode mask by 15px. Minutiae in the eroded region (edges) are skipped.
            kernel = np.ones((15, 15), np.uint8)
            safe_zone = cv2.erode(binary_mask, kernel, iterations=1)
            # --------------------------------------------------

            extracted_data = []
            for m in minutiae_list:
                mx, my = int(m.x), int(m.y)
                
                # Bounds check
                if mx < 0 or mx >= w or my < 0 or my >= h: continue
                
                # Filter: Is it in the Safe Zone?
                if safe_zone[my, mx] == 0: continue 

                # Filter: Quality Check (Redundant but safe)
                if hasattr(m, 'quality') and m.quality < 40: continue

                theta_radians = (m.angle * 360.0 / 256.0) * (math.pi / 180.0)
                
                extracted_data.append({
                    "id": len(extracted_data),
                    "x": m.x, 
                    "y": m.y, 
                    "theta": theta_radians,
                    "type": get_type_string(int(m.type)),
                    "quality": m.quality if hasattr(m, 'quality') else 0
                })
                
            with open(output_path, 'w') as f:
                json.dump(extracted_data, f, indent=4)
            return True
        else:
            with open(output_path, 'w') as f: json.dump([], f)
            return True

    except Exception as e:
        print(f"Step 4 Error: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    eng = init_sdk()
    try:
        process_minutiae(eng, args.input, args.output)
    finally:
        del eng