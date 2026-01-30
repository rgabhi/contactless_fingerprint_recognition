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

# --- CONFIGURATION ---
MAX_MINUTIAE = 80  # Limit to top 80 to prevent Matrix W explosion
MIN_QUALITY = 40   # Minimum SDK quality score

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
    engine.fingers_quality_threshold = MIN_QUALITY
    return engine

def process_minutiae(engine, input_path, output_path):
    # 1. Set Quality Threshold
    engine.fingers_quality_threshold = MIN_QUALITY

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
        
        if status != NBiometricStatus.ok:
            with open(output_path, 'w') as f: json.dump([], f)
            return True

        # 4. Parse Results
        template = subject.template
        if template and template.fingers and template.fingers.records.count > 0:
            nf_record = template.fingers.records[0]
            minutiae_list = nf_record.minutiae
            
            # --- BOUNDARY FILTERING ---
            cv_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            h, w = cv_img.shape
            _, binary_mask = cv2.threshold(cv_img, 1, 255, cv2.THRESH_BINARY)
            
            # Erode mask to ignore edges
            kernel = np.ones((15, 15), np.uint8)
            safe_zone = cv2.erode(binary_mask, kernel, iterations=1)

            extracted_data = []
            for m in minutiae_list:
                mx, my = int(m.x), int(m.y)
                
                # Bounds check
                if mx < 0 or mx >= w or my < 0 or my >= h: continue
                
                # Filter: Is it in the Safe Zone?
                if safe_zone[my, mx] == 0: continue 

                # Retrieve quality
                qual = m.quality if hasattr(m, 'quality') else 0
                if qual < MIN_QUALITY: continue

                theta_radians = (m.angle * 360.0 / 256.0) * (math.pi / 180.0)
                
                extracted_data.append({
                    "x": m.x, 
                    "y": m.y, 
                    "theta": theta_radians,
                    "type": get_type_string(int(m.type)),
                    "quality": qual
                })
            
            # --- CRITICAL FIX: Sort by quality and limit count ---
            # Sort descending by quality
            extracted_data.sort(key=lambda x: x['quality'], reverse=True)
            
            # Keep only top N
            if len(extracted_data) > MAX_MINUTIAE:
                extracted_data = extracted_data[:MAX_MINUTIAE]
                
            # Re-assign IDs after filtering
            for i, m in enumerate(extracted_data):
                m['id'] = i

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