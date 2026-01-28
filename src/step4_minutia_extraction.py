import argparse
import json
import os
import sys
import math

# Import NSDK modules
from pynsdk.media import NImage
from pynsdk.biometrics import NBiometricEngine, NBiometricOperations, NFinger, NSubject, NBiometricStatus
from pynsdk.licensing import NLicense, NLicenseManager

def get_type_string(type_int):
    # Standard Neurotechnology NMinutiaType mapping
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
            
            extracted_data = []
            print(f"Found {len(minutiae_list)} minutiae.")
            
            for idx, m in enumerate(minutiae_list):
                # Convert VeriFinger angle (0-255) to radians
                theta_radians = (m.angle * 360.0 / 256.0) * (math.pi / 180.0)
                
                # FIX: m.type is an int, handle it directly
                type_val = int(m.type)
                type_str = get_type_string(type_val)

                minutia_point = {
                    "id": idx,
                    "x": m.x,
                    "y": m.y,
                    "theta": theta_radians, 
                    "raw_angle": m.angle,
                    "type": type_str,
                    "type_id": type_val
                }
                extracted_data.append(minutia_point)
                
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