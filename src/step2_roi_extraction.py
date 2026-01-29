import argparse
import sys
import os

from pynsdk.media import NImage
from pynsdk.biometrics import NBiometricEngine, NBiometricOperations, NFinger, NSubject, NBiometricStatus
from pynsdk.licensing import NLicense, NLicenseManager

def main(args):
    # 1. Initialize License
    is_trial_mode = True
    NLicenseManager.set_trial_mode(is_trial_mode)
    
    license_name = "FingerExtractor"
    if not NLicense.obtain("/local", 5000, license_name):
        print(f"Failed to obtain license: {license_name}")
        sys.exit(1)

    # 2. Setup the Biometric Engine
    engine = NBiometricEngine()
    engine.fingers_return_binarized_image = True

    # 3. Load Image
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        sys.exit(1)

    print(f"Loading image: {args.input}")
    image = NImage(file_name=args.input)
    
    # --- CRITICAL FIX: FORCE 500 PPI ---
    # The SDK often misreads JPEG metadata as 72/96 DPI.
    # We must enforce the standard 500 DPI for the algorithm to work.
    image.horz_resolution = 500
    image.vert_resolution = 500
    # -----------------------------------

    # 4. Prepare Subject
    subject = NSubject()
    finger = NFinger()
    finger.image = image
    subject.fingers.add(finger)

    # 5. Extract
    print("Extracting ROI and features...")
    status = engine.perform_operation(subject, NBiometricOperations.create_template)

    if status != NBiometricStatus.ok:
        print(f"Extraction failed with status: '{status}'")
        # Exit with error so Step 8 knows it failed
        sys.exit(1)
    
    # 6. Save ROI
    if finger.binarized_image:
        print(f"Saving ROI to: {args.output}")
        finger.binarized_image.save_to_file(args.output)
    else:
        print("Error: No binarized image returned.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 2: ROI Extraction')
    parser.add_argument('--input', required=True, help='Input standardized image')
    parser.add_argument('--output', required=True, help='Output ROI image')
    args_ = parser.parse_args()
    main(args_)