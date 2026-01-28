import argparse
import sys

from pynsdk.media import NImage
from pynsdk.biometrics import NBiometricEngine, NBiometricOperations, NFinger, NSubject, NBiometricStatus
from pynsdk.licensing import NLicense, NLicenseManager

def main(args):
    #init license
    is_trial_mode = True
    NLicenseManager.set_trial_mode(is_trial_mode)
    print(f"Trial mode: {is_trial_mode}")

    # Obtain 'FingerExtractor' license which allows segmentation/extraction
    license_name = "FingerExtractor"
    if not NLicense.obtain("/local", 5000, license_name):
        print(f"Failed to obtain license: {license_name}")
        return
    print(f"License obtained successfully: {license_name}")


    # 2. Setup the Biometric Engine
    engine = NBiometricEngine()
    
    # CRITICAL: This flag tells the engine to return the processed (segmented) image
    # The binarized image effectively isolates the ROI (ridges) from the background.
    engine.fingers_return_binarized_image = True

    # You can also request the processed image (enhanced gray-scale) if preferred:
    # engine.fingers_return_processed_image = True


    # 3. Load the Standardized Image (Output from Step 1)
    print(f"Loading image: {args.input}")
    image = NImage(file_name=args.input)
    
    # 4. prepare subject and finger
    subject = NSubject()
    finger = NFinger()
    finger.image = image
    subject.fingers.add(finger)

    #5. extraction
    print("Extracting ROI and features...")
    status = engine.perform_operation(subject, NBiometricOperations.create_template)

    if status != NBiometricStatus.ok:
        print(f"Extraction failed with status: '{status.name}'")
        sys.exit(1)
    
    # 6. Retrieve and Save the ROI (Binarized Image)
    # The binarized image contains the ridges (white/black) with the background removed.
    if finger.binarized_image:
        print(f"Saving ROI (Binarized Image) to: {args.output}")
        finger.binarized_image.save_to_file(args.output)
    else:
        print("Error: No binarized image returned. Ensure the image quality is sufficient.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 2: ROI Extraction using VeriFinger')
    parser.add_argument('--input', type=str, required=True, help='Path to standardized image (from Step 1)')
    parser.add_argument('--output', type=str, required=True, help='Path to save ROI image')
    args_ = parser.parse_args()
    main(args_)
