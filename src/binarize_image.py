import os
from pynsdk.media import NImage
from pynsdk.biometrics import NBiometricEngine, NBiometricOperations, NFinger, NSubject, NBiometricStatus
from pynsdk.licensing import NLicense, NLicenseManager


def main():
    # 1. Setup License & Engine (ONCE)
    NLicenseManager.set_trial_mode(True)
    if not NLicense.obtain("/local", 5000, "FingerExtractor"):
        print("Could not obtain license!")
        return
    
    # Initialize the engine
    engine = NBiometricEngine()
    # Crucial Step: Tell the engine we want the binarized image back
    engine.fingers_return_binarized_image = True
    
    # Update this path to your specific dataset folder
    dataset_path = "/media/rgabhi/ABHINAV2/SIL7175/dataset/doi_10_5061_dryad_612jm649q__v20231226/DS1/"

    # 2. Loop through all files
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            # Skip 'raw' folder
            if "raw" in root:
                continue

            # Process only specific .bmp files (skip HT, R414, and hidden files)
            if file.endswith(".bmp") and "HT" not in file and "R414" not in file and not file.startswith("._"):
                image_path = os.path.join(root, file)
                print(f"Processing: {image_path}")

                try:
                    # Load the image
                    image = NImage(image_path)
                    
                    # --- FORCE RESOLUTION (The Fix) ---
                    image.horz_resolution = 500
                    image.vert_resolution = 500

                    # Create Subject and Finger objects for the SDK
                    subject = NSubject()
                    finger = NFinger()
                    finger.image = image
                    subject.fingers.add(finger)

                    # Run the extraction (this generates the binarized image internally)
                    status = engine.perform_operation(subject, NBiometricOperations.create_template)

                    if status != NBiometricStatus.ok:
                        print(f"  - Failed with status: {status}")
                        continue

                    # Save the binarized image
                    # We'll save it as 'OriginalName_bin.bmp'
                    save_path = image_path.replace(".bmp", "_bin.bmp")
                    
                    if finger.binarized_image:
                        finger.binarized_image.save_to_file(save_path)
                        print(f"  -> Saved binarized image to {save_path}")
                    else:
                        print("  - No binarized image returned.")

                except Exception as e:
                    print(f"  - Error processing {file}: {e}")

if __name__ == "__main__":
    main()