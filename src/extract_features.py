import argparse
from nsdk.media import NImage, NPixelFormat
from nsdk.biometrics import FingerEngine
from pynsdk.licensing import NLicense, NLicenseManager
import cv2
import numpy as np
import os



def main():
    rad = np.pi / 180.
    # 1. Setup License & Engine (ONCE)
    NLicenseManager.set_trial_mode(True)
    if not NLicense.obtain("/local", 5000, "FingerExtractor"):
        print("Could not obtain license!")
        return
    
    engine = FingerEngine()
    dataset_path = "/media/rgabhi/ABHINAV2/SIL7175/dataset/doi_10_5061_dryad_612jm649q__v20231226/DS1/"

    # 2. Loop through all files
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            # skip raw
            if "raw" in root:
                continue

            # process only specific .bmp files
            if file.endswith(".bmp") and "HT" not in file and "R414" not in file:
                image_path = os.path.join(root, file)
                print(f"Processing: {image_path}")
                # print(f"{root}, {dirs}")
                # break
                try:
                    # Load and Extract
                    nimage = NImage(image_path)
                    status, finger_templates = engine.extract_finger(nimage)
                    
                    if status != "succeeded" or len(finger_templates) == 0:
                        print(f"  - Extraction failed for {file}")
                        continue

                    # Get the list of minutiae
                    minutiae_list = finger_templates[0].fingers.records[0].minutiae

                    # 1. Construct the new filename (e.g., replace .bmp with .txt)
                    txt_path = image_path.replace(".bmp", ".txt")

                    # 2. Open that file and write "x, y, angle" for each minutia
                    with open(txt_path, "w") as f:
                            for minutiae in enumerate(minutiae_list):
                                # x y angle
                                f.write(f"{minutiae.x} {minutiae.y} {minutiae.angle}\n")

                    print(f"  - Extracted {len(minutiae_list)} minutiae to {txt_path}")
                except Exception as e:
                    print(f"  - Error processing {file}: {e}")


if __name__ == "__main__":
    main()