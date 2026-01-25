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
            if file.endswith(".bmp"):
                image_path = os.path.join(root, file)
                print(f"Processing: {image_path}")

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
                            np_image = nimage.to_numpy(NPixelFormat.rgb_8u)
                            for idx, minutiae in enumerate(minutiae_list):
                                x1 = minutiae.x
                                y1 = minutiae.y
                                angle = minutiae.angle
                                x2 = int(x1 + 15 * np.cos(angle * 1.411764706 * rad))
                                y2 = int(y1 + 15 * np.sin(angle * 1.411764706 * rad))
                                cv2.line(np_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                                cv2.circle(np_image, (x1, y1), 3, (0, 0, 255), thickness=-1, lineType=8, shift=0)
                                (f"{idx} minutiae cordinate ({minutiae.x}, {minutiae.y}) angle {minutiae.angle}")

                    
                except Exception as e:
                    print(f"  - Error processing {file}: {e}")

if __name__ == "__main__":
    main()