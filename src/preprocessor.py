import os
import glob
from PIL import Image
from pynsdk.licensing import NLicense, NLicenseManager # for trial
from pynsdk.media import NImage
from pynsdk.biometrics import NBiometricEngine, NBiometricOperations, NFinger, NSubject, NBiometricStatus, NFPosition
import cv2
import numpy as np
import json
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from skimage.draw import line
import random


class FingerprintPreprocessor:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        # Initialize the SDK
        self.engine = self.init_sdk()
        self.all_features = {}

    def standardize_image(self, image_path, filename, target_ppi=500):
        try:
            with Image.open(image_path) as img:
                # Get current PPI, default to 1000 if not found
                current_ppi = img.info.get('dpi', (1000, 1000))[0]

                if current_ppi < target_ppi:
                    print(f"Warning: {filename} PPI ({current_ppi}) < target ({target_ppi}). Copying original.")
                    scale_factor = 1
                else:
                    scale_factor = target_ppi / current_ppi
                
                # Calculate new dims
                new_w = int(img.width * scale_factor)
                new_h = int(img.height * scale_factor)
                
                # Resize
                resized_image = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # Save to the processed folder
                save_path = os.path.join(self.output_path, filename)
                
                # Setting dpi in save is good metadata hygiene
                resized_image.save(save_path, dpi=(target_ppi, target_ppi))
                
                print(f"Processed: {filename} | {img.size} -> {resized_image.size}")
                return save_path
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return None

    def process_dataset(self):
        try:
            # Iterate through all subdirectories in the dataset path
            # We assume folders are named '1', '2', etc.
            subject_folders = [f for f in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, f))]
            
            count = 0
            for subject in subject_folders:
                subject_path = os.path.join(self.dataset_path, subject)
                
                # Find all BMP files
                bmp_files = glob.glob(os.path.join(subject_path, "*.bmp"))
                
                for file_path in bmp_files:
                    filename = os.path.basename(file_path)
                    
                    # Filter out the pre-processed files (HT, R414)
                    # We only want the pattern SIRE-SubjectID_FingerID_Capture.bmp
                    if "_HT" in filename or "_R414" in filename:
                        continue
                    
                    # Step 1: Standardize
                    resized_path = self.standardize_image(file_path, filename)
                    
                    if resized_path:
                        # Step 2: Segment
                        segmented_path = self.segment_image(resized_path, filename)
                        
                        if segmented_path:
                            # Step 3: Enhance
                            # We pass the segmented image path and the original filename
                            enhanced_img = self.enhance_image(segmented_path, filename)
                            
                            if enhanced_img is not None:
                                enhanced_filename = filename.replace(".bmp", "_enhanced.png")
                                enhanced_full_path = os.path.join(self.output_path, enhanced_filename)
                                cv2.imwrite(enhanced_full_path, enhanced_img)
                                print(f"Enhanced: {enhanced_filename}")

                                binary_img = self.binarize_image(enhanced_img)
                                binarized_filename = filename.replace(".bmp", "_binarized.png")
                                binarized_full_path = os.path.join(self.output_path, binarized_filename)                                
                                cv2.imwrite(binarized_full_path, binary_img)
                                print(f"Binarized: {binarized_filename}")

                                # After thinning
                                skeleton_img = self.thin_image(binary_img)
                                skeleton_filename = filename.replace(".bmp", "_skeleton.png")
                                skeleton_full_path = os.path.join(self.output_path, skeleton_filename)                                
                                cv2.imwrite(skeleton_full_path, skeleton_img)
                                print(f"Skeleton: {skeleton_filename}")

                                
                                # Clean skeleton
                                skeleton_img = self.remove_spurs(skeleton_img, min_length=12)

                                # Apply ROI
                                roi = self.get_roi_mask(enhanced_img)
                                skeleton_img = cv2.bitwise_and(skeleton_img, roi)
                                cv2.imwrite(skeleton_full_path, skeleton_img)


                                # --- CORRECTION ---
                                # Feed the ENHANCED GRAYSCALE image to the SDK, not the skeleton.
                                raw_minutiae = self.extract_minutiae(enhanced_full_path, enhanced_filename)

                                # --- POST-PROCESSING ---
                                # Apply your cleaning filters to the SDK's output
                                # Note: You'll need to pass the image shape for border removal
                                h, w = enhanced_img.shape
                                
                                clean_data = self.remove_border_minutiae(raw_minutiae, (h, w))
                                clean_data = self.prune_close_minutiae(clean_data, min_dist=12)
                                clean_data = self.remove_isolated(clean_data)
                                
                                self.all_features[filename] = clean_data
                                print(f"Minutiae extracted: {len(raw_minutiae)} -> {len(clean_data)} (cleaned)")
        
                    count += 1
            print(" -- writing minutiae -- ")
            minutiae_full_path = os.path.join(self.output_path, 'minutiae.json')
            with open(minutiae_full_path, 'w') as f:
                json.dump(self.all_features, f, indent=4)
            print(" -- writing minutiae done!-- ")        
            print(f"\nTotal images processed: {count}")
        except Exception as e:
            print(f"Error [process_dataset] {e}")
        finally:
            del self.engine
    
    def init_sdk(self):
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

    def segment_image(self, image_path, filename):
            # 1. Load the image
            nimage = NImage(image_path)
            
            # 2. Prepare the Finger object
            finger = NFinger()
            finger.image = nimage
            finger.position = NFPosition.nfpUnknown
            
            # 3. Prepare the Subject object
            subject = NSubject()
            subject.fingers.add(finger)

            # 4. Perform the Segmentation
            # This modifies the 'subject' object in place!
            status = self.engine.perform_operation(subject, NBiometricOperations.segment)
            
            # 5. Check Result and Save
            if status == NBiometricStatus.ok:
                # CRITICAL FIX: The segmented result is usually at index 1 (or higher)
                # Index 0 is the original input image.
                if subject.fingers.count > 1:
                    # We grab the last one added, which is typically the segmented result
                    segmented_image = subject.fingers[subject.fingers.count - 1].image
                    
                    new_filename = os.path.splitext(filename)[0] + "_seg.png"
                    output_full_path = os.path.join(self.output_path, new_filename)
                    
                    segmented_image.save_to_file(output_full_path)
                    print(f"Segmented: {new_filename}")
                    return output_full_path
                else:
                    print(f"Warning: Segmentation status OK, but no new segment found for {filename}")
                    return None
            else:
                print(f"Segmentation failed for {filename}: {status}")
                return None


    def enhance_image(self, image_path, filename):
        img = cv2.imread(image_path, 0)
        if img is None:
            return None

        I = img.astype(np.float32) / 255.0

        # Illumination
        sigma = 20
        L = cv2.GaussianBlur(I, (0, 0), sigma)

        epsilon = 1e-6
        R = I / (L + epsilon)

        # Guided filter
        radius = 16
        eps = (0.01)**2
        refined_R = cv2.ximgproc.guidedFilter(I, R, radius, eps)

        refined_R = np.clip(refined_R, 0, 1)
        enhanced = (refined_R * 255).astype(np.uint8)

        # Contrast stretch
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

        # CLAHE (recommended for contactless)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)

        return enhanced

    def binarize_image(self, enhanced_img):
        binary = cv2.adaptiveThreshold(
            enhanced_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=3
        )
        return binary
    
    def thin_image(self, binary_img):
        skeleton = cv2.ximgproc.thinning(
            binary_img,
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
        return skeleton

    def remove_spurs(self, skel, min_length=10):
        """
        Remove short spurs from a skeleton image.
        skel: uint8 skeleton (255 = ridge)
        """
        skel = skel.copy()
        h, w = skel.shape

        def neighbors(y, x):
            return [(y+i, x+j) for i in [-1,0,1] for j in [-1,0,1]
                    if not (i == 0 and j == 0)
                    and 0 <= y+i < h and 0 <= x+j < w]

        changed = True
        while changed:
            changed = False
            endpoints = []

            for y in range(h):
                for x in range(w):
                    if skel[y, x] == 255:
                        n = sum(skel[ny, nx] == 255 for ny, nx in neighbors(y, x))
                        if n == 1:  # endpoint
                            endpoints.append((y, x))

            for y, x in endpoints:
                length = 0
                cy, cx = y, x
                py, px = -1, -1

                while True:
                    nbrs = [(ny, nx) for ny, nx in neighbors(cy, cx)
                            if skel[ny, nx] == 255 and (ny, nx) != (py, px)]
                    if len(nbrs) != 1:
                        break
                    py, px = cy, cx
                    cy, cx = nbrs[0]
                    length += 1
                    if length > min_length:
                        break

                if length <= min_length:
                    skel[y, x] = 0
                    changed = True

        return skel

    def get_roi_mask(self, enhanced_img):
        _, mask = cv2.threshold(
            enhanced_img, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def prune_close_minutiae(self, minutiae, min_dist=12):
        pruned = []
        for m in minutiae:
            if all((m['x']-n['x'])**2 + (m['y']-n['y'])**2 > min_dist**2 for n in pruned):
                pruned.append(m)
        return pruned

    def remove_border_minutiae(self, minutiae, shape, margin=15):
        h, w = shape
        return [
            m for m in minutiae
            if margin < m['x'] < w-margin and margin < m['y'] < h-margin
        ]

    def remove_isolated(self, minutiae, min_neighbors=1, radius=20):
        filtered = []
        for m in minutiae:
            neighbors = sum(
                (m['x']-n['x'])**2 + (m['y']-n['y'])**2 < radius**2
                for n in minutiae if m != n
            )
            if neighbors >= min_neighbors:
                filtered.append(m)
        return filtered


    def extract_minutiae(self, image_path, filename):
        # 1. Load the enhanced image
        nimage = NImage(image_path)

        # --- ROBUST FIX: Force Resolution ---
        # Set both horizontal and vertical resolution explicitly
        nimage.horz_resolution = 500
        nimage.vert_resolution = 500
        nimage.resolution = 500 # Set this too just in case
        
        # DEBUG: Verify the SDK accepted the value
        # This will tell us if the value is sticking or being ignored
        print(f"DEBUG: Resolution for {filename} set to: {nimage.horz_resolution}")
        # ------------------------------------
        
        finger = NFinger()
        finger.image = nimage
        finger.position = NFPosition.nfpUnknown
        
        subject = NSubject()
        subject.fingers.add(finger)

        # 2. Extract Features (Create Template)
        # This operation detects minutiae and generates the template
        status = self.engine.perform_operation(subject, NBiometricOperations.create_template)
        
        if status == NBiometricStatus.ok:
            # The template is stored in the subject
            # Hierarchy: Subject -> Template -> FingerRecords -> Minutiae
            # FIX: Use the property '.template' instead of the method '.get_template()'
            template = subject.template
            
            if template:
                # Based on finger_get_minutia.py, the path to minutiae is nested:
                # Template -> Fingers (Collection) -> Records (NFRecord) -> Minutiae
                # Note: The SDK wrapper structure can be tricky, so we use a try-block or inspect it.
                
                # Let's try the structure from your tutorial file:
                # minutia = finger_templates[0].fingers.records[0].minutiae
                
                try:
                    # Access the first finger record in the template
                    minutiae_list = template.fingers.records[0].minutiae
                except AttributeError:
                    # Fallback: sometimes it is accessed differently depending on the wrapper version
                    print(f"DEBUG: Template structure: {dir(template)}")
                    # Try direct access if the above fails (sometimes template.fingers[0].minutiae)
                    minutiae_list = template.fingers[0].minutiae
                
                # Prepare data for Graph Construction (Step 5)
                extracted_data = []
                
                # Prepare visualization image
                # We need to convert NImage to a numpy format OpenCV can handle
                # (The tutorial uses a helper, but we can reload via cv2 for simplicity)
                vis_image = cv2.imread(image_path)
                
                # Conversion factor: SDK byte (0-255) -> Degrees -> Radians
                # 360 / 255 ≈ 1.411764706
                byte_to_rad = (360.0 / 255.0) * (np.pi / 180.0)

                print(f"Minutiae found for {filename}: {len(minutiae_list)}")
                
                for m in minutiae_list:
                    # Geometric attributes
                    x = m.x
                    y = m.y
                    angle_rad = m.angle * byte_to_rad
                    
                    # Type: 0=Unknown, 1=Ending, 2=Bifurcation (SDK dependent)
                    m_type = m.type 
                    
                    # Store for the Graph
                    extracted_data.append({
                        'x': x,
                        'y': y,
                        'angle': angle_rad,
                        'type': m_type
                    })

                    # --- Visualization (Optional but Recommended) ---
                    # Draw location
                    cv2.circle(vis_image, (x, y), 3, (0, 0, 255), -1)
                    
                    # Draw direction
                    line_len = 15
                    x2 = int(x + line_len * np.cos(angle_rad))
                    y2 = int(y + line_len * np.sin(angle_rad))
                    cv2.line(vis_image, (x, y), (x2, y2), (255, 0, 0), 1)
                
                # Save Visualization
                vis_filename = filename.replace("_enhanced.png", "_minutiae.png")
                cv2.imwrite(os.path.join(self.output_path, vis_filename), vis_image)
                
                return extracted_data
            else:
                print(f"No template created for {filename}")
                return []
        else:
            print(f"Extraction failed for {filename}: {status}")
            return []
    
    
