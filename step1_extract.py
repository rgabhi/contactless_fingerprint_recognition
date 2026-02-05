import os
import cv2
import numpy as np
import glob
from src.preprocessor import FingerprintPreprocessor
import src.matcher_utils as utils
import json

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data", "DS1_sample")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_images")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")

def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    
    # 1. Initialize Preprocessor
    preprocessor = FingerprintPreprocessor(DATASET_DIR, PROCESSED_DIR)

    try:
        # Iterate through all subdirectories in the dataset path
        # We assume folders are named '1', '2', etc.
        subject_folders = [f for f in os.listdir(preprocessor.dataset_path) if os.path.isdir(os.path.join(preprocessor.dataset_path, f))]

        count = 0
        for subject in subject_folders:
            subject_path = os.path.join(preprocessor.dataset_path, subject)
            
            # Find all BMP files
            bmp_files = glob.glob(os.path.join(subject_path, "*.bmp"))
            print(f"Found {len(bmp_files)} total files in dataset.")
            
            for file_path in bmp_files:
                filename = os.path.basename(file_path)
                
                # Filter out the pre-processed files (HT, R414)
                # We only want the pattern SIRE-SubjectID_FingerID_Capture.bmp
                if "_HT" in filename or "_R414" in filename:
                    continue
                
                # Optional: Skip standard pre-processed files if they exist in the source folder
                if "_enhanced" in filename or "_skeleton" in filename or "_seg" in filename:
                    continue
                # ----------------------------

                # Skip if feature file already exists (Resume capability)
                feature_save_path = os.path.join(FEATURES_DIR, filename.replace(".bmp", ".npz"))
                if os.path.exists(feature_save_path):
                    # print(f"Skipping {filename} (already processed)")
                    continue
                
                # Step 1: Standardize
                resized_path = preprocessor.standardize_image(file_path, filename)
                
                if resized_path:
                    # Step 2: Segment
                    segmented_path = preprocessor.segment_image(resized_path, filename)
                    
                    if segmented_path:
                        # Step 3: Enhance
                        # We pass the segmented image path and the original filename
                        enhanced_img = preprocessor.enhance_image(segmented_path, filename)
                        
                        if enhanced_img is not None:
                            enhanced_filename = filename.replace(".bmp", "_enhanced.png")
                            enhanced_full_path = os.path.join(preprocessor.output_path, enhanced_filename)
                            cv2.imwrite(enhanced_full_path, enhanced_img)
                            print(f"Enhanced: {enhanced_filename}")

                            binary_img = preprocessor.binarize_image(enhanced_img)
                            binarized_filename = filename.replace(".bmp", "_binarized.png")
                            binarized_full_path = os.path.join(preprocessor.output_path, binarized_filename)                                
                            # cv2.imwrite(binarized_full_path, binary_img)
                            print(f"Binarized: {binarized_filename}")

                            # After thinning
                            skeleton_img = preprocessor.thin_image(binary_img)
                            skeleton_filename = filename.replace(".bmp", "_skeleton.png")
                            skeleton_full_path = os.path.join(preprocessor.output_path, skeleton_filename)                                
                          
                            # Clean skeleton
                            skeleton_img = preprocessor.remove_spurs(skeleton_img, min_length=12)

                            # Apply ROI
                            roi = preprocessor.get_roi_mask(enhanced_img)
                            skeleton_img = cv2.bitwise_and(skeleton_img, roi)
                            cv2.imwrite(skeleton_full_path, skeleton_img)
                            print(f"Skeleton: {skeleton_filename}")


                            # --- CORRECTION ---
                            # Feed the ENHANCED GRAYSCALE image to the SDK, not the skeleton.
                            raw_minutiae = preprocessor.extract_minutiae(enhanced_full_path, enhanced_filename)

                            # --- POST-PROCESSING ---
                            # Apply your cleaning filters to the SDK's output
                            # Note: You'll need to pass the image shape for border removal
                            h, w = enhanced_img.shape
                            
                            clean_data = preprocessor.remove_border_minutiae(raw_minutiae, (h, w))
                            clean_data = preprocessor.prune_close_minutiae(clean_data, min_dist=12)
                            clean_data = preprocessor.remove_isolated(clean_data)
                            
                            preprocessor.all_features[filename] = clean_data
                            print(f"Minutiae extracted: {len(raw_minutiae)} -> {len(clean_data)} (cleaned)")


                             # Compute Orientation Field
                            orientation_map = utils.compute_orientation_field(enhanced_img)
                            
                            # Convert dicts to list of tuples for utils -> (x, y, angle)
                            minutiae_list = [(m['x'], m['y'], m['angle']) for m in clean_data]
                            
                            # Descriptors
                            descriptors = []
                            for m in minutiae_list:
                                desc = utils.get_local_descriptor(m, orientation_map)
                                descriptors.append(desc)
                            descriptors = np.array(descriptors) 

                            
                            # Ridge Count Matrix
                            rc_matrix = utils.precompute_ridge_counts(skeleton_img, minutiae_list)

                            # 7. Save Features
                            np.savez_compressed(
                                feature_save_path,
                                minutiae=np.array(minutiae_list), 
                                descriptors=descriptors,          
                                rc_matrix=rc_matrix,              
                                original_filename=filename
                            )
                            print(f"  -> Features saved to {os.path.basename(feature_save_path)}")
        
                    count += 1
            print(" -- writing minutiae -- ")
            minutiae_full_path = os.path.join(preprocessor.output_path, 'minutiae.json')
            with open(minutiae_full_path, 'w') as f:
                json.dump(preprocessor.all_features, f, indent=4)
            print(" -- writing minutiae done!-- ")        
            print(f"\nTotal images processed: {count}")
    except Exception as e:
        print(f"Error [process_dataset] {e}")
    finally:
        del preprocessor.engine

if __name__ == "__main__":
    main()