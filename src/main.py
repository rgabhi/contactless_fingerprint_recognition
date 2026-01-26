import cv2
import numpy as np
import os
import argparse

# Import your modules
from segmentation import segment_fingerprint
from enhancement import enhance_fingerprint
from matcher import (compute_orientation_map, compute_ridge_count_matrix, 
                     compute_diagonal_similarity, compute_global_consistency_matrix,
                     calculate_final_score)
from lga import LooseGA
from refinement import refinement_step
from utils import Minutia

# --- SDK HELPER FUNCTION ---
# You need a way to call your extraction logic for a SINGLE file
from nsdk.media import NImage
from nsdk.biometrics import FingerEngine, NFinger, NSubject, NBiometricOperations, NBiometricStatus
from pynsdk.licensing import NLicense, NLicenseManager

def process_and_extract(image_path, engine):
    """
    1. Loads Image
    2. Segments & Enhances (Your Custom Code)
    3. Saves Temp
    4. Calls SDK for Minutiae & Binarization
    """
    # A. Pre-processing (Your Implementation)
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None: raise ValueError("Image not found")
    
    # Resize to standard (approx 500 DPI if needed, or skip if dataset is already 500)
    # raw_img = cv2.resize(raw_img, (500, 500)) 

    # Step 2: Segmentation
    seg_img, mask = segment_fingerprint(raw_img, use_fallback=True)
    
    # Step 3: Enhancement
    enh_img = enhance_fingerprint(seg_img)
    
    # Save enhanced image temporarily so SDK can load it
    temp_path = image_path.replace(".bmp", "_processed_temp.bmp")
    cv2.imwrite(temp_path, enh_img)
    
    # B. SDK Extraction
    # Create NImage from the enhanced file
    nimage = NImage(temp_path)
    nimage.horz_resolution = 500
    nimage.vert_resolution = 500
    
    subject = NSubject()
    finger = NFinger()
    finger.image = nimage
    subject.fingers.add(finger)
    
    # We need both Template (Minutiae) and Binarized Image
    engine.fingers_return_binarized_image = True
    status = engine.perform_operation(subject, NBiometricOperations.create_template)
    
    if status != NBiometricStatus.ok:
        raise Exception(f"SDK Extraction failed: {status}")
        
    # Extract Minutiae List
    minutiae_list = []
    sdk_minutiae = subject.fingers[0].objects[0].minutiae
    for m in sdk_minutiae:
        # Convert SDK object to our simple Python object
        # Note: Check if SDK returns angle in radians or degrees. 
        # Usually SDKs are varying. If degrees, convert to radians.
        # Assuming SDK is radians or we convert:
        minutiae_list.append(Minutia(m.x, m.y, m.angle))
        
    # Extract Binarized Image (for Ridge Count)
    bin_img_obj = subject.fingers[0].binarized_image
    # Save/Load binarized image to numpy
    bin_path = image_path.replace(".bmp", "_bin_temp.bmp")
    bin_img_obj.save_to_file(bin_path)
    bin_numpy = cv2.imread(bin_path, cv2.IMREAD_GRAYSCALE)
    
    # Cleanup Temp Files
    # os.remove(temp_path) 
    # os.remove(bin_path)
    
    return minutiae_list, bin_numpy, enh_img

# --- MAIN PIPELINE ---
def match_fingerprints(path_T, path_Q):
    # 0. Setup SDK License (Do this once globally)
    NLicenseManager.set_trial_mode(True)
    if not NLicense.obtain("/local", 5000, "FingerExtractor"):
        print("License failed")
        return 0
    engine = NBiometricEngine()

    print(f"--- Processing Template: {path_T} ---")
    minutiae_T, bin_T, enh_T = process_and_extract(path_T, engine)
    
    print(f"--- Processing Query: {path_Q} ---")
    minutiae_Q, bin_Q, enh_Q = process_and_extract(path_Q, engine)
    
    Nt = len(minutiae_T)
    Nq = len(minutiae_Q)
    print(f"Minutiae Count -> Template: {Nt}, Query: {Nq}")
    
    # 1. Orientation Maps (Needed for Descriptors)
    print("Computing Orientation Maps...")
    orient_T = compute_orientation_map(enh_T)
    orient_Q = compute_orientation_map(enh_Q)
    
    # 2. Ridge Count Matrices
    print("Computing Ridge Count Matrices...")
    rc_T = compute_ridge_count_matrix(minutiae_T, bin_T)
    rc_Q = compute_ridge_count_matrix(minutiae_Q, bin_Q)
    
    # 3. Diagonal Similarity (Minutia-Wise)
    print("Computing Similarity Matrix...")
    diag_sim = compute_diagonal_similarity(minutiae_T, minutiae_Q, orient_T, orient_Q)
    
    # 4. Global Compatibility Matrix (W)
    W = compute_global_consistency_matrix(minutiae_T, minutiae_Q, rc_T, rc_Q, diag_sim)
    
    # 5. Optimization (LGA)
    print("Running Loose Genetic Algorithm...")
    lga = LooseGA(W, Nt, Nq)
    
    # Get greedy seed from diag_sim
    greedy_seed = np.zeros(Nt, dtype=int)
    for i in range(Nt):
        best_match = np.argmax(diag_sim[i])
        if diag_sim[i][best_match] > 0: # Threshold check optional
            greedy_seed[i] = best_match + 1 # 1-based index
            
    best_solution = lga.run(greedy_seed)
    
    # 6. Refinement
    print("Running Refinement Step...")
    refined_T, refined_Q = refinement_step(best_solution, minutiae_T, minutiae_Q, rc_T, rc_Q)
    
    # 7. Final Score
    # Recalculate energy of the FINAL refined set for the score formula
    # Construct a sparse vector x_final for the energy calculation
    x_final = np.zeros(Nt * Nq)
    for k in range(len(refined_T)):
        i = refined_T[k]
        j = refined_Q[k]
        x_final[i * Nq + j] = 1
    
    energy_final = x_final.T @ W @ x_final
    
    score = calculate_final_score(refined_T, refined_Q, energy_final, Nt, Nq)
    print(f"Match Found: {len(refined_T)} pairs")
    print(f"FINAL SCORE: {score:.4f}")
    
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("template", help="Path to template fingerprint image")
    parser.add_argument("query", help="Path to query fingerprint image")
    args = parser.parse_args()
    
    match_fingerprints(args.template, args.query)