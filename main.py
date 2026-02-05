import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time

# Import your modules
from src.preprocessor import FingerprintPreprocessor
from src.loose_ga import LooseGeneticAlgorithm
import src.matcher_utils as utils

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data", "DS1_sample")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_images")

def extract_features_for_dataset(preprocessor, processed_dir):
    """
    1. Runs preprocessing (Segmentation/Enhancement).
    2. Computes and CACHES: Minutiae, Ridge Counts, Descriptors.
    Returns a dictionary: {filename: {minutiae, rc_matrix, descriptors}}
    """
    print("--- Step 1: Preprocessing & Feature Extraction ---")
    
    # 1. Run standard preprocessing (saves images to disk)
    # This calls your existing method which handles Sizing/Segmentation/Enhancement
    preprocessor.process_dataset()
    
    gallery_data = {}
    
    # 2. Iterate processed files to build the in-memory gallery
    # We look for the enhanced images or skeletons
    # Assuming file naming: "filename_enhanced.png"
    enhanced_files = glob.glob(os.path.join(processed_dir, "*_enhanced.png"))
    
    print(f"Extracting Graph Features for {len(enhanced_files)} images...")
    
    for filepath in enhanced_files:
        filename = os.path.basename(filepath)
        original_name = filename.replace("_enhanced.png", ".bmp") # Adjust extension if needed
        
        # Load necessary images
        enhanced_img = cv2.imread(filepath, 0)
        
        # Load skeleton (created by preprocessor)
        skeleton_path = filepath.replace("_enhanced.png", "_skeleton.png")
        skeleton = cv2.imread(skeleton_path, 0)
        
        if enhanced_img is None or skeleton is None:
            print(f"Skipping {filename}: Missing enhanced or skeleton image.")
            continue

        # A. Orientation Field (Dense)
        orientation_map = utils.compute_orientation_field(enhanced_img)
        
        # B. Extract Minutiae 
        # (We can use the preprocessor's method, or load from the JSON if already saved)
        # Here we re-run extraction to ensure we have the objects in memory
        minutiae_data = preprocessor.extract_minutiae(skeleton, orientation_map)
        
        # Convert dictionaries to tuples (x, y, theta) for the matcher
        minutiae_list = []
        for m in minutiae_data:
            # Check if your extract_minutiae returns dicts or objects
            # Based on your previous code, it returned dicts {'x':, 'y':, 'angle':}
            minutiae_list.append((m['x'], m['y'], m['angle']))
            
        if len(minutiae_list) < 5: # Skip poor quality
            continue

        # C. Descriptors (Concentric Circles)
        descriptors = []
        for m in minutiae_list:
            desc = utils.get_local_descriptor(m, orientation_map)
            descriptors.append(desc)
            
        # D. Ridge Count Matrix
        rc_matrix = utils.precompute_ridge_counts(skeleton, minutiae_list)
        
        # Store in Gallery
        gallery_data[original_name] = {
            "minutiae": minutiae_list,
            "descriptors": descriptors,
            "rc_matrix": rc_matrix,
            "path": filepath
        }
        
    print(f"Feature Extraction Complete. {len(gallery_data)} templates ready.")
    return gallery_data

def run_matching_experiment(gallery_data):
    """
    Runs All-vs-All matching on the gallery_data.
    """
    print("\n--- Step 2: Running Graph Matching Experiment (Loose GA) ---")
    
    filenames = list(gallery_data.keys())
    N = len(filenames)
    
    genuine_scores = []
    imposter_scores = []
    
    start_time = time.time()
    
    # Loop pairs
    for i in range(N):
        for j in range(i + 1, N):
            name_a = filenames[i]
            name_b = filenames[j]
            
            data_a = gallery_data[name_a]
            data_b = gallery_data[name_b]
            
            # 1. Build Compatibility Matrix
            W, assignments = utils.compute_compatibility_matrix(
                data_a['minutiae'], data_a['descriptors'], data_a['rc_matrix'],
                data_b['minutiae'], data_b['descriptors'], data_b['rc_matrix']
            )
            
            # 2. Run Loose GA
            ga = LooseGeneticAlgorithm(
                W, assignments,
                len(data_a['minutiae']), len(data_b['minutiae']),
                pop_size=100, max_generations=30 # Tuned for speed/accuracy trade-off
            )
            best_p, _ = ga.run()
            
            # 3. Refinement (Expansion)
            final_pairs = utils.expand_minutia_pairs(
                data_a['minutiae'], data_b['minutiae'],
                data_a['rc_matrix'], data_b['rc_matrix'],
                best_p
            )
            
            # 4. Final Score
            score = utils.calculate_comparison_score(
                final_pairs, W, assignments,
                len(data_a['minutiae']), len(data_b['minutiae'])
            )
            
            # 5. Label
            label = utils.get_match_label(name_a, name_b)
            if label == "Genuine":
                genuine_scores.append(score)
            else:
                imposter_scores.append(score)
                
            if (len(genuine_scores) + len(imposter_scores)) % 10 == 0:
                print(f"Processed {len(genuine_scores) + len(imposter_scores)} pairs...", end='\r')

    print(f"\nMatching finished in {time.time() - start_time:.2f}s")
    return genuine_scores, imposter_scores

def main():
    # 1. Initialize
    preprocessor = FingerprintPreprocessor(DATASET_DIR, PROCESSED_DIR)
    
    # 2. Extract Features (Batch)
    gallery_data = extract_features_for_dataset(preprocessor, PROCESSED_DIR)
    
    # 3. Run Matching
    gen_scores, imp_scores = run_matching_experiment(gallery_data)
    
    print(f"\nGenuine Scores: {len(gen_scores)} | Avg: {np.mean(gen_scores) if gen_scores else 0:.4f}")
    print(f"Imposter Scores: {len(imp_scores)} | Avg: {np.mean(imp_scores) if imp_scores else 0:.4f}")
    
    # 4. Evaluation
    eer, threshold = utils.find_eer(gen_scores, imp_scores)
    print("-" * 30)
    print(f"FINAL EER: {eer:.2f}% @ Threshold: {threshold:.4f}")
    print("-" * 30)
    
    # 5. Plot ROC
    far, tar = utils.get_roc_data(gen_scores, imp_scores)
    plt.figure(figsize=(6, 6))
    plt.plot(far, tar, label=f"ROC (EER={eer:.2f}%)", color='blue')
    plt.plot([0, 100], [0, 100], linestyle='--', color='gray')
    plt.xlabel("FAR (%)")
    plt.ylabel("TAR (%)")
    plt.title("ROC Curve - Global Minutia Topology")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(PROCESSED_DIR, "roc_curve_final.png")
    plt.savefig(save_path)
    print(f"ROC Curve saved to {save_path}")

if __name__ == "__main__":
    main()