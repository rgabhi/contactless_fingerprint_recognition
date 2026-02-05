import os
import glob
import numpy as np
import time
import multiprocessing
from functools import partial

# Ensure these imports point to your actual file structure
from src.loose_ga import LooseGeneticAlgorithm
import src.matcher_utils as utils

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def load_gallery():
    """Loads all .npz feature files into memory."""
    files = glob.glob(os.path.join(FEATURES_DIR, "*.npz"))
    gallery = []
    print(f"Loading {len(files)} templates...")
    for f in files:
        data = np.load(f)
        gallery.append({
            'minutiae': data['minutiae'],
            'descriptors': data['descriptors'],
            'rc_matrix': data['rc_matrix'],
            'filename': str(data['original_filename'])
        })
    return gallery

def match_pair(pair_data):
    """
    Worker function for a single pair comparison.
    """
    data_a, data_b = pair_data
    
    # 1. Build Compatibility Matrix (Optimized with Pruning)
    # Threshold=0.3 means descriptors must be somewhat similar to be considered
    W, assignments = utils.compute_compatibility_matrix(
        data_a['minutiae'], data_a['descriptors'], data_a['rc_matrix'],
        data_b['minutiae'], data_b['descriptors'], data_b['rc_matrix'],
        similarity_threshold=0.3 
    )
    
    # If no compatible minutiae found
    if len(assignments) == 0:
        return (0.0, utils.get_match_label(data_a['filename'], data_b['filename']))

    # 2. Run Loose GA
    # FIX: max_generations removed from __init__
    ga = LooseGeneticAlgorithm(
        W, assignments,
        len(data_a['minutiae']), len(data_b['minutiae']),
        pop_size=50
    )
    
    # FIX: max_generations added here
    best_p, _ = ga.run(max_generations=20)
    
    # 3. Refinement
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
    
    label = utils.get_match_label(data_a['filename'], data_b['filename'])
    return (score, label)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load Data
    gallery = load_gallery()
    N = len(gallery)
    if N < 2:
        print("Not enough templates to match.")
        return

    print(f"Starting matching for {N} templates ({N*(N-1)//2} pairs)...")
    
    # 2. Generate Pairs
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((gallery[i], gallery[j]))
    
    # 3. Run Matching (Parallel)
    start_time = time.time()
    
    # Use fewer cores to be safe, or multiprocessing.cpu_count() - 2
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"Using {num_cores} cores.")
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(match_pair, pairs, chunksize=10)
        
    print(f"Matching finished in {time.time() - start_time:.2f}s")
    
    # 4. Process Results
    gen_scores = [r[0] for r in results if r[1] == 'Genuine']
    imp_scores = [r[0] for r in results if r[1] == 'Imposter']
    
    print(f"Genuine Scores: {len(gen_scores)} | Avg: {np.mean(gen_scores) if gen_scores else 0:.4f}")
    print(f"Imposter Scores: {len(imp_scores)} | Avg: {np.mean(imp_scores) if imp_scores else 0:.4f}")
    
    # 5. Evaluation & Plotting
    if len(gen_scores) > 0 and len(imp_scores) > 0:
        eer, threshold = utils.find_eer(gen_scores, imp_scores)
        print(f"FINAL EER: {eer*100:.2f}% @ Threshold: {threshold:.4f}")
        
        far, tar = utils.get_roc_data(gen_scores, imp_scores)
        
        # Save Results
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(far, tar, label=f"ROC (EER={eer*100:.2f}%)")
        plt.plot([0, 100], [0, 100], 'k--')
        plt.xlabel("FAR (%)")
        plt.ylabel("TAR (%)")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, "roc_curve_final.png"))
        
        # Save raw scores for later analysis
        np.savez(os.path.join(RESULTS_DIR, "scores.npz"), gen=gen_scores, imp=imp_scores)
    else:
        print("Insufficient data to calculate EER/ROC (Need both Genuine and Imposter matches).")

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()