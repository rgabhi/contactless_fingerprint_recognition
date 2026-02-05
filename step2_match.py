import os
import glob
import numpy as np
import time
import multiprocessing
from functools import partial

from src.loose_ga import LooseGeneticAlgorithm
import src.matcher_utils as utils

# --- config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def load_gallery():
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
    data_a, data_b = pair_data
    
    # 1. build  compatibility mat
    W, assignments = utils.compute_compatibility_matrix(
        data_a['minutiae'], data_a['descriptors'], data_a['rc_matrix'],
        data_b['minutiae'], data_b['descriptors'], data_b['rc_matrix']
    )
    
    if len(assignments) == 0:
        return (0.0, utils.get_match_label(data_a['filename'], data_b['filename']))

    # 2. run lga
    ga = LooseGeneticAlgorithm(
        W, assignments,
        len(data_a['minutiae']), len(data_b['minutiae']),
        pop_size=100
    )
    best_p, _ = ga.run(max_generations=50)
    
    final_pairs = utils.expand_minutia_pairs(
        data_a['minutiae'], data_b['minutiae'],
        data_a['rc_matrix'], data_b['rc_matrix'],
        best_p
    )
    
    # final Score
    score = utils.calculate_comparison_score(
        final_pairs, W, assignments,
        len(data_a['minutiae']), len(data_b['minutiae'])
    )
    
    label = utils.get_match_label(data_a['filename'], data_b['filename'])
    return (score, label)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    gallery = load_gallery()
    N = len(gallery)
    if N < 2:
        print("Not enough templates to match.")
        return

    print(f"Starting matching for {N} templates ({N*(N-1)//2} pairs)...")
    
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((gallery[i], gallery[j]))
    
    start_time = time.time()
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"Using {num_cores} cores.")
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(match_pair, pairs, chunksize=10)
        
    print(f"Matching finished in {time.time() - start_time:.2f}s")
    
    gen_scores = [r[0] for r in results if r[1] == 'Genuine']
    imp_scores = [r[0] for r in results if r[1] == 'Imposter']
    
    print(f"Genuine Scores: {len(gen_scores)} | Avg: {np.mean(gen_scores) if gen_scores else 0:.4f}")
    print(f"Imposter Scores: {len(imp_scores)} | Avg: {np.mean(imp_scores) if imp_scores else 0:.4f}")
    
    if len(gen_scores) > 0 and len(imp_scores) > 0:
        eer, threshold = utils.find_eer(gen_scores, imp_scores)
        print(f"FINAL EER: {eer:.2f}% @ Threshold: {threshold:.4f}")
        
        far, tar = utils.get_roc_data(gen_scores, imp_scores)
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(far, tar, label=f"ROC (EER={eer:.2f}%)")
        plt.plot([0, 100], [0, 100], 'k--')
        plt.xlabel("FAR (%)")
        plt.ylabel("TAR (%)")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, "roc_curve_final.png"))
        np.savez(os.path.join(RESULTS_DIR, "scores.npz"), gen=gen_scores, imp=imp_scores)
    else:
        print("Insufficient data to calculate EER.")

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()