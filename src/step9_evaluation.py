import os
import argparse
import glob
import itertools
import random
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging

# Set up logging to both shell and file
def setup_logging(log_file="evaluation.log"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_processed_filename(filename):
    try:
        parts = filename.split('_')
        return f"{parts[0]}_{parts[1]}" # Subject_Finger
    except Exception:
        return None

def run_matching_pipeline(t_json, t_img, q_json, q_img, temp_dir="temp_match"):
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    
    matrix_path = os.path.join(temp_dir, "W.npy")
    lga_path = os.path.join(temp_dir, "lga.npy")
    result_path = os.path.join(temp_dir, "result.json")
    
    try:
        with open(t_json) as f: Nt = len(json.load(f))
        with open(q_json) as f: Nq = len(json.load(f))
        if Nt < 5 or Nq < 5: return 0.0

        subprocess.check_output([
            "python3", "src/step5_build_matrix.py",
            "--template_json", t_json, "--template_img", t_img,
            "--query_json", q_json, "--query_img", q_img,
            "--output", os.path.splitext(matrix_path)[0]
        ], stderr=subprocess.DEVNULL)

        subprocess.check_output([
            "python3", "src/step6_lga.py",
            "--matrix_path", matrix_path, "--Nt", str(Nt), "--Nq", str(Nq),
            "--output", os.path.splitext(lga_path)[0]
        ], stderr=subprocess.DEVNULL)
        
        subprocess.check_output([
            "python3", "src/step7_refinement.py",
            "--lga_solution", lga_path, "--matrix_path", matrix_path,
            "--template_json", t_json, "--template_img", t_img,
            "--query_json", q_json, "--query_img", q_img,
            "--output", result_path
        ], stderr=subprocess.DEVNULL)
        
        with open(result_path) as f:
            return float(json.load(f)['score'])
            
    except Exception as e:
        logging.error(f"Pipeline failure for {os.path.basename(t_json)} vs {os.path.basename(q_json)}: {e}")
        return 0.0

def main(args):
    setup_logging(args.log_file)
    processed_dir = args.processed_dir
    logging.info(f"Loading data from {processed_dir}...")
    
    json_files = sorted(glob.glob(os.path.join(processed_dir, "*_minutiae.json")))
    if not json_files:
        logging.error("No processed data found!")
        sys.exit(1)

    samples = []
    for jf in json_files:
        base = jf.replace("_minutiae.json", "")
        img = base + "_enhanced.jpg"
        if os.path.exists(img):
            fid = parse_processed_filename(os.path.basename(jf))
            if fid: samples.append({"id": fid, "json": jf, "img": img})
    
    logging.info(f"Loaded {len(samples)} samples from {len(set(s['id'] for s in samples))} unique fingers.")

    groups = {}
    for s in samples:
        groups.setdefault(s['id'], []).append(s)
    unique_ids = list(groups.keys())

    # --- Generate Pairs with Limits ---
    genuine_pairs = []
    impostor_pairs = []

    # 1. Genuine Pairs (Same Finger)
    all_possible_genuines = []
    for fid in unique_ids:
        items = groups[fid]
        if len(items) > 1:
            pairs = list(itertools.combinations(items, 2))
            all_possible_genuines.extend(pairs)
    
    # Apply limit to genuine pairs
    if len(all_possible_genuines) > args.max_genuines:
        logging.info(f"Downsampling genuine pairs from {len(all_possible_genuines)} to {args.max_genuines}")
        genuine_pairs = random.sample(all_possible_genuines, args.max_genuines)
    else:
        genuine_pairs = all_possible_genuines

    # 2. Impostor Pairs (Different Fingers)
    while len(impostor_pairs) < args.max_impostors:
        id1, id2 = random.sample(unique_ids, 2)
        s1 = random.choice(groups[id1])
        s2 = random.choice(groups[id2])
        # Avoid duplicates (simple check)
        if (s1, s2) not in impostor_pairs and (s2, s1) not in impostor_pairs:
            impostor_pairs.append((s1, s2))

    logging.info(f"Final Evaluation Set: {len(genuine_pairs)} Genuine, {len(impostor_pairs)} Impostor.")
    logging.info("Running matching pipeline...")

    # --- Run Matching ---
    gen_scores = []
    for i, (p1, p2) in enumerate(genuine_pairs):
        s = run_matching_pipeline(p1['json'], p1['img'], p2['json'], p2['img'])
        gen_scores.append(s)
        if i % 5 == 0: 
            logging.info(f"Genuine Progress: {i}/{len(genuine_pairs)} (Last Score: {s:.4f})")
    
    imp_scores = []
    for i, (p1, p2) in enumerate(impostor_pairs):
        s = run_matching_pipeline(p1['json'], p1['img'], p2['json'], p2['img'])
        imp_scores.append(s)
        if i % 5 == 0: 
            logging.info(f"Impostor Progress: {i}/{len(impostor_pairs)} (Last Score: {s:.4f})")

    # --- Calculate Metrics (FAR, FRR, TPR) ---
    thresholds = np.linspace(0, 1.0, 200)
    far_list = []
    frr_list = []
    tpr_list = [] # True Positive Rate
    
    min_dist = float('inf')
    eer = 0.0
    eer_thresh = 0.0
    
    for t in thresholds:
        # FAR: Fraction of impostors accepted (score >= threshold)
        fa = sum(1 for s in imp_scores if s >= t) / len(imp_scores) if imp_scores else 0
        
        # FRR: Fraction of genuines rejected (score < threshold)
        fr = sum(1 for s in gen_scores if s < t) / len(genuine_pairs) if genuine_pairs else 0
        
        # TPR: Fraction of genuines accepted (score >= threshold)
        tpr = 1.0 - fr
        
        far_list.append(fa)
        frr_list.append(fr)
        tpr_list.append(tpr)
        
        # EER Calculation (where FAR approx FRR)
        if abs(fa - fr) < min_dist:
            min_dist = abs(fa - fr)
            eer = (fa + fr) / 2
            eer_thresh = t

    logging.info(f"EER: {eer:.4f} at Threshold: {eer_thresh:.4f}")

    # --- Plot ROC (TPR vs FAR) ---
    plt.figure()
    plt.plot(far_list, tpr_list, label=f'ROC Curve (EER={eer:.2f})', color='blue')
    
    # Plot diagonal (Random Guess line)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(args.plot_output)
    logging.info(f"ROC Curve saved to {args.plot_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 5: Experimental Evaluation")
    parser.add_argument('--processed_dir', required=True, help="Directory containing processed .json and .jpg files")
    parser.add_argument('--max_genuines', type=int, default=100, help="Maximum number of genuine pairs to test")
    parser.add_argument('--max_impostors', type=int, default=100, help="Maximum number of impostor pairs to test")
    parser.add_argument('--plot_output', default='roc_curve.png', help="Filename for the output ROC plot")
    parser.add_argument('--log_file', default='evaluation.log', help="Filename for the log output")
    args = parser.parse_args()
    main(args)