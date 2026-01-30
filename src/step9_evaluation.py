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

def setup_logging(log_file="evaluation.log"):
    # Reset handlers to avoid duplicate logs if run multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'), # 'w' overwrites log each run
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_processed_filename(filename):
    try:
        parts = filename.split('_')
        # Returns SubjectID_FingerID (e.g., "101_1")
        return f"{parts[0]}_{parts[1]}"
    except Exception:
        return None

def save_metrics_table(output_path, eer, threshold, gen_count, imp_count):
    """Saves the performance metrics to a text file as a table."""
    try:
        with open(output_path, 'w') as f:
            f.write("==================================================\n")
            f.write("           PERFORMANCE METRICS REPORT             \n")
            f.write("==================================================\n")
            f.write(f"{'Metric':<25} | {'Value':<20}\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Equal Error Rate (EER)':<25} | {eer:.4f}\n")
            f.write(f"{'Operating Threshold':<25} | {threshold:.4f}\n")
            f.write(f"{'Genuine Pairs Tested':<25} | {gen_count}\n")
            f.write(f"{'Impostor Pairs Tested':<25} | {imp_count}\n")
            f.write("==================================================\n")
        logging.info(f"Performance metrics saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save metrics table: {e}")

def run_matching_pipeline(t_json, t_img, q_json, q_img, temp_dir="temp_match"):
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    
    matrix_path = os.path.join(temp_dir, "W.npy")
    lga_path = os.path.join(temp_dir, "lga.npy")
    result_path = os.path.join(temp_dir, "result.json")
    
    # Clean previous run artifacts
    if os.path.exists(result_path): os.remove(result_path)

    try:
        with open(t_json) as f: Nt = len(json.load(f))
        with open(q_json) as f: Nq = len(json.load(f))
        
        # Skip if too few minutiae (prevent matrix errors)
        if Nt < 5 or Nq < 5: return 0.0

        # Step 5: Build Matrix
        cmd_step5 = [
            "python3", "src/step5_build_matrix.py",
            "--template_json", t_json, "--template_img", t_img,
            "--query_json", q_json, "--query_img", q_img,
            "--output", os.path.splitext(matrix_path)[0]
        ]
        res5 = subprocess.run(cmd_step5, capture_output=True, text=True)
        if res5.returncode != 0:
            logging.error(f"Step 5 Error ({os.path.basename(t_json)}): {res5.stderr.strip()}")
            return 0.0

        # Step 6: LGA
        cmd_step6 = [
            "python3", "src/step6_lga.py",
            "--matrix_path", matrix_path, "--Nt", str(Nt), "--Nq", str(Nq),
            "--output", os.path.splitext(lga_path)[0]
        ]
        res6 = subprocess.run(cmd_step6, capture_output=True, text=True)
        if res6.returncode != 0:
            logging.error(f"Step 6 Error: {res6.stderr.strip()}")
            return 0.0
        
        # Step 7: Refinement
        cmd_step7 = [
            "python3", "src/step7_refinement.py",
            "--lga_solution", lga_path, "--matrix_path", matrix_path,
            "--template_json", t_json, "--template_img", t_img,
            "--query_json", q_json, "--query_img", q_img,
            "--output", result_path
        ]
        res7 = subprocess.run(cmd_step7, capture_output=True, text=True)
        if res7.returncode != 0:
            logging.error(f"Step 7 Error: {res7.stderr.strip()}")
            return 0.0
        
        if os.path.exists(result_path):
            with open(result_path) as f:
                return float(json.load(f)['score'])
        return 0.0
            
    except Exception as e:
        logging.error(f"Pipeline Exception: {e}")
        return 0.0

def main(args):
    setup_logging(args.log_file)
    processed_dir = args.processed_dir
    logging.info(f"Loading data from {processed_dir}...")
    
    # Load JSON files
    json_files = sorted(glob.glob(os.path.join(processed_dir, "*_minutiae.json")))
    if not json_files:
        logging.error("No processed data found! Run steps 1-4 first.")
        sys.exit(1)

    # Link JSONs to Images
    samples = []
    for jf in json_files:
        base = jf.replace("_minutiae.json", "")
        img = base + "_enhanced.jpg"
        if os.path.exists(img):
            fid = parse_processed_filename(os.path.basename(jf))
            if fid: samples.append({"id": fid, "json": jf, "img": img})
    
    logging.info(f"Loaded {len(samples)} valid samples from {len(set(s['id'] for s in samples))} unique fingers.")

    # Group by Finger ID
    groups = {}
    for s in samples:
        groups.setdefault(s['id'], []).append(s)
    unique_ids = list(groups.keys())

    # --- Generate Pairs ---
    genuine_pairs = []
    all_gen = []
    for fid in unique_ids:
        items = groups[fid]
        if len(items) > 1:
            all_gen.extend(list(itertools.combinations(items, 2)))
    
    if len(all_gen) > args.max_genuines:
        logging.info(f"Downsampling genuine pairs from {len(all_gen)} to {args.max_genuines}")
        genuine_pairs = random.sample(all_gen, args.max_genuines)
    else:
        genuine_pairs = all_gen

    impostor_pairs = []
    attempts = 0
    # Safety limit to prevent infinite loops
    max_attempts = args.max_impostors * 10 
    
    while len(impostor_pairs) < args.max_impostors and attempts < max_attempts:
        attempts += 1
        id1, id2 = random.sample(unique_ids, 2)
        s1 = random.choice(groups[id1])
        s2 = random.choice(groups[id2])
        
        pair = (s1, s2)
        # Check reverse pair too just in case
        if pair not in impostor_pairs and (s2, s1) not in impostor_pairs:
            impostor_pairs.append(pair)

    logging.info(f"Final Evaluation Set: {len(genuine_pairs)} Genuine, {len(impostor_pairs)} Impostor.")
    logging.info("Starting matching pipeline...")

    # --- Run Genuine Matching ---
    gen_scores = []
    logging.info("--- Processing Genuine Pairs ---")
    for i, (p1, p2) in enumerate(genuine_pairs):
        s = run_matching_pipeline(p1['json'], p1['img'], p2['json'], p2['img'])
        gen_scores.append(s)
        
        # LOGGING UPDATE: Log specific files
        f1 = os.path.basename(p1['json']).split('_minutiae')[0]
        f2 = os.path.basename(p2['json']).split('_minutiae')[0]
        logging.info(f"[GEN {i+1}/{len(genuine_pairs)}] {f1} vs {f2} -> Score: {s:.4f}")
    
    # --- Run Impostor Matching ---
    imp_scores = []
    logging.info("--- Processing Impostor Pairs ---")
    for i, (p1, p2) in enumerate(impostor_pairs):
        s = run_matching_pipeline(p1['json'], p1['img'], p2['json'], p2['img'])
        imp_scores.append(s)
        
        # LOGGING UPDATE: Log specific files
        f1 = os.path.basename(p1['json']).split('_minutiae')[0]
        f2 = os.path.basename(p2['json']).split('_minutiae')[0]
        logging.info(f"[IMP {i+1}/{len(impostor_pairs)}] {f1} vs {f2} -> Score: {s:.4f}")

    # --- Metrics Calculation ---
    thresholds = np.linspace(0, 1.0, 500) # Increased resolution for better EER precision
    far_list, tpr_list = [], []
    min_dist, eer, eer_thresh = float('inf'), 0.0, 0.0
    
    for t in thresholds:
        # FAR: Fraction of impostors accepted (score >= t)
        fa = sum(1 for s in imp_scores if s >= t) / len(imp_scores) if imp_scores else 0
        
        # FRR: Fraction of genuines rejected (score < t)
        fr = sum(1 for s in gen_scores if s < t) / len(genuine_pairs) if genuine_pairs else 0
        
        tpr = 1.0 - fr # TPR = 1 - FRR
        
        far_list.append(fa)
        tpr_list.append(tpr)
        
        # EER is where FAR ~= FRR
        if abs(fa - fr) < min_dist:
            min_dist = abs(fa - fr)
            eer, eer_thresh = (fa + fr)/2, t

    logging.info(f"Final Result -> EER: {eer:.4f} at Threshold: {eer_thresh:.4f}")

    # --- Save Metrics to Text File ---
    save_metrics_table(args.metrics_output, eer, eer_thresh, len(genuine_pairs), len(impostor_pairs))

    # --- Plot ROC ---
    plt.figure()
    plt.plot(far_list, tpr_list, label=f'ROC (EER={eer:.2f})', linewidth=2)
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve - Contactless Fingerprint Recognition')
    plt.grid(True)
    plt.legend()
    plt.savefig(args.plot_output)
    logging.info(f"ROC Curve saved to {args.plot_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', required=True, help="Directory containing processed JSONs/JPGs")
    parser.add_argument('--max_genuines', type=int, default=100)
    parser.add_argument('--max_impostors', type=int, default=100)
    parser.add_argument('--plot_output', default='roc_curve.png')
    parser.add_argument('--log_file', default='evaluation.log')
    parser.add_argument('--metrics_output', default='performance_metrics.txt', help="Output file for EER table")
    args = parser.parse_args()
    main(args)