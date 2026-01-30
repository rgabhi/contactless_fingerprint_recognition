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
        return f"{parts[0]}_{parts[1]}"
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

        # Step 5: Build Matrix
        cmd_step5 = [
            "python3", "src/step5_build_matrix.py",
            "--template_json", t_json, "--template_img", t_img,
            "--query_json", q_json, "--query_img", q_img,
            "--output", os.path.splitext(matrix_path)[0]
        ]
        res5 = subprocess.run(cmd_step5, capture_output=True, text=True)
        if res5.returncode != 0:
            logging.error(f"Step 5 Failed: {res5.stderr}")
            return 0.0

        # Step 6: LGA
        cmd_step6 = [
            "python3", "src/step6_lga.py",
            "--matrix_path", matrix_path, "--Nt", str(Nt), "--Nq", str(Nq),
            "--output", os.path.splitext(lga_path)[0]
        ]
        res6 = subprocess.run(cmd_step6, capture_output=True, text=True)
        if res6.returncode != 0:
            logging.error(f"Step 6 Failed: {res6.stderr}")
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
            logging.error(f"Step 7 Failed: {res7.stderr}")
            return 0.0
        
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

    # --- Generate Pairs ---
    genuine_pairs = []
    impostor_pairs = []

    # Genuine
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

    # Impostor
    attempts = 0
    while len(impostor_pairs) < args.max_impostors and attempts < args.max_impostors * 5:
        attempts += 1
        id1, id2 = random.sample(unique_ids, 2)
        s1 = random.choice(groups[id1])
        s2 = random.choice(groups[id2])
        if (s1, s2) not in impostor_pairs:
            impostor_pairs.append((s1, s2))

    logging.info(f"Final Evaluation Set: {len(genuine_pairs)} Genuine, {len(impostor_pairs)} Impostor.")
    logging.info("Running matching pipeline...")

    # --- Run Matching ---
    gen_scores = []
    for i, (p1, p2) in enumerate(genuine_pairs):
        s = run_matching_pipeline(p1['json'], p1['img'], p2['json'], p2['img'])
        gen_scores.append(s)
        if i % 5 == 0: logging.info(f"Genuine Progress: {i}/{len(genuine_pairs)} (Last: {s:.4f})")
    
    imp_scores = []
    for i, (p1, p2) in enumerate(impostor_pairs):
        s = run_matching_pipeline(p1['json'], p1['img'], p2['json'], p2['img'])
        imp_scores.append(s)
        if i % 5 == 0: logging.info(f"Impostor Progress: {i}/{len(impostor_pairs)} (Last: {s:.4f})")

    # --- Metrics ---
    thresholds = np.linspace(0, 1.0, 200)
    far_list, tpr_list = [], []
    min_dist, eer, eer_thresh = float('inf'), 0.0, 0.0
    
    for t in thresholds:
        fa = sum(1 for s in imp_scores if s >= t) / len(imp_scores) if imp_scores else 0
        fr = sum(1 for s in gen_scores if s < t) / len(genuine_pairs) if genuine_pairs else 0
        tpr = 1.0 - fr
        
        far_list.append(fa)
        tpr_list.append(tpr)
        
        if abs(fa - fr) < min_dist:
            min_dist = abs(fa - fr)
            eer, eer_thresh = (fa + fr)/2, t

    logging.info(f"EER: {eer:.4f} at Threshold: {eer_thresh:.4f}")

    plt.figure()
    plt.plot(far_list, tpr_list, label=f'ROC (EER={eer:.2f})')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FAR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(args.plot_output)
    logging.info(f"ROC Curve saved to {args.plot_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', required=True)
    parser.add_argument('--max_genuines', type=int, default=100)
    parser.add_argument('--max_impostors', type=int, default=100)
    parser.add_argument('--plot_output', default='roc_curve.png')
    parser.add_argument('--log_file', default='evaluation.log')
    args = parser.parse_args()
    main(args)