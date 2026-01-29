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

def parse_processed_filename(filename):
    """
    Parses '1_8_1_5_enhanced.jpg' -> Subject=1, Finger=8
    Returns: unique_finger_id (Subject_Finger)
    """
    try:
        parts = filename.split('_')
        subject_id = parts[0]
        finger_id = parts[1]
        return f"{subject_id}_{finger_id}"
    except:
        return None

def run_matching_pipeline(t_json, t_img, q_json, q_img, temp_dir="temp_match"):
    os.makedirs(temp_dir, exist_ok=True)
    matrix_path = os.path.join(temp_dir, "W.npy")
    lga_path = os.path.join(temp_dir, "lga.npy")
    result_path = os.path.join(temp_dir, "result.json")
    
    try:
        # Step 5: Build Matrix
        subprocess.check_output([
            "python3", "src2/step5_build_matrix.py",
            "--template_json", t_json, "--template_img", t_img,
            "--query_json", q_json, "--query_img", q_img,
            "--output", os.path.splitext(matrix_path)[0]
        ], stderr=subprocess.STDOUT)
        
        with open(t_json) as f: Nt = len(json.load(f))
        with open(q_json) as f: Nq = len(json.load(f))
        
        if Nt < 5 or Nq < 5: return 0.0

        # Step 6: LGA
        subprocess.check_output([
            "python3", "src2/step6_lga.py",
            "--matrix_path", matrix_path, "--Nt", str(Nt), "--Nq", str(Nq),
            "--output", os.path.splitext(lga_path)[0]
        ], stderr=subprocess.STDOUT)
        
        # Step 7: Refinement
        subprocess.check_output([
            "python3", "src2/step7_refinement.py",
            "--lga_solution", lga_path, "--matrix_path", matrix_path,
            "--template_json", t_json, "--template_img", t_img,
            "--query_json", q_json, "--query_img", q_img,
            "--output", result_path
        ], stderr=subprocess.STDOUT)
        
        with open(result_path) as f:
            return float(json.load(f)['score'])
            
    except Exception:
        return 0.0

def main(args):
    processed_dir = args.processed_dir
    json_files = sorted(glob.glob(os.path.join(processed_dir, "*_minutiae.json")))
    
    if not json_files:
        print(f"No processed files found in {processed_dir}")
        sys.exit(1)

    samples = []
    for jf in json_files:
        base = jf.replace("_minutiae.json", "")
        img = base + "_enhanced.jpg"
        if os.path.exists(img):
            fid = parse_processed_filename(os.path.basename(base))
            if fid:
                samples.append({"id": fid, "json": jf, "img": img})
    
    print(f"Loaded {len(samples)} processed samples.")

    # Group by Finger Identity
    groups = {}
    for s in samples:
        groups.setdefault(s['id'], []).append(s)
        
    unique_ids = list(groups.keys())
    
    # Generate Pairs
    genuine_pairs = []
    impostor_pairs = []
    
    # Genuine: All unique pairs within the same finger group
    for fid in unique_ids:
        items = groups[fid]
        # Only compare if we have >1 image for this finger
        if len(items) > 1:
            # We limit to max 10 genuine pairs per finger to avoid explosion
            # (e.g., comparing Capture 1 vs Capture 2)
            pairs = list(itertools.combinations(items, 2))
            # Optional: shuffle and pick subset if too many
            if len(pairs) > 10: pairs = random.sample(pairs, 10)
            genuine_pairs.extend(pairs)

    # Impostor: Pairs from different finger groups
    while len(impostor_pairs) < args.max_impostors:
        id1, id2 = random.sample(unique_ids, 2)
        s1 = random.choice(groups[id1])
        s2 = random.choice(groups[id2])
        impostor_pairs.append((s1, s2))

    print(f"Evaluation: {len(genuine_pairs)} Genuine vs {len(impostor_pairs)} Impostor pairs.")

    # Run Matching
    gen_scores = []
    imp_scores = []
    
    print("Running Genuine...")
    for i, (p1, p2) in enumerate(genuine_pairs):
        s = run_matching_pipeline(p1['json'], p1['img'], p2['json'], p2['img'])
        gen_scores.append(s)
        print(f"[{i+1}/{len(genuine_pairs)}] Gen Score: {s:.4f}", end='\r')

    print("\nRunning Impostor...")
    for i, (p1, p2) in enumerate(impostor_pairs):
        s = run_matching_pipeline(p1['json'], p1['img'], p2['json'], p2['img'])
        imp_scores.append(s)
        print(f"[{i+1}/{len(impostor_pairs)}] Imp Score: {s:.4f}", end='\r')

    # Metrics
    thresholds = np.linspace(0, 1.0, 100)
    far, frr = [], []
    min_dist, eer, eer_thresh = float('inf'), 0.0, 0.0
    
    for t in thresholds:
        fa = sum(1 for s in imp_scores if s >= t) / len(imp_scores) if imp_scores else 0
        fr = sum(1 for s in gen_scores if s < t) / len(genuine_pairs) if genuine_pairs else 0
        far.append(fa)
        frr.append(fr)
        if abs(fa - fr) < min_dist:
            min_dist = abs(fa - fr)
            eer, eer_thresh = (fa + fr)/2, t

    print(f"\n\nEER: {eer:.4f} @ Threshold: {eer_thresh:.4f}")

    # Plot
    plt.figure()
    plt.plot(far, frr, label=f'EER={eer:.2f}')
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('FAR'); plt.ylabel('FRR'); plt.title('ROC Curve')
    plt.legend(); plt.grid(True)
    plt.savefig(args.plot_output)
    print(f"ROC saved to {args.plot_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', required=True)
    parser.add_argument('--max_impostors', type=int, default=100)
    parser.add_argument('--plot_output', default='roc_curve.png')
    args = parser.parse_args()
    main(args)