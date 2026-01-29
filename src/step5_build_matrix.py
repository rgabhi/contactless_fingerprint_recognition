import argparse
import json
import numpy as np
import cv2
import math
import sys
from feature_utils import compute_orientation_map, get_local_descriptor, get_ridge_count

# CONSTANTS
MU = 16.0
L_CIRCLES = 5
K_POINTS = 8
R_RADIUS_STEP = 10 

# Thresholds
DELTA_RC = 3 
DELTA_THETA = np.deg2rad(30) 
DELTA_DIST = 30 

def angle_diff(a1, a2):
    diff = abs(a1 - a2)
    return min(diff, np.pi - diff)

def calculate_gamma(desc1, desc2):
    total_score = 0
    count = len(desc1)
    if count == 0: return 0
    for i in range(count):
        delta = angle_diff(desc1[i], desc2[i])
        term = math.exp(-(2 * MU / np.pi) * delta)
        total_score += term
    return (1.0 / count) * total_score

def psi_function(val, threshold):
    return 1 if val < threshold else -1

def main(args):
    # 1. Load Data
    print("Loading data...")
    with open(args.template_json) as f: t_minutiae = json.load(f)
    with open(args.query_json) as f: q_minutiae = json.load(f)
    
    t_img = cv2.imread(args.template_img, cv2.IMREAD_GRAYSCALE)
    q_img = cv2.imread(args.query_img, cv2.IMREAD_GRAYSCALE)

    if t_img is None or q_img is None:
        print("Error loading images")
        sys.exit(1)
        
    Nt = len(t_minutiae)
    Nq = len(q_minutiae)
    print(f"Template Size (Nt): {Nt}, Query Size (Nq): {Nq}")
    print(f"Total Matrix Size: ({Nt*Nq} x {Nt*Nq})")

    # 2. Pre-compute Features
    print("Computing Orientation Maps and Descriptors...")
    t_omap = compute_orientation_map(t_img)
    q_omap = compute_orientation_map(q_img)

    t_decriptors = [get_local_descriptor(m['x'], m['y'], t_omap) for m in t_minutiae]
    q_decriptors = [get_local_descriptor(m['x'], m['y'], q_omap) for m in q_minutiae]

    # 3. Calculate Diagonal Scores
    print("Calculating Diagonal Scores (S_aa)...")
    gamma_matrix = np.zeros((Nt, Nq))

    for i in range(Nt):
        for j in range(Nq):
            # FIX: Use q_decriptors[j], not [i]
            gamma = calculate_gamma(t_decriptors[i], q_decriptors[j])
            gamma_matrix[i, j] = gamma

    # Normalization
    gamma_mean = np.mean(gamma_matrix)
    gamma_min = np.min(gamma_matrix)
    gamma_max = np.max(gamma_matrix)

    denom = gamma_max - gamma_min
    if denom == 0: denom = 1.0
    
    S_aa_matrix = np.zeros((Nt, Nq))
    for i in range(Nt):
        for j in range(Nq):
            gamma_prime = gamma_matrix[i, j] - gamma_mean
            gamma_min_prime = gamma_min - gamma_mean
            S_aa_matrix[i, j] = 2 * ((gamma_prime - gamma_min_prime) / denom) - 1

    # 4. Construct Full Matrix W
    dim = Nt * Nq
    W = np.zeros((dim, dim), dtype=np.float32)

    print("Constructing W (This may take a moment)...")
    print("Pre-computing Ridge Counts...")
    RC_T = np.zeros((Nt, Nt))
    RC_Q = np.zeros((Nq, Nq))

    for i in range(Nt):
        for j in range(i + 1, Nt):
            rc = get_ridge_count(t_img, (t_minutiae[i]['x'], t_minutiae[i]['y']), 
                                        (t_minutiae[j]['x'], t_minutiae[j]['y']))
            RC_T[i, j] = RC_T[j, i] = rc
    
    for i in range(Nq):
        # FIX: Loop range from i+1, not 1+1
        for j in range(i + 1, Nq):
            # FIX: Added comma after image argument
            rc = get_ridge_count(q_img, (q_minutiae[i]['x'], q_minutiae[i]['y']), 
                                        (q_minutiae[j]['x'], q_minutiae[j]['y']))
            RC_Q[i, j] = RC_Q[j, i] = rc

    # Fill W
    for i in range(Nt):
        for i_prime in range(Nq):
            row_idx = i * Nq + i_prime

            # 1. Diagonal element
            W[row_idx, row_idx] = S_aa_matrix[i, i_prime]

            # 2. Off-Diagonal Elements
            for j in range(Nt):
                for j_prime in range(Nq):
                    col_idx = j * Nq + j_prime

                    if row_idx == col_idx: continue 

                    # Conflict constraint
                    if (i == j and i_prime != j_prime) or (i != j and i_prime == j_prime):
                        W[row_idx, col_idx] = 0.0
                        continue
                
                    # Calculate Edge Similarity
                    rc_t = RC_T[i, j]
                    rc_q = RC_Q[i_prime, j_prime]
                    diff_rc = abs(rc_t - rc_q)

                    psi_t = angle_diff(t_minutiae[i]['theta'], t_minutiae[j]['theta'])
                    psi_q = angle_diff(q_minutiae[i_prime]['theta'], q_minutiae[j_prime]['theta'])
                    diff_theta = abs(psi_t - psi_q)

                    dist_t = math.hypot(t_minutiae[i]['x'] - t_minutiae[j]['x'], 
                                        t_minutiae[i]['y'] - t_minutiae[j]['y'])
                    dist_q = math.hypot(q_minutiae[i_prime]['x'] - q_minutiae[j_prime]['x'], 
                                        q_minutiae[i_prime]['y'] - q_minutiae[j_prime]['y'])
                    diff_dist = abs(dist_t - dist_q)

                    term1 = psi_function(diff_rc, DELTA_RC)
                    term2 = psi_function(diff_theta, DELTA_THETA)
                    term3 = psi_function(diff_dist, DELTA_DIST)

                    S_ab = term1 * term2 * term3
                    W[row_idx, col_idx] = S_ab
    
    # 5. Save Matrix
    output_file = args.output
    np.save(output_file, W)
    print(f"Similarity Matrix W constructed and saved to {output_file}.npy")
    print(f"Shape: {W.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Part 2.2: Construct Similarity Matrix W')
    parser.add_argument('--template_json', required=True, help='Template minutiae JSON')
    parser.add_argument('--template_img', required=True, help='Template enhanced image')
    parser.add_argument('--query_json', required=True, help='Query minutiae JSON')
    parser.add_argument('--query_img', required=True, help='Query enhanced image')
    parser.add_argument('--output', required=True, help='Output filename for Matrix W')
    args = parser.parse_args()
    main(args)