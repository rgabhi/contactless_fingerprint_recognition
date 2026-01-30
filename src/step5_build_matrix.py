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

def main(args):
    # 1. Load Data
    with open(args.template_json) as f: t_minutiae = json.load(f)
    with open(args.query_json) as f: q_minutiae = json.load(f)
    
    t_img = cv2.imread(args.template_img, cv2.IMREAD_GRAYSCALE)
    q_img = cv2.imread(args.query_img, cv2.IMREAD_GRAYSCALE)

    if t_img is None or q_img is None:
        sys.exit(1)
        
    Nt = len(t_minutiae)
    Nq = len(q_minutiae)
    
    # Safety Check: Matrix size should now be reasonable (~160MB max)
    if Nt * Nq > 250000:
        print(f"Error: Matrix too large ({Nt}x{Nq}). Skipping.")
        sys.exit(1)

    # 2. Pre-compute Features
    t_omap = compute_orientation_map(t_img)
    q_omap = compute_orientation_map(q_img)

    t_decriptors = [get_local_descriptor(m['x'], m['y'], t_omap) for m in t_minutiae]
    q_decriptors = [get_local_descriptor(m['x'], m['y'], q_omap) for m in q_minutiae]

    # 3. Calculate Diagonal Scores
    gamma_matrix = np.zeros((Nt, Nq), dtype=np.float32)

    for i in range(Nt):
        for j in range(Nq):
            gamma = calculate_gamma(t_decriptors[i], q_decriptors[j])
            gamma_matrix[i, j] = gamma

    gamma_mean = np.mean(gamma_matrix)
    gamma_min = np.min(gamma_matrix)
    gamma_max = np.max(gamma_matrix)

    denom = gamma_max - gamma_min
    if denom == 0: denom = 1.0
    
    S_aa_matrix = np.zeros((Nt, Nq), dtype=np.float32)
    for i in range(Nt):
        for j in range(Nq):
            gamma_prime = gamma_matrix[i, j] - gamma_mean
            gamma_min_prime = gamma_min - gamma_mean
            S_aa_matrix[i, j] = 2 * ((gamma_prime - gamma_min_prime) / denom) - 1

    # 4. Construct Full Matrix W (float32)
    dim = Nt * Nq
    W = np.zeros((dim, dim), dtype=np.float32)

    RC_T = np.zeros((Nt, Nt), dtype=np.int16)
    RC_Q = np.zeros((Nq, Nq), dtype=np.int16)

    for i in range(Nt):
        for j in range(i + 1, Nt):
            rc = get_ridge_count(t_img, (t_minutiae[i]['x'], t_minutiae[i]['y']), 
                                        (t_minutiae[j]['x'], t_minutiae[j]['y']))
            RC_T[i, j] = RC_T[j, i] = rc
    
    for i in range(Nq):
        for j in range(i + 1, Nq):
            rc = get_ridge_count(q_img, (q_minutiae[i]['x'], q_minutiae[i]['y']), 
                                        (q_minutiae[j]['x'], q_minutiae[j]['y']))
            RC_Q[i, j] = RC_Q[j, i] = rc

    # Vectorized construction
    RC_T_big = np.repeat(np.repeat(RC_T, Nq, axis=0), Nq, axis=1)
    RC_Q_big = np.tile(RC_Q, (Nt, Nt))
    Diff_RC = np.abs(RC_T_big - RC_Q_big)
    
    thetas_t = np.array([m['theta'] for m in t_minutiae])
    thetas_q = np.array([m['theta'] for m in q_minutiae])
    
    def pairwise_angle_diff(thetas):
        diff = np.abs(thetas[:, None] - thetas)
        return np.minimum(diff, np.pi - diff)

    Psi_T = pairwise_angle_diff(thetas_t)
    Psi_Q = pairwise_angle_diff(thetas_q)
    
    Psi_T_big = np.repeat(np.repeat(Psi_T, Nq, axis=0), Nq, axis=1)
    Psi_Q_big = np.tile(Psi_Q, (Nt, Nt))
    
    Diff_Theta = np.abs(Psi_T_big - Psi_Q_big)
    Diff_Theta = np.minimum(Diff_Theta, np.pi - Diff_Theta)
    
    coords_t = np.array([[m['x'], m['y']] for m in t_minutiae])
    coords_q = np.array([[m['x'], m['y']] for m in q_minutiae])
    
    Dist_T = np.sqrt(np.sum((coords_t[:, None] - coords_t)**2, axis=-1))
    Dist_Q = np.sqrt(np.sum((coords_q[:, None] - coords_q)**2, axis=-1))
    
    Dist_T_big = np.repeat(np.repeat(Dist_T, Nq, axis=0), Nq, axis=1)
    Dist_Q_big = np.tile(Dist_Q, (Nt, Nt))
    
    Diff_Dist = np.abs(Dist_T_big - Dist_Q_big)

    Term1 = np.where(Diff_RC < DELTA_RC, 1.0, -1.0)
    Term2 = np.where(Diff_Theta < DELTA_THETA, 1.0, -1.0)
    Term3 = np.where(Diff_Dist < DELTA_DIST, 1.0, -1.0)
    
    W = (Term1 * Term2 * Term3).astype(np.float32)
    
    S_aa_flat = S_aa_matrix.flatten()
    np.fill_diagonal(W, S_aa_flat)
    
    I_idx = np.repeat(np.arange(Nt), Nq)
    I_prime_idx = np.tile(np.arange(Nq), Nt)
    
    i_mat = np.tile(I_idx[:, None], (1, Nt*Nq))
    j_mat = np.tile(I_idx[None, :], (Nt*Nq, 1))
    ip_mat = np.tile(I_prime_idx[:, None], (1, Nt*Nq))
    jp_mat = np.tile(I_prime_idx[None, :], (Nt*Nq, 1))
    
    conflict_1 = (i_mat == j_mat) & (ip_mat != jp_mat)
    conflict_2 = (i_mat != j_mat) & (ip_mat == jp_mat)
    
    W[conflict_1 | conflict_2] = 0.0
    
    np.save(args.output, W)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_json', required=True)
    parser.add_argument('--template_img', required=True)
    parser.add_argument('--query_json', required=True)
    parser.add_argument('--query_img', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args)