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

    # ... (Previous code for RC_T, RC_Q calculations remains the same) ...

    # 4. Construct Full Matrix W (Vectorized)
    print("Constructing W using Vectorization...")
    
    # Create the mapping from global index k to (i, i_prime)
    # Row indices (u): Template i, Query i_prime
    # Col indices (v): Template j, Query j_prime
    
    # Expand RC matrices to (Nt*Nq, Nt*Nq)
    # RC_T_big[u, v] should depend on i, j. Repeat repeats elements, Tile repeats whole matrix.
    # RC_T (Nt x Nt) -> Repeat rows Nq times, Repeat cols Nq times
    RC_T_big = np.repeat(np.repeat(RC_T, Nq, axis=0), Nq, axis=1)
    
    # RC_Q_big[u, v] should depend on i_prime, j_prime.
    # RC_Q (Nq x Nq) -> Tile the whole matrix Nt x Nt times
    RC_Q_big = np.tile(RC_Q, (Nt, Nt))
    
    # 1. Ridge Count Difference
    Diff_RC = np.abs(RC_T_big - RC_Q_big)
    
    # 2. Orientation Difference
    # Extract Thetas
    thetas_t = np.array([m['theta'] for m in t_minutiae])
    thetas_q = np.array([m['theta'] for m in q_minutiae])
    
    # Compute Pairwise diffs within T and Q
    # psi_T[i, j] = diff(theta_i, theta_j)
    # Using broadcasting: |A - A.T| with wrap-around
    def pairwise_angle_diff(thetas):
        diff = np.abs(thetas[:, None] - thetas)
        return np.minimum(diff, np.pi - diff)

    Psi_T = pairwise_angle_diff(thetas_t) # Nt x Nt
    Psi_Q = pairwise_angle_diff(thetas_q) # Nq x Nq
    
    # Expand like RC
    Psi_T_big = np.repeat(np.repeat(Psi_T, Nq, axis=0), Nq, axis=1)
    Psi_Q_big = np.tile(Psi_Q, (Nt, Nt))
    
    # Theta Difference between edges
    Diff_Theta = np.abs(Psi_T_big - Psi_Q_big)
    Diff_Theta = np.minimum(Diff_Theta, np.pi - Diff_Theta) # Handle angle wrap
    
    # 3. Distance Difference
    # Compute pairwise Euclidean dists for T and Q
    coords_t = np.array([[m['x'], m['y']] for m in t_minutiae])
    coords_q = np.array([[m['x'], m['y']] for m in q_minutiae])
    
    # Dist matrix: sqrt((x-x')^2 + (y-y')^2)
    # Scipy would be easier, but using pure numpy:
    Dist_T = np.sqrt(np.sum((coords_t[:, None] - coords_t)**2, axis=-1))
    Dist_Q = np.sqrt(np.sum((coords_q[:, None] - coords_q)**2, axis=-1))
    
    Dist_T_big = np.repeat(np.repeat(Dist_T, Nq, axis=0), Nq, axis=1)
    Dist_Q_big = np.tile(Dist_Q, (Nt, Nt))
    
    Diff_Dist = np.abs(Dist_T_big - Dist_Q_big)

    # --- CALCULATE SCORES ---
    # Apply Thresholds (Eq 6) - Terms are 1 if < delta, -1 otherwise
    Term1 = np.where(Diff_RC < DELTA_RC, 1.0, -1.0)
    Term2 = np.where(Diff_Theta < DELTA_THETA, 1.0, -1.0)
    Term3 = np.where(Diff_Dist < DELTA_DIST, 1.0, -1.0)
    
    W = Term1 * Term2 * Term3
    
    # --- APPLY CONSTRAINTS ---
    # Diagonal S_aa
    # S_aa_matrix is Nt x Nq. Flatten it to diagonal.
    S_aa_flat = S_aa_matrix.flatten() # Order is i=0(j=0..Nq), i=1...
    np.fill_diagonal(W, S_aa_flat)
    
    # Conflict Constraint:
    # W[u, v] = 0 if (i == j and i' != j') OR (i != j and i' == j')
    # Create index grids
    I_idx = np.repeat(np.arange(Nt), Nq) # [0,0,0, 1,1,1...]
    I_prime_idx = np.tile(np.arange(Nq), Nt) # [0,1,2, 0,1,2...]
    
    # Grid of u (rows) and v (cols)
    # We need masks for u vs v
    # i_grid[u, v] is the template index 'i' for row u
    i_mat = np.tile(I_idx[:, None], (1, Nt*Nq))
    j_mat = np.tile(I_idx[None, :], (Nt*Nq, 1))
    
    ip_mat = np.tile(I_prime_idx[:, None], (1, Nt*Nq))
    jp_mat = np.tile(I_prime_idx[None, :], (Nt*Nq, 1))
    
    # Conflict Mask
    # Condition 1: Same Template, Different Query
    conflict_1 = (i_mat == j_mat) & (ip_mat != jp_mat)
    # Condition 2: Different Template, Same Query
    conflict_2 = (i_mat != j_mat) & (ip_mat == jp_mat)
    
    W[conflict_1 | conflict_2] = 0.0
    
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