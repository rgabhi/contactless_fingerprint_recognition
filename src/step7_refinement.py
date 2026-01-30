import argparse
import json
import numpy as np
import cv2
import math
import sys
from feature_utils import get_ridge_count

# CONSTANTS (Same as Step 5)
DELTA_RC = 5          # Was 3. Relaxed for noisy ridge extraction
DELTA_THETA = np.deg2rad(45) # Was 30 deg. Relaxed for perspective distortion

# --- Helper Class for JSON Saving ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
# ------------------------------------

def angle_diff(a1, a2):
    """Smallest difference between two angles in range [0, pi]"""
    diff = abs(a1 - a2)
    return min(diff, np.pi - diff)

def precompute_internal_features(img, minutiae):
    """
    Pre-computes Ridge Counts and Relative Orientations between ALL pairs within a fingerprint.
    Returns: RC_matrix, Relative_Theta_matrix
    """
    N = len(minutiae)
    RC = np.zeros((N, N), dtype=int)
    
    print(f"Pre-computing internal features for {N} minutiae...")
    for i in range(N):
        for j in range(i + 1, N):
            # Ridge Count
            rc = get_ridge_count(img, (minutiae[i]['x'], minutiae[i]['y']), 
                                      (minutiae[j]['x'], minutiae[j]['y']))
            RC[i, j] = RC[j, i] = rc
            
    return RC

def calculate_energy(matches, W, Nq):
    """
    Calculates f(x) = x^T * W * x for the given set of matches.
    matches: List of tuples [(t_idx, q_idx), ...]
    """
    score = 0.0
    active_indices = []
    
    # Convert (t, q) pairs to global indices in W
    for t, q in matches:
        k = t * Nq + q
        active_indices.append(k)
        
    if not active_indices:
        return 0.0

    # Extract submatrix for active matches and sum
    # This is equivalent to x^T * W * x where x is the binary vector
    sub_W = W[np.ix_(active_indices, active_indices)]
    score = np.sum(sub_W)
    
    return score

def main(args):
    # 1. Load Data
    print("Loading data...")
    try:
        W = np.load(args.matrix_path)
        lga_vector = np.load(args.lga_solution)
        with open(args.template_json) as f: t_minutiae = json.load(f)
        with open(args.query_json) as f: q_minutiae = json.load(f)
        t_img = cv2.imread(args.template_img, cv2.IMREAD_GRAYSCALE)
        q_img = cv2.imread(args.query_img, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    Nt = len(t_minutiae)
    Nq = len(q_minutiae)
    
    # 2. Pre-compute Internal Features (RC) needed for Eq 7
    # We do this once to avoid re-calculating pixel-level ridge counts in the loop
    RC_T = precompute_internal_features(t_img, t_minutiae)
    RC_Q = precompute_internal_features(q_img, q_minutiae)

    # 3. Initialize Sets (Algorithm 3)
    # T1, Q1: Indices of matched minutiae
    # T2, Q2: Indices of unpaired minutiae
    
    matches = []
    T1 = set()
    Q1 = set()
    
    # Parse LGA Skeleton
    for t_idx, val in enumerate(lga_vector):
        if val > 0:
            q_idx = val - 1 # 0-based
            matches.append((t_idx, q_idx))
            T1.add(t_idx)
            Q1.add(q_idx)
            
    T2 = [i for i in range(Nt) if i not in T1]
    Q2 = [j for j in range(Nq) if j not in Q1]
    
    print(f"Initial LGA Skeleton: {len(matches)} matches.")
    
    # 4. Expansion Loop
    print("Starting Minutia-Pair Expansion...")
    
    while T2 and Q2:
        best_pair = None
        min_dist = float('inf')
        
        # Check every unpaired candidate (i, j)
        for i in T2:
            for j in Q2:
                
                dist_sum = 0.0
                valid_candidate = True
                
                # Check against ALL established matches (k, k_prime) in (T1, Q1)
                # Eq 7 logic
                for k, k_prime in matches:
                    
                    # 1. Retrieve Ridge Counts
                    rc_ik = RC_T[i, k]
                    rc_jk = RC_Q[j, k_prime]
                    
                    # 2. Retrieve Relative Orientations
                    # psi_ik = diff(theta_i, theta_k)
                    psi_ik = angle_diff(t_minutiae[i]['theta'], t_minutiae[k]['theta'])
                    psi_jk = angle_diff(q_minutiae[j]['theta'], q_minutiae[k_prime]['theta'])
                    
                    # 3. Check Constraints
                    diff_rc = abs(rc_ik - rc_jk)
                    diff_theta = abs(psi_ik - psi_jk)
                    
                    if diff_theta > DELTA_THETA or diff_rc > DELTA_RC:
                        valid_candidate = False
                        break # Constraints failed, d = infinity
                    
                    # 4. Sum Distance
                    dist_sum += diff_rc
                
                if valid_candidate:
                    if dist_sum < min_dist:
                        min_dist = dist_sum
                        best_pair = (i, j)
        
        # Selection
        if best_pair is not None:
            # Add to matches
            matches.append(best_pair)
            
            # Move from T2/Q2 to T1/Q1
            i_sel, j_sel = best_pair
            T1.add(i_sel)
            Q1.add(j_sel)
            T2.remove(i_sel)
            Q2.remove(j_sel)
            
        else:
            # Termination: No valid pairs found
            break

    print(f"Final Refined Set: {len(matches)} matches.")
    
    # 5. Final Scoring (Eq 8)
    # Score = 2 * f(x) / (N_match * (Nt + Nq))
    
    energy_f = calculate_energy(matches, W, Nq)
    N_match = len(matches)
    
    if N_match > 0:
        final_score = (2 * energy_f) / (N_match * (Nt + Nq))
    else:
        final_score = 0.0

    print("-" * 30)
    print(f"Final Energy f(x): {energy_f:.4f}")
    print(f"Final Normalized Score: {final_score:.6f}")
    print("-" * 30)
    
    # Save Results
    results = {
        "final_matches": matches,
        "energy": energy_f,
        "score": final_score,
        "Nt": Nt,
        "Nq": Nq
    }
    
    # FIX: Use NumpyEncoder here
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Part 4: Refinement & Scoring')
    parser.add_argument('--lga_solution', required=True, help='LGA skeleton vector (.npy)')
    parser.add_argument('--matrix_path', required=True, help='Similarity Matrix W (.npy)')
    parser.add_argument('--template_json', required=True, help='Template minutiae JSON')
    parser.add_argument('--template_img', required=True, help='Template enhanced image')
    parser.add_argument('--query_json', required=True, help='Query minutiae JSON')
    parser.add_argument('--query_img', required=True, help='Query enhanced image')
    parser.add_argument('--output', required=True, help='Output result JSON')
    
    args = parser.parse_args()
    main(args)