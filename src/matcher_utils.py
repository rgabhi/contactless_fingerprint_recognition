import numpy as np
from skimage.draw import line
import cv2

# =========================================================
# 1. Feature Extraction Helpers
# =========================================================

def get_ridge_count(skeleton, p1, p2):
    """Calculate ridge count between two minutiae points on a skeleton image."""
    x1, y1 = p1
    x2, y2 = p2
    
    # skimage.draw.line expects (row, col) -> (y, x)
    rr, cc = line(int(y1), int(x1), int(y2), int(x2))
    
    # Clip coordinates to be within image bounds
    h, w = skeleton.shape
    rr = np.clip(rr, 0, h - 1)
    cc = np.clip(cc, 0, w - 1)
    
    line_pixels = skeleton[rr, cc]
    
    # Count 0 -> 255 transitions (entering a ridge)
    # Assuming ridges are 255 (white) and background is 0 (black)
    ridge_count = np.sum((line_pixels[:-1] < 128) & (line_pixels[1:] >= 128))
    
    return int(ridge_count)

def compute_orientation_field(img, sigma=5):
    """Computes the dense orientation field of the fingerprint."""
    # Ensure image is float32 for calculations
    img = img.astype(np.float32) / 255.0
    
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    gxx = cv2.GaussianBlur(gx ** 2, (0, 0), sigma)
    gyy = cv2.GaussianBlur(gy ** 2, (0, 0), sigma)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    
    theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy) + (np.pi / 2)
    return theta

def get_local_descriptor(minutia, orientation_map, radii=[20, 40, 60], K=8):
    """Compute rotation-invariant local descriptor."""
    # Unpack minutia (handling both tuple/list and dictionary input)
    if isinstance(minutia, dict):
        x_m, y_m, theta_m = minutia['x'], minutia['y'], minutia['angle']
    else:
        x_m, y_m, theta_m = minutia[0], minutia[1], minutia[2]

    h, w = orientation_map.shape
    descriptor = []
    
    for r in radii:
        for k in range(K):
            alpha_k = theta_m + (2 * np.pi * k) / K
            x_s = int(round(x_m + r * np.cos(alpha_k)))
            y_s = int(round(y_m + r * np.sin(alpha_k)))
            
            if 0 <= x_s < w and 0 <= y_s < h:
                theta_s = orientation_map[y_s, x_s]
                feature = theta_s - theta_m
                feature = (feature + np.pi) % (2 * np.pi) - np.pi
            else:
                feature = 0.0
            descriptor.append(feature)
            
    return np.array(descriptor, dtype=np.float32)

def precompute_ridge_counts(skeleton, minutiae):
    """Precompute ridge counts between all pairs of minutiae."""
    N = len(minutiae)
    rc_matrix = np.zeros((N, N), dtype=np.int32)
    
    # Extract coordinates list for faster indexing
    coords = []
    for m in minutiae:
        if isinstance(m, dict):
             coords.append((m['x'], m['y']))
        else:
             coords.append((m[0], m[1]))

    for i in range(N):
        p1 = coords[i]
        for j in range(i + 1, N):
            p2 = coords[j]
            rc = get_ridge_count(skeleton, p1, p2)
            rc_matrix[i, j] = rc
            rc_matrix[j, i] = rc
    return rc_matrix

# =========================================================
# 2. Compatibility & Similarity Helpers
# =========================================================

def compute_minutia_similarity(desc_a, desc_b, mu=16):
    """Compute raw minutia-wise similarity gamma."""
    diff = np.abs(desc_a - desc_b)
    delta = np.minimum(diff, 2 * np.pi - diff)
    sim = np.exp(-(2 * mu / np.pi) * delta)
    return float(np.mean(sim))

def compute_edge_similarity(rc_ij, rc_i_j_prime, lam=0.5):
    """Compute edge (ridge-count) similarity."""
    diff = abs(rc_ij - rc_i_j_prime)
    return float(np.exp(-lam * diff))

def compute_compatibility_matrix(minutiae1, desc1, rc1,
                                 minutiae2, desc2, rc2, 
                                 similarity_threshold=0.3):
    """
    Build the compatibility matrix W.
    OPTIMIZATION: Prunes assignments based on descriptor similarity.
    Only pairs with similarity > threshold are added to the graph.
    """
    N1 = len(minutiae1)
    N2 = len(minutiae2)
    
    assignments = []
    
    # 1. Pruning Step: Identify plausible pairs
    # This prevents creating a massive NxM matrix.
    for i in range(N1):
        for j in range(N2):
            sim = compute_minutia_similarity(desc1[i], desc2[j])
            if sim > similarity_threshold:
                # Store (index_T, index_Q, similarity_score)
                assignments.append((i, j, sim))
    
    M = len(assignments)
    W = np.zeros((M, M), dtype=np.float32)

    # 2. Fill W for only valid assignments
    # 'assignments' list contains the mapping from Matrix Index u -> (i, j)
    # We strip the score for the final list used by GA
    final_assignments_map = [(idx[0], idx[1]) for idx in assignments]

    for u in range(M):
        i, i_p, node_sim = assignments[u]
        
        # Diagonal: Node Similarity
        W[u, u] = node_sim
        
        for v in range(u + 1, M):
            j, j_p, _ = assignments[v]
            
            # Conflict check (One-to-One constraint)
            if i == j or i_p == j_p:
                compat = 0.0
            else:
                # Edge Compatibility
                rc_ij = rc1[i, j]
                rc_i_jp = rc2[i_p, j_p]
                compat = compute_edge_similarity(rc_ij, rc_i_jp)
            
            if compat > 0:
                W[u, v] = compat
                W[v, u] = compat # Symmetry
            
    return W, final_assignments_map

# =========================================================
# 3. Refinement & Scoring
# =========================================================

def signed_angle_diff(a, b):
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi

def expand_minutia_pairs(minutiae1, minutiae2, rc1, rc2, ga_result, 
                         delta_theta=np.pi/6, delta_rc=1):
    N1 = len(minutiae1)
    N2 = len(minutiae2)
    
    # Helper to access angle regardless of dict/tuple format
    def get_angle(m):
        return m['angle'] if isinstance(m, dict) else m[2]

    T1, Q1 = set(), set()
    matches = {}
    
    for i, val in enumerate(ga_result):
        if val > 0:
            j = val - 1
            matches[i] = j
            T1.add(i)
            Q1.add(j)
            
    T2 = set(range(N1)) - T1
    Q2 = set(range(N2)) - Q1
    
    while True:
        best_pair = None
        best_dist = np.inf
        
        for i in T2:
            theta_i = get_angle(minutiae1[i])
            for j in Q2:
                theta_j = get_angle(minutiae2[j])
                
                valid = True
                dist_ij = 0.0
                
                for k, k_p in matches.items():
                    psi_ik = signed_angle_diff(get_angle(minutiae1[k]), theta_i)
                    psi_jk = signed_angle_diff(get_angle(minutiae2[k_p]), theta_j)
                    
                    if abs(signed_angle_diff(psi_ik, psi_jk)) > delta_theta:
                        valid = False
                        break
                    
                    rc_ik = rc1[i, k]
                    rc_jk = rc2[j, k_p]
                    diff_rc = abs(rc_ik - rc_jk)
                    
                    if diff_rc > delta_rc:
                        valid = False
                        break
                        
                    dist_ij += diff_rc 
                
                if valid and dist_ij < best_dist:
                    best_dist = dist_ij
                    best_pair = (i, j)
        
        if best_pair is None:
            break
            
        i_best, j_best = best_pair
        matches[i_best] = j_best
        T1.add(i_best)
        Q1.add(j_best)
        T2.remove(i_best)
        Q2.remove(j_best)
        
    return [(i, j) for i, j in matches.items()]

def calculate_comparison_score(final_pairs, W, assignments, n_t, n_q):
    pair_to_idx = {pair: u for u, pair in enumerate(assignments)}
    # Only use pairs that actually exist in our pruned assignments
    active_indices = [pair_to_idx[pair] for pair in final_pairs if pair in pair_to_idx]
    
    if not active_indices:
        return 0.0
    
    # Fix for np.ix_: ensure indices are valid integers
    if len(active_indices) > 0:
        energy = np.sum(W[np.ix_(active_indices, active_indices)])
    else:
        energy = 0.0

    N_match = len(final_pairs)
    
    if N_match == 0 or (n_t + n_q) == 0:
        return 0.0
        
    return (2 * energy) / (N_match * (n_t + n_q))

# =========================================================
# 4. Evaluation & Labeling Utils
# =========================================================

def get_match_label(filename_a, filename_b):
    import os
    name_a = os.path.splitext(os.path.basename(filename_a))[0]
    name_b = os.path.splitext(os.path.basename(filename_b))[0]
    parts_a = name_a.split('_')
    parts_b = name_b.split('_')
    if len(parts_a) < 3 or len(parts_b) < 3: return "Imposter"
    # Adjust indices based on your filename format (e.g. 1_1_1_0.bmp)
    # Subject_Finger_Capture_Index
    return "Genuine" if (parts_a[0] == parts_b[0] and parts_a[1] == parts_b[1]) else "Imposter"



def calculate_error_rates(threshold, genuine_scores, imposter_scores):
    if len(genuine_scores) == 0 or len(imposter_scores) == 0: return 0.0, 0.0
    false_accepts = sum(score >= threshold for score in imposter_scores)
    FAR = (false_accepts / len(imposter_scores)) * 100
    false_rejects = sum(score < threshold for score in genuine_scores)
    FRR = (false_rejects / len(genuine_scores)) * 100
    return FAR, FRR

def find_eer(genuine_scores, imposter_scores):
    best_threshold, min_diff, eer = None, float('inf'), None
    for threshold in np.linspace(0, 1, 1000):
        FAR, FRR = calculate_error_rates(threshold, genuine_scores, imposter_scores)
        diff = abs(FAR - FRR)
        if diff < min_diff:
            min_diff = diff
            best_threshold = threshold
            eer = (FAR + FRR) / 2.0
    return eer, best_threshold

def get_roc_data(genuine_scores, imposter_scores, num_points=100):
    thresholds = np.linspace(0, 1, num_points)
    far_list, tar_list = [], []
    for t in thresholds:
        far, frr = calculate_error_rates(t, genuine_scores, imposter_scores)
        far_list.append(far)
        tar_list.append(100 - frr)
    return far_list, tar_list