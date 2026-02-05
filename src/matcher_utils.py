import numpy as np
from skimage.draw import line
import cv2


#######
# Feature Extraction Helpers


def get_ridge_count(skeleton, p1, p2):
    """Calculate ridge count between two minutiae points on a skeleton image."""
    x1, y1 = p1
    x2, y2 = p2
    
    rr, cc = line(int(y1), int(x1), int(y2), int(x2))
    h, w = skeleton.shape
    rr = np.clip(rr, 0, h - 1)
    cc = np.clip(cc, 0, w - 1)
    
    line_pixels = skeleton[rr, cc]
    ridge_count = np.sum((line_pixels[:-1] < 128) & (line_pixels[1:] >= 128))
    return int(ridge_count)

def compute_orientation_field(img, sigma=5):
    img = img.astype(np.float32) / 255.0
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gxx = cv2.GaussianBlur(gx ** 2, (0, 0), sigma)
    gyy = cv2.GaussianBlur(gy ** 2, (0, 0), sigma)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy) + (np.pi / 2)
    return theta

def get_local_descriptor(minutia, orientation_map, radii=[20, 40, 60], K=8):
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
    N = len(minutiae)
    rc_matrix = np.zeros((N, N), dtype=np.int32)
    coords = []
    for m in minutiae:
        if isinstance(m, dict): coords.append((m['x'], m['y']))
        else: coords.append((m[0], m[1]))

    for i in range(N):
        p1 = coords[i]
        for j in range(i + 1, N):
            p2 = coords[j]
            rc = get_ridge_count(skeleton, p1, p2)
            rc_matrix[i, j] = rc
            rc_matrix[j, i] = rc
    return rc_matrix


#####
# Compatibility & Similarity Helpers


def compute_minutia_similarity_raw(desc_a, desc_b, mu=4):
    """
    Compute raw minutia-wise similarity.
    CHANGED: mu=4 (Lower is more tolerant). 
    If mu is too high (16), small orientation errors kill the score.
    """
    diff = np.abs(desc_a - desc_b)
    delta = np.minimum(diff, 2 * np.pi - diff)
    sim = np.exp(-(2 * mu / np.pi) * delta)
    return float(np.mean(sim))

def compute_compatibility_matrix(minutiae1, desc1, rc1,
                                 minutiae2, desc2, rc2, 
                                 similarity_threshold=0.1): # Keep low to allow potential matches
    """
    Build the compatibility matrix W with Absolute Normalization.
    """
    N1 = len(minutiae1)
    N2 = len(minutiae2)
    
    coords1 = np.array([[m['x'], m['y']] if isinstance(m, dict) else [m[0], m[1]] for m in minutiae1])
    coords2 = np.array([[m['x'], m['y']] if isinstance(m, dict) else [m[0], m[1]] for m in minutiae2])
    
    # 1. Pruning
    assignments = []
    raw_sims = {} # Map u -> raw_similarity
    
    for i in range(N1):
        for j in range(N2):
            val = compute_minutia_similarity_raw(desc1[i], desc2[j])
            if val > similarity_threshold:
                assignments.append((i, j))
                raw_sims[len(assignments)-1] = val
    
    # Fill W
    M = len(assignments)
    W = np.zeros((M, M), dtype=np.float32)
    
    delta_rc = 1             
    delta_theta = np.pi / 6  
    delta_d = 25             

    def get_angle(m): return m['angle'] if isinstance(m, dict) else m[2]
    
    for u in range(M):
        i, j = assignments[u]
        
        raw_val = raw_sims[u]
        W[u, u] = 2.0 * raw_val - 1.0
        
        for v in range(u + 1, M):
            i_p, j_p = assignments[v]
            
            # 1-to-1 constraint
            if i == i_p or j == j_p:
                W[u, v] = -1.0 
                W[v, u] = -1.0
                continue

            # ridge count Difference
            rc_diff = abs(rc1[i, i_p] - rc2[j, j_p])
            
            # orientation Difference
            def angle_diff(a1, a2):
                d = a1 - a2
                return (d + np.pi) % (2 * np.pi) - np.pi

            psi_i = angle_diff(get_angle(minutiae1[i]), get_angle(minutiae1[i_p]))
            psi_j = angle_diff(get_angle(minutiae2[j]), get_angle(minutiae2[j_p]))
            theta_diff = abs(angle_diff(psi_i, psi_j))
            
            # euclidean Distance Difference
            d1 = np.linalg.norm(coords1[i] - coords1[i_p])
            d2 = np.linalg.norm(coords2[j] - coords2[j_p])
            dist_diff = abs(d1 - d2)
            
            # strict compatibility check
            is_compat = (rc_diff <= delta_rc) and \
                        (theta_diff <= delta_theta) and \
                        (dist_diff <= delta_d)
            
            edge_weight = 1.0 if is_compat else -1.0
            
            W[u, v] = edge_weight
            W[v, u] = edge_weight

    return W, assignments

# Refinement & Scoring

def signed_angle_diff(a, b):
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi

def expand_minutia_pairs(minutiae1, minutiae2, rc1, rc2, ga_result, 
                         delta_theta=np.pi/6, delta_rc=1):
    N1 = len(minutiae1)
    N2 = len(minutiae2)
    
    def get_angle(m): return m['angle'] if isinstance(m, dict) else m[2]

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
                total_dev = 0.0
                
                for k, k_p in matches.items():
                    psi_ik = signed_angle_diff(get_angle(minutiae1[k]), theta_i)
                    psi_jk = signed_angle_diff(get_angle(minutiae2[k_p]), theta_j)
                    ang_dev = abs(signed_angle_diff(psi_ik, psi_jk))
                    
                    if ang_dev > delta_theta:
                        valid = False; break
                    
                    rc_ik = rc1[i, k]
                    rc_jk = rc2[j, k_p]
                    rc_dev = abs(rc_ik - rc_jk)
                    
                    if rc_dev > delta_rc:
                        valid = False; break
                        
                    total_dev += (rc_dev + ang_dev)
                
                if valid and total_dev < best_dist:
                    best_dist = total_dev
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
    active_indices = [pair_to_idx[pair] for pair in final_pairs if pair in pair_to_idx]
    
    if not active_indices:
        return 0.0
    
    energy = np.sum(W[np.ix_(active_indices, active_indices)])

    N_match = len(final_pairs)
    if N_match == 0 or (n_t + n_q) == 0:
        return 0.0
    
    # Paper Eq 26: 2 * Energy / (N * (nt + nq))
    score = (2 * energy) / (N_match * (n_t + n_q))
    return score

# Evaluation Helpers

def get_match_label(filename_a, filename_b):
    import os
    name_a = os.path.splitext(os.path.basename(filename_a))[0]
    name_b = os.path.splitext(os.path.basename(filename_b))[0]
    parts_a = name_a.split('_')
    parts_b = name_b.split('_')
    if len(parts_a) < 3 or len(parts_b) < 3: return "Imposter"
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
    for threshold in np.linspace(-1, 1, 2000):
        FAR, FRR = calculate_error_rates(threshold, genuine_scores, imposter_scores)
        diff = abs(FAR - FRR)
        if diff < min_diff:
            min_diff = diff
            best_threshold = threshold
            eer = (FAR + FRR) / 2.0
    return eer, best_threshold

def get_roc_data(genuine_scores, imposter_scores, num_points=100):
    thresholds = np.linspace(-1, 1, num_points)
    far_list, tar_list = [], []
    for t in thresholds:
        far, frr = calculate_error_rates(t, genuine_scores, imposter_scores)
        far_list.append(far)
        tar_list.append(100 - frr)
    return far_list, tar_list