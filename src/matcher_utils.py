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
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    gxx = cv2.GaussianBlur(gx ** 2, (0, 0), sigma)
    gyy = cv2.GaussianBlur(gy ** 2, (0, 0), sigma)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    
    theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy) + (np.pi / 2)
    return theta

def get_local_descriptor(minutia, orientation_map, radii=[20, 40, 60], K=8):
    """Compute rotation-invariant local descriptor."""
    x_m, y_m, theta_m = minutia
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
    
    for i in range(N):
        p1 = minutiae[i][:2] # (x, y)
        for j in range(i + 1, N):
            p2 = minutiae[j][:2]
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

def compute_edge_similarity(rc_ij, rc_i_j_prime, lam=0.5): # Tunable lambda
    """Compute edge (ridge-count) similarity."""
    diff = abs(rc_ij - rc_i_j_prime)
    return float(np.exp(-lam * diff))

def compute_compatibility_matrix(minutiae1, desc1, rc1,
                                 minutiae2, desc2, rc2):
    """Build the compatibility matrix W."""
    N1 = len(minutiae1)
    N2 = len(minutiae2)
    assignments = [(i, i_p) for i in range(N1) for i_p in range(N2)]
    M = len(assignments)
    W = np.zeros((M, M), dtype=np.float32)

    # Fill W
    for u, (i, i_p) in enumerate(assignments):
        # Diagonal: Node Similarity
        W[u, u] = compute_minutia_similarity(desc1[i], desc2[i_p])
        
        for v in range(u + 1, M):
            j, j_p = assignments[v]
            
            # Conflict check (One-to-One constraint)
            if i == j or i_p == j_p:
                compat = 0.0
            else:
                # Edge Compatibility
                rc_ij = rc1[i, j]
                rc_i_jp = rc2[i_p, j_p]
                compat = compute_edge_similarity(rc_ij, rc_i_jp)
            
            W[u, v] = compat
            W[v, u] = compat # Symmetry
            
    return W, assignments

# =========================================================
# 3. Refinement & Scoring
# =========================================================

def signed_angle_diff(a, b):
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi

def expand_minutia_pairs(minutiae1, minutiae2, rc1, rc2, ga_result, 
                         delta_theta=np.pi/6, delta_rc=1):
    """Refinement Step (Algorithm 3)."""
    N1 = len(minutiae1)
    N2 = len(minutiae2)
    
    # Initialization
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
    
    # Expansion
    while True:
        best_pair = None
        best_dist = np.inf
        
        for i in T2:
            theta_i = minutiae1[i][2]
            for j in Q2:
                theta_j = minutiae2[j][2]
                
                valid = True
                dist_ij = 0.0
                
                for k, k_p in matches.items():
                    # Orientation Check
                    psi_ik = signed_angle_diff(minutiae1[k][2], theta_i)
                    psi_jk = signed_angle_diff(minutiae2[k_p][2], theta_j)
                    
                    if abs(signed_angle_diff(psi_ik, psi_jk)) > delta_theta:
                        valid = False
                        break
                        
                    # Ridge Count Check
                    rc_ik = rc1[i, k]
                    rc_jk = rc2[j, k_p]
                    diff_rc = abs(rc_ik - rc_jk)
                    
                    if diff_rc > delta_rc:
                        valid = False
                        break
                        
                    dist_ij += diff_rc # L1 Norm
                
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
    """Calculate final verification score (Eq 26)."""
    pair_to_idx = {pair: u for u, pair in enumerate(assignments)}
    active_indices = [pair_to_idx[pair] for pair in final_pairs if pair in pair_to_idx]
    
    if not active_indices:
        return 0.0
        
    energy = np.sum(W[np.ix_(active_indices, active_indices)])
    N_match = len(final_pairs)
    
    if N_match == 0 or (n_t + n_q) == 0:
        return 0.0
        
    return (2 * energy) / (N_match * (n_t + n_q))


# =========================================================
# 4. Evaluation & Labeling Utils
# =========================================================

def get_match_label(filename_a, filename_b):
    """
    Returns 'Genuine' if files are from the same finger of the same subject,
    else 'Imposter'.
    Assumes format: SIRE-SubjectID_FingerID_CaptureID (e.g., SIRE-1_1_1.bmp)
    """
    import os
    # 1. Remove extension
    name_a = os.path.splitext(os.path.basename(filename_a))[0]
    name_b = os.path.splitext(os.path.basename(filename_b))[0]

    # 2. Split components
    parts_a = name_a.split('_')
    parts_b = name_b.split('_')

    # Safety check
    if len(parts_a) < 3 or len(parts_b) < 3:
        return "Imposter"

    # 3. Extract Subject ID and Finger ID
    # parts_a[0] might look like "SIRE-1" or just "1" depending on your files
    subject_a = parts_a[0]
    finger_a = parts_a[1]

    subject_b = parts_b[0]
    finger_b = parts_b[1]

    # 4. Decide label
    if subject_a == subject_b and finger_a == finger_b:
        return "Genuine"
    else:
        return "Imposter"

def calculate_error_rates(threshold, genuine_scores, imposter_scores):
    """Calculate FAR and FRR for a given threshold."""
    if len(genuine_scores) == 0 or len(imposter_scores) == 0:
        return 0.0, 0.0

    # False Acceptances (Impostors incorrectly accepted)
    false_accepts = sum(score >= threshold for score in imposter_scores)
    FAR = (false_accepts / len(imposter_scores)) * 100

    # False Rejections (Genuine users incorrectly rejected)
    false_rejects = sum(score < threshold for score in genuine_scores)
    FRR = (false_rejects / len(genuine_scores)) * 100

    return FAR, FRR

def find_eer(genuine_scores, imposter_scores):
    """Find the Equal Error Rate (EER)."""
    best_threshold = None
    min_diff = float('inf')
    eer = None

    # Sweep thresholds in [0, 1]
    # Since our scores are normalized 0-1 (roughly), this range is fine.
    # If scores are low (e.g. 0.05), we need fine granularity.
    for threshold in np.linspace(0, 1, 1000):
        FAR, FRR = calculate_error_rates(threshold, genuine_scores, imposter_scores)
        diff = abs(FAR - FRR)
        if diff < min_diff:
            min_diff = diff
            best_threshold = threshold
            eer = (FAR + FRR) / 2.0

    return eer, best_threshold

def get_roc_data(genuine_scores, imposter_scores, num_points=100):
    """Generates (FAR, TAR) pairs for plotting the ROC curve."""
    thresholds = np.linspace(0, 1, num_points)
    far_list = []
    tar_list = []

    for t in thresholds:
        far, frr = calculate_error_rates(t, genuine_scores, imposter_scores)
        tar = 100 - frr
        far_list.append(far)
        tar_list.append(tar)

    return far_list, tar_list