import numpy as np
import math
import cv2

def get_ridge_count(binarized_image, x1, y1, x2, y2):
    """
    Counts the number of ridges (0s) crossed between (x1, y1) and (x2, y2).
    A ridge is counted when we transition from valley (255/1) to ridge (0).
    """
    # 1. Get the length of the line to determine how many points to sample
    dist = int(math.hypot(x2 - x1, y2 - y1))
    if dist == 0:
        return 0

    # 2. Generate all (x, y) coordinates along the line
    # We use num=dist to sample roughly 1 pixel per step
    x_values = np.linspace(x1, x2, dist).astype(int)
    y_values = np.linspace(y1, y2, dist).astype(int)

    # 3. Get the pixel values at these coordinates
    # Note: OpenCV images are accessed as [y, x]
    try:
        pixel_values = binarized_image[y_values, x_values]
    except IndexError:
        # If coordinates go out of bounds (margin error), return 0
        return 0

    # 4. Count Transitions (1 -> 0)
    # Assuming the image is 0 for Ridge, 255 for Valley. 
    # Let's normalize to 0 and 1 first to be safe.
    # (Ridges = 0, Valleys = 1)
    normalized_pixels = (pixel_values > 128).astype(int) 

    transitions = 0
    # Loop through the pixels along the line
    for i in range(1, len(normalized_pixels)):
        current_pixel = normalized_pixels[i]
        prev_pixel = normalized_pixels[i-1]

        # Check if we just stepped onto a ridge
        # Previous was Valley (1), Current is Ridge (0)
        if prev_pixel == 1 and current_pixel == 0:
            transitions += 1
            
    return transitions


def compute_ridge_count_matrix(minutiae_list, binarized_image):
    num_minutiae = len(minutiae_list)
    # Initialize an N x N matrix with zeros
    matrix = np.zeros((num_minutiae, num_minutiae), dtype=int)
    
    for i in range(num_minutiae):
        for j in range(i + 1, num_minutiae): # Only compute upper triangle to save time
            m1 = minutiae_list[i]
            m2 = minutiae_list[j]
            
            # Calculate count
            count = get_ridge_count(binarized_image, m1.x, m1.y, m2.x, m2.y)
            
            # Fill both spots (it's symmetric!)
            matrix[i][j] = count
            matrix[j][i] = count
            
    return matrix


def compute_orientation_map(image):
    """
    Computes the local orientation angle for every pixel in the image.
    Returns a matrix of angles in radians.
    """
    # 1. Calculate Gradients using Sobel
    # cv2.CV_64F ensures we don't lose precision with negative numbers
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # 2. Calculate the angle for every pixel
    # np.arctan2 returns values in the range [-pi, pi]
    orientation_map = np.arctan2(gy, gx)
    
    return orientation_map


def get_angle_difference(t1, t2):
    """
    Calculates phi(t1, t2) according to Eq. 10 in the paper.
    """
    # Ensure angles are in [0, 2pi]
    t1 = t1 % (2 * np.pi)
    t2 = t2 % (2 * np.pi)
    
    if t1 > t2:
        return t1 - t2
    else:
        return (2 * np.pi) + t1 - t2

def compute_minutia_descriptor(minutia, orientation_map, L=5, K_l=8, r_step=5):
    """
    Constructs the descriptor d_i = {alpha_k,l} for a minutia.
    L: Number of concentric circles
    K_l: Sampling points per circle
    r_step: Distance between circles (in pixels)
    """
    descriptor = []
    h, w = orientation_map.shape
    
    x_c, y_c, theta_c = minutia.x, minutia.y, minutia.angle
    
    for l in range(1, L + 1):
        radius = l * r_step
        for k in range(K_l):
            # Angle for this sampling point (around the circle)
            # We add theta_c to make it rotation invariant (relative to minutia angle)
            angle_k = (2 * np.pi * k) / K_l + theta_c
            
            # Calculate sampling coordinates
            x_s = int(x_c + radius * np.cos(angle_k))
            y_s = int(y_c + radius * np.sin(angle_k))
            
            # Get orientation from map (check bounds)
            if 0 <= x_s < w and 0 <= y_s < h:
                alpha = orientation_map[y_s, x_s]
            else:
                alpha = 0 # Default if out of bounds
                
            descriptor.append(alpha)
            
    return np.array(descriptor)




def compute_minutia_similarity(desc1, desc2, mu=16):
    """
    Calculates the raw similarity gamma(i, i') using Eq. 2.
    """
    K = len(desc1)
    if K == 0:
        return 0
        
    similarity_sum = 0
    
    for k in range(K):
        # Calculate angle difference (delta)
        diff = get_angle_difference(desc1[k], desc2[k])
        
        # Apply the exponential formula
        # Eq 2: exp( -2 * mu / pi * diff )
        term = np.exp(- (2 * mu / np.pi) * diff)
        similarity_sum += term
        
    return similarity_sum / K


import numpy as np
import math
import cv2

# Keep your existing helper functions: 
# get_ridge_count, compute_ridge_count_matrix, compute_orientation_map, 
# get_angle_difference, compute_minutia_descriptor, compute_minutia_similarity

def compute_diagonal_similarity(minutiae_T, minutiae_Q, orientation_T, orientation_Q):
    """
    Computes Minutia-Wise Similarity with Normalization (Eq. 3 and 4).
    """
    Nt = len(minutiae_T)
    Nq = len(minutiae_Q)
    
    # 1. Raw Similarity (Eq. 2)
    raw_sim_matrix = np.zeros((Nt, Nq))
    
    # Pre-compute descriptors
    descriptors_T = [compute_minutia_descriptor(m, orientation_T) for m in minutiae_T]
    descriptors_Q = [compute_minutia_descriptor(m, orientation_Q) for m in minutiae_Q]
    
    for i in range(Nt):
        for j in range(Nq):
            raw_sim_matrix[i, j] = compute_minutia_similarity(descriptors_T[i], descriptors_Q[j])

    # 2. Subtract Mean (Eq. 3) [cite: 62]
    avg_score = np.mean(raw_sim_matrix)
    gamma_prime = raw_sim_matrix - avg_score
    
    # 3. Min-Max Normalization to [-1, 1] (Eq. 4) 
    # The PDF states gamma'_max = 1, gamma'_min = -1
    # We normalize the observed values to this range.
    min_val = np.min(gamma_prime)
    max_val = np.max(gamma_prime)
    
    if max_val - min_val == 0:
        return np.zeros((Nt, Nq)) # Avoid divide by zero
        
    # Scale to [-1, 1]
    S_aa = 2 * ((gamma_prime - min_val) / (max_val - min_val)) - 1
            
    return S_aa

def sign_func(x, delta):
    """Eq. 15: Returns 1 if x < delta, else -1"""
    return 1 if x < delta else -1

def compute_global_consistency_matrix(minutiae_T, minutiae_Q, 
                                      rc_matrix_T, rc_matrix_Q, 
                                      diagonal_sim_matrix,
                                      delta_rc=2, delta_theta=np.pi/6, delta_d=10):
    """
    Builds the Compatibility Matrix W using Eq. 6.
    """
    Nt = len(minutiae_T)
    Nq = len(minutiae_Q)
    num_candidates = Nt * Nq
    
    # Initialize Sparse or Dense Matrix (Dense for simplicity as per assignment)
    W = np.zeros((num_candidates, num_candidates))
    
    print(f"Building Global Matrix W ({num_candidates}x{num_candidates})...")

    for i in range(Nt):
        for i_prime in range(Nq):
            a = i * Nq + i_prime # Flattened index for candidate pair (i, i')
            
            # --- 1. Diagonal Terms (Minutia-Wise Similarity) [cite: 49] ---
            W[a, a] = diagonal_sim_matrix[i, i_prime]
            
            for j in range(Nt):
                for j_prime in range(Nq):
                    b = j * Nq + j_prime # Flattened index for candidate pair (j, j')
                    
                    if a == b: continue
                    
                    # --- 2. Conflict Constraint [cite: 69, 86] ---
                    # If i==j but i'!=j' (one template mapped to two query points)
                    # OR i!=j but i'==j' (two template points mapped to one query point)
                    if (i == j and i_prime != j_prime) or (i != j and i_prime == j_prime):
                        W[a, b] = 0
                        continue
                    
                    # --- 3. Pairwise Similarity Calculation [cite: 71-75] ---
                    
                    # A. Ridge Count Difference
                    rc_T = rc_matrix_T[i, j]
                    rc_Q = rc_matrix_Q[i_prime, j_prime]
                    diff_rc = abs(rc_T - rc_Q)
                    
                    # B. Orientation Difference
                    # psi_ij = phi(theta_i, theta_j)
                    psi_T = get_angle_difference(minutiae_T[i].angle, minutiae_T[j].angle)
                    psi_Q = get_angle_difference(minutiae_Q[i_prime].angle, minutiae_Q[j_prime].angle)
                    diff_theta = get_angle_difference(psi_T, psi_Q)
                    
                    # C. Euclidean Distance Difference
                    dist_T = math.hypot(minutiae_T[i].x - minutiae_T[j].x, minutiae_T[i].y - minutiae_T[j].y)
                    dist_Q = math.hypot(minutiae_Q[i_prime].x - minutiae_Q[j_prime].x, minutiae_Q[i_prime].y - minutiae_Q[j_prime].y)
                    diff_d = abs(dist_T - dist_Q)
                    
                    # D. Scoring Logic (Eq. 6)
                    # Product of sign functions
                    score = (sign_func(diff_rc, delta_rc) * sign_func(diff_theta, delta_theta) * sign_func(diff_d, delta_d))
                             
                    W[a, b] = score

    return W



def calculate_final_score(refined_T1, refined_Q1, energy_val, Nt, Nq):
    """
    Computes final score using Eq. 8:
    Score = (2 * f(x_final)) / (N_match * (Nt + Nq))
    """
    N_match = len(refined_T1)
    
    if N_match == 0 or (Nt + Nq) == 0:
        return 0.0
        
    score = (2 * energy_val) / (N_match * (Nt + Nq))
    return score