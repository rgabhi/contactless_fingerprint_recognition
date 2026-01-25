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


def compute_diagonal_similarity(minutiae_T, minutiae_Q, orientation_T, orientation_Q):
    """
    Computes the matrix of minutia-wise similarities (Eq. 2).
    Returns a matrix of size Nt x Nq.
    """
    Nt = len(minutiae_T)
    Nq = len(minutiae_Q)
    
    similarity_matrix = np.zeros((Nt, Nq))
    
    # Pre-compute descriptors for Template to save time
    descriptors_T = []
    for m in minutiae_T:
        descriptors_T.append(compute_minutia_descriptor(m, orientation_T))
        
    # Loop through Query minutiae
    for j in range(Nq):
        desc_Q = compute_minutia_descriptor(minutiae_Q[j], orientation_Q)
        
        # Compare against every Template minutia
        for i in range(Nt):
            sim = compute_minutia_similarity(descriptors_T[i], desc_Q)
            similarity_matrix[i, j] = sim
            
    return similarity_matrix