import numpy as np
import cv2
import math

def compute_orientation_map(image, block_size=16):
    """
    Computes the local orientation (theta) for every pixel block.
    Returns a grid of angles in radians [0, pi].
    """
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)

    h, w = image.shape
    orientations = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            blk_gx = gx[y:y+block_size, x:x+block_size]
            blk_gy = gy[y:y+block_size, x:x+block_size]

            Gxx = np.sum(blk_gx**2)
            Gyy = np.sum(blk_gy**2)
            Gxy = np.sum(blk_gx*blk_gy)

            denom = Gxx - Gyy
            if denom == 0: denom = 0.000001
            
            theta = 0.5 * np.arctan2(2*Gxy, denom) + (np.pi/2) 
            orientations[y:y + block_size, x:x+block_size] = theta
    return orientations

def get_local_descriptor(x, y, orientation_map, L=5, K=8, R_step=10):
    """
    Constructs the descriptor for Eq (2).
    """
    descriptor = []
    h, w = orientation_map.shape

    for l in range(1, L + 1):
        radius = l * R_step
        for k in range(K):
            angle = (2 * np.pi * k) / K
            sx = int(x + radius * np.cos(angle))
            sy = int(y + radius * np.sin(angle))

            # FIX: Strict bounds check (< instead of <=) and index order [sy, sx]
            if 0 <= sx < w and 0 <= sy < h:
                descriptor.append(orientation_map[sy, sx])
            else:
                descriptor.append(0.0)
    return np.array(descriptor)

def get_ridge_count(image, p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    # Increase sampling density (1.5x distance) to catch thin ridges
    num_points = int(math.hypot(x2 - x1, y2 - y1) * 1.5)
    if num_points == 0: return 0
    
    x_values = np.linspace(x1, x2, num_points)
    y_values = np.linspace(y1, y2, num_points)

    intensities = []
    h, w = image.shape
    for i in range(num_points):
        ix, iy = int(x_values[i]), int(y_values[i])
        if 0 <= ix < w and 0 <= iy < h:
            intensities.append(image[iy, ix])

    if not intensities: return 0

    # Smooth the intensity profile to remove pixel noise
    profile = np.array(intensities)
    # Simple moving average of size 3
    if len(profile) > 3:
        profile = np.convolve(profile, np.ones(3)/3, mode='valid')

    # Adaptive Thresholding on the profile
    local_threshold = np.mean(profile)
    
    ridges = 0
    in_ridge = False
    
    # Count crossings
    for val in profile:
        # Check if currently on a ridge (assuming dark ridges on light background? 
        # CHECK: If your ridges are white (255), use val > local_threshold)
        # Based on Step 3 CLAHE, ridges are likely dark (low intensity).
        is_ridge_pixel = val < local_threshold 

        if is_ridge_pixel and not in_ridge:
            ridges += 1
            in_ridge = True
        elif not is_ridge_pixel:
            in_ridge = False
            
    return ridges