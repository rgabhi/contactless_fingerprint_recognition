import numpy as np
import math

# Re-use angle diff and descriptors from matcher
from matcher import get_angle_difference

def refinement_step(X_skel, minutiae_T, minutiae_Q, rc_matrix_T, rc_matrix_Q, 
                    delta_rc=2, delta_theta=np.pi/6):
    """
    Algorithm 3: Minutia-Pair Expanding Algorithm.
    X_skel: Vector from GA (index i contains match j+1).
    """
    Nt = len(minutiae_T)
    Nq = len(minutiae_Q)
    
    # Initialization [cite: 120]
    T1, Q1 = [], [] # Paired sets
    T2, Q2 = [], [] # Unpaired sets
    
    # Populate initial sets based on LGA skeleton
    for i, val in enumerate(X_skel):
        if val != 0:
            T1.append(i)
            Q1.append(val - 1) # Convert back to 0-index
        else:
            T2.append(i)
            
    # Add remaining query minutiae to Q2
    matched_Q = set(Q1)
    for j in range(Nq):
        if j not in matched_Q:
            Q2.append(j)
            
    # Expansion Loop [cite: 121]
    while len(T2) > 0 and len(Q2) > 0:
        candidates = []
        
        # (a) For every possible unpaired candidate pair (i, j)
        for i in T2:
            for j in Q2:
                # (b) Distance Calculation Check against all established matches (k, k')
                d_ij = 0
                valid_candidate = True
                
                if len(T1) == 0:
                    valid_candidate = False # No anchors to compare against
                
                for idx in range(len(T1)):
                    k = T1[idx]
                    k_prime = Q1[idx]
                    
                    # Check constraints [cite: 126]
                    # Orientation difference check
                    psi_ik = get_angle_difference(minutiae_T[i].angle, minutiae_T[k].angle)
                    psi_jk = get_angle_difference(minutiae_Q[j].angle, minutiae_Q[k_prime].angle)
                    
                    if get_angle_difference(psi_ik, psi_jk) > delta_theta:
                        valid_candidate = False
                        break
                        
                    # Ridge count check
                    rc_T = rc_matrix_T[i, k]
                    rc_Q = rc_matrix_Q[j, k_prime]
                    
                    if abs(rc_T - rc_Q) > delta_rc:
                        valid_candidate = False
                        break
                        
                    # Accumulate distance [cite: 127]
                    d_ij += abs(rc_T - rc_Q)
                
                if valid_candidate:
                    candidates.append({'i': i, 'j': j, 'dist': d_ij})
        
        # (c) Selection [cite: 133]
        if not candidates:
            break # Termination
            
        # Find min distance
        best_cand = min(candidates, key=lambda x: x['dist'])
        
        # Add to matched, remove from unmatched
        T1.append(best_cand['i'])
        Q1.append(best_cand['j'])
        T2.remove(best_cand['i'])
        Q2.remove(best_cand['j'])
        
    return T1, Q1