import os
import glob
from PIL import Image
from pynsdk.licensing import NLicense, NLicenseManager # for trial
from pynsdk.media import NImage
from pynsdk.biometrics import NBiometricEngine, NBiometricOperations, NFinger, NSubject, NBiometricStatus, NFPosition
import cv2
import numpy as np
import json
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


class FingerprintPreprocessor:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        # Initialize the SDK
        self.engine = self.init_sdk()
        self.all_features = {}

    def standardize_image(self, image_path, filename, target_ppi=500):
        try:
            with Image.open(image_path) as img:
                # Get current PPI, default to 1000 if not found
                current_ppi = img.info.get('dpi', (1000, 1000))[0]

                if current_ppi < target_ppi:
                    print(f"Warning: {filename} PPI ({current_ppi}) < target ({target_ppi}). Copying original.")
                    scale_factor = 1
                else:
                    scale_factor = target_ppi / current_ppi
                
                # Calculate new dims
                new_w = int(img.width * scale_factor)
                new_h = int(img.height * scale_factor)
                
                # Resize
                resized_image = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # Save to the processed folder
                save_path = os.path.join(self.output_path, filename)
                
                # Setting dpi in save is good metadata hygiene
                resized_image.save(save_path, dpi=(target_ppi, target_ppi))
                
                print(f"Processed: {filename} | {img.size} -> {resized_image.size}")
                return save_path
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return None

    def process_dataset(self):
        try:
            # Iterate through all subdirectories in the dataset path
            # We assume folders are named '1', '2', etc.
            subject_folders = [f for f in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, f))]
            
            count = 0
            for subject in subject_folders:
                subject_path = os.path.join(self.dataset_path, subject)
                
                # Find all BMP files
                bmp_files = glob.glob(os.path.join(subject_path, "*.bmp"))
                
                for file_path in bmp_files:
                    filename = os.path.basename(file_path)
                    
                    # Filter out the pre-processed files (HT, R414)
                    # We only want the pattern SIRE-SubjectID_FingerID_Capture.bmp
                    if "_HT" in filename or "_R414" in filename:
                        continue
                    
                    # Step 1: Standardize
                    resized_path = self.standardize_image(file_path, filename)
                    
                    if resized_path:
                        # Step 2: Segment
                        segmented_path = self.segment_image(resized_path, filename)
                        
                        if segmented_path:
                            # Step 3: Enhance
                            # We pass the segmented image path and the original filename
                            enhanced_img = self.enhance_image(segmented_path, filename)
                            
                            if enhanced_img is not None:
                                enhanced_filename = filename.replace(".bmp", "_enhanced.png")
                                enhanced_full_path = os.path.join(self.output_path, enhanced_filename)
                                cv2.imwrite(enhanced_full_path, enhanced_img)
                                print(f"Enhanced: {enhanced_filename}")

                                binary_img = self.binarize_image(enhanced_img)
                                binarized_filename = filename.replace(".bmp", "_binarized.png")
                                binarized_full_path = os.path.join(self.output_path, binarized_filename)                                
                                cv2.imwrite(binarized_full_path, binary_img)
                                print(f"Binarized: {binarized_filename}")

                                # After thinning
                                skeleton_img = self.thin_image(binary_img)
                                skeleton_filename = filename.replace(".bmp", "_skeleton.png")
                                skeleton_full_path = os.path.join(self.output_path, skeleton_filename)                                
                                cv2.imwrite(skeleton_full_path, skeleton_img)
                                print(f"Skeleton: {skeleton_filename}")

                                
                                # Clean skeleton
                                skeleton_img = self.remove_spurs(skeleton_img, min_length=12)

                                # Apply ROI
                                roi = self.get_roi_mask(enhanced_img)
                                skeleton_img = cv2.bitwise_and(skeleton_img, roi)
                                cv2.imwrite(skeleton_full_path, skeleton_img)


                                # --- CORRECTION ---
                                # Feed the ENHANCED GRAYSCALE image to the SDK, not the skeleton.
                                raw_minutiae = self.extract_minutiae(enhanced_full_path, enhanced_filename)

                                # --- POST-PROCESSING ---
                                # Apply your cleaning filters to the SDK's output
                                # Note: You'll need to pass the image shape for border removal
                                h, w = enhanced_img.shape
                                
                                clean_data = self.remove_border_minutiae(raw_minutiae, (h, w))
                                clean_data = self.prune_close_minutiae(clean_data, min_dist=12)
                                clean_data = self.remove_isolated(clean_data)
                                
                                self.all_features[filename] = clean_data
                                print(f"Minutiae extracted: {len(raw_minutiae)} -> {len(clean_data)} (cleaned)")
        
                    count += 1
            print(" -- writing minutiae -- ")
            minutiae_full_path = os.path.join(self.output_path, 'minutiae.json')
            with open(minutiae_full_path, 'w') as f:
                json.dump(self.all_features, f, indent=4)
            print(" -- writing minutiae done!-- ")        
            print(f"\nTotal images processed: {count}")
        except Exception as e:
            print(f"Error [process_dataset] {e}")
        finally:
            del self.engine
    
    def init_sdk(self):
        """Initializes the SDK and returns the Engine object."""
        # 1. Initialize License
        is_trial_mode = True
        NLicenseManager.set_trial_mode(is_trial_mode)
        
        print("Initializing VeriFinger SDK License...")
        license_name = "FingerClient"
        if not NLicense.obtain("/local", 5000, license_name):
            print(f"Failed to obtain license: {license_name}")
            # Fallback to FingerExtractor if Client not available (often bundle)
            if not NLicense.obtain("/local", 5000, "FingerExtractor"):
                sys.exit(1)

        # 2. Setup the Biometric Engine
        engine = NBiometricEngine()
        # We do NOT set fingers_return_binarized_image = True because we want grayscale for Step 3
        engine.fingers_return_binarized_image = False
        return engine

    def segment_image(self, image_path, filename):
            # 1. Load the image
            nimage = NImage(image_path)
            
            # 2. Prepare the Finger object
            finger = NFinger()
            finger.image = nimage
            finger.position = NFPosition.nfpUnknown
            
            # 3. Prepare the Subject object
            subject = NSubject()
            subject.fingers.add(finger)

            # 4. Perform the Segmentation
            # This modifies the 'subject' object in place!
            status = self.engine.perform_operation(subject, NBiometricOperations.segment)
            
            # 5. Check Result and Save
            if status == NBiometricStatus.ok:
                # CRITICAL FIX: The segmented result is usually at index 1 (or higher)
                # Index 0 is the original input image.
                if subject.fingers.count > 1:
                    # We grab the last one added, which is typically the segmented result
                    segmented_image = subject.fingers[subject.fingers.count - 1].image
                    
                    new_filename = os.path.splitext(filename)[0] + "_seg.png"
                    output_full_path = os.path.join(self.output_path, new_filename)
                    
                    segmented_image.save_to_file(output_full_path)
                    print(f"Segmented: {new_filename}")
                    return output_full_path
                else:
                    print(f"Warning: Segmentation status OK, but no new segment found for {filename}")
                    return None
            else:
                print(f"Segmentation failed for {filename}: {status}")
                return None


    def enhance_image(self, image_path, filename):
        img = cv2.imread(image_path, 0)
        if img is None:
            return None

        I = img.astype(np.float32) / 255.0

        # Illumination
        sigma = 20
        L = cv2.GaussianBlur(I, (0, 0), sigma)

        epsilon = 1e-6
        R = I / (L + epsilon)

        # Guided filter
        radius = 16
        eps = (0.01)**2
        refined_R = cv2.ximgproc.guidedFilter(I, R, radius, eps)

        refined_R = np.clip(refined_R, 0, 1)
        enhanced = (refined_R * 255).astype(np.uint8)

        # Contrast stretch
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

        # CLAHE (recommended for contactless)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)

        return enhanced

    def binarize_image(self, enhanced_img):
        binary = cv2.adaptiveThreshold(
            enhanced_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=3
        )
        return binary
    
    def thin_image(self, binary_img):
        skeleton = cv2.ximgproc.thinning(
            binary_img,
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
        return skeleton

    def remove_spurs(self, skel, min_length=10):
        """
        Remove short spurs from a skeleton image.
        skel: uint8 skeleton (255 = ridge)
        """
        skel = skel.copy()
        h, w = skel.shape

        def neighbors(y, x):
            return [(y+i, x+j) for i in [-1,0,1] for j in [-1,0,1]
                    if not (i == 0 and j == 0)
                    and 0 <= y+i < h and 0 <= x+j < w]

        changed = True
        while changed:
            changed = False
            endpoints = []

            for y in range(h):
                for x in range(w):
                    if skel[y, x] == 255:
                        n = sum(skel[ny, nx] == 255 for ny, nx in neighbors(y, x))
                        if n == 1:  # endpoint
                            endpoints.append((y, x))

            for y, x in endpoints:
                length = 0
                cy, cx = y, x
                py, px = -1, -1

                while True:
                    nbrs = [(ny, nx) for ny, nx in neighbors(cy, cx)
                            if skel[ny, nx] == 255 and (ny, nx) != (py, px)]
                    if len(nbrs) != 1:
                        break
                    py, px = cy, cx
                    cy, cx = nbrs[0]
                    length += 1
                    if length > min_length:
                        break

                if length <= min_length:
                    skel[y, x] = 0
                    changed = True

        return skel

    def get_roi_mask(self, enhanced_img):
        _, mask = cv2.threshold(
            enhanced_img, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def prune_close_minutiae(self, minutiae, min_dist=12):
        pruned = []
        for m in minutiae:
            if all((m['x']-n['x'])**2 + (m['y']-n['y'])**2 > min_dist**2 for n in pruned):
                pruned.append(m)
        return pruned

    def remove_border_minutiae(self, minutiae, shape, margin=15):
        h, w = shape
        return [
            m for m in minutiae
            if margin < m['x'] < w-margin and margin < m['y'] < h-margin
        ]

    def remove_isolated(self, minutiae, min_neighbors=1, radius=20):
        filtered = []
        for m in minutiae:
            neighbors = sum(
                (m['x']-n['x'])**2 + (m['y']-n['y'])**2 < radius**2
                for n in minutiae if m != n
            )
            if neighbors >= min_neighbors:
                filtered.append(m)
        return filtered


    def extract_minutiae(self, image_path, filename):
        # 1. Load the enhanced image
        nimage = NImage(image_path)

        # --- ROBUST FIX: Force Resolution ---
        # Set both horizontal and vertical resolution explicitly
        nimage.horz_resolution = 500
        nimage.vert_resolution = 500
        nimage.resolution = 500 # Set this too just in case
        
        # DEBUG: Verify the SDK accepted the value
        # This will tell us if the value is sticking or being ignored
        print(f"DEBUG: Resolution for {filename} set to: {nimage.horz_resolution}")
        # ------------------------------------
        
        finger = NFinger()
        finger.image = nimage
        finger.position = NFPosition.nfpUnknown
        
        subject = NSubject()
        subject.fingers.add(finger)

        # 2. Extract Features (Create Template)
        # This operation detects minutiae and generates the template
        status = self.engine.perform_operation(subject, NBiometricOperations.create_template)
        
        if status == NBiometricStatus.ok:
            # The template is stored in the subject
            # Hierarchy: Subject -> Template -> FingerRecords -> Minutiae
            # FIX: Use the property '.template' instead of the method '.get_template()'
            template = subject.template
            
            if template:
                # Based on finger_get_minutia.py, the path to minutiae is nested:
                # Template -> Fingers (Collection) -> Records (NFRecord) -> Minutiae
                # Note: The SDK wrapper structure can be tricky, so we use a try-block or inspect it.
                
                # Let's try the structure from your tutorial file:
                # minutia = finger_templates[0].fingers.records[0].minutiae
                
                try:
                    # Access the first finger record in the template
                    minutiae_list = template.fingers.records[0].minutiae
                except AttributeError:
                    # Fallback: sometimes it is accessed differently depending on the wrapper version
                    print(f"DEBUG: Template structure: {dir(template)}")
                    # Try direct access if the above fails (sometimes template.fingers[0].minutiae)
                    minutiae_list = template.fingers[0].minutiae
                
                # Prepare data for Graph Construction (Step 5)
                extracted_data = []
                
                # Prepare visualization image
                # We need to convert NImage to a numpy format OpenCV can handle
                # (The tutorial uses a helper, but we can reload via cv2 for simplicity)
                vis_image = cv2.imread(image_path)
                
                # Conversion factor: SDK byte (0-255) -> Degrees -> Radians
                # 360 / 255 ≈ 1.411764706
                byte_to_rad = (360.0 / 255.0) * (np.pi / 180.0)

                print(f"Minutiae found for {filename}: {len(minutiae_list)}")
                
                for m in minutiae_list:
                    # Geometric attributes
                    x = m.x
                    y = m.y
                    angle_rad = m.angle * byte_to_rad
                    
                    # Type: 0=Unknown, 1=Ending, 2=Bifurcation (SDK dependent)
                    m_type = m.type 
                    
                    # Store for the Graph
                    extracted_data.append({
                        'x': x,
                        'y': y,
                        'angle': angle_rad,
                        'type': m_type
                    })

                    # --- Visualization (Optional but Recommended) ---
                    # Draw location
                    cv2.circle(vis_image, (x, y), 3, (0, 0, 255), -1)
                    
                    # Draw direction
                    line_len = 15
                    x2 = int(x + line_len * np.cos(angle_rad))
                    y2 = int(y + line_len * np.sin(angle_rad))
                    cv2.line(vis_image, (x, y), (x2, y2), (255, 0, 0), 1)
                
                # Save Visualization
                vis_filename = filename.replace("_enhanced.png", "_minutiae.png")
                cv2.imwrite(os.path.join(self.output_path, vis_filename), vis_image)
                
                return extracted_data
            else:
                print(f"No template created for {filename}")
                return []
        else:
            print(f"Extraction failed for {filename}: {status}")
            return []
    
    
    def normalize_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi


    def build_graph(self, minutiae_list):
        """
        Build a Delaunay graph and store invariant edge features.

        Each edge stores:
            (distance, relative_angle_at_i, relative_angle_at_j)

        Returns:
            points: Nx2 array of (x, y)
            edges: list of (i, j)
            edge_features: dict {(i, j): (d, alpha_i, alpha_j)}
        """

        # 1. Extract minutiae coordinates
        points = np.array(
            [[m['x'], m['y']] for m in minutiae_list],
            dtype=np.float32
        )

        if len(points) < 3:
            return points, [], {}

        # 2. Delaunay triangulation
        tri = Delaunay(points)

        # 3. Extract unique edges
        edges = set()
        for simplex in tri.simplices:
            i, j, k = simplex
            edges.add(tuple(sorted((i, j))))
            edges.add(tuple(sorted((j, k))))
            edges.add(tuple(sorted((k, i))))

        edges = list(edges)

        # 4. Compute invariant edge features
        edge_features = {}

        for i, j in edges:
            xi, yi = points[i]
            xj, yj = points[j]

            dx = xj - xi
            dy = yj - yi

            # Edge length (translation + rotation invariant)
            distance = np.sqrt(dx * dx + dy * dy)

            if distance > 80:   # pixels, tune 60–100
                continue


            # Edge direction
            phi = np.arctan2(dy, dx)

            # Minutiae orientations
            theta_i = minutiae_list[i]['angle']
            theta_j = minutiae_list[j]['angle']

            # Relative angles (rotation invariant)
            alpha_i = self.normalize_angle(theta_i - phi)
            alpha_j = self.normalize_angle(theta_j - (phi + np.pi))

            edge_features[(i, j)] = (distance, alpha_i, alpha_j, phi)

        return points, edges, edge_features


    def match_graphs(self,probe_edge_features,gallery_edge_features,dist_tol=6,angle_tol=0.25,rot_tol=0.15):
        """
        Graph matching with rotation consensus.
        """

        probe_edges = list(probe_edge_features.values())
        gallery_edges = list(gallery_edge_features.values())

        rotation_diffs = []
        used_gallery = set()

        # -----------------------------
        # Step 1: Collect candidates
        # -----------------------------
        for p_idx, (dp, a1p, a2p, phi_p) in enumerate(probe_edges):
            for g_idx, (dg, a1g, a2g, phi_g) in enumerate(gallery_edges):

                if g_idx in used_gallery:
                    continue

                # Distance check
                if abs(dp - dg) > dist_tol:
                    continue

                # Angle checks (order invariant)
                direct_match = (
                    abs(a1p - a1g) < angle_tol and
                    abs(a2p - a2g) < angle_tol
                )

                swapped_match = (
                    abs(a1p - a2g) < angle_tol and
                    abs(a2p - a1g) < angle_tol
                )

                if direct_match:
                    diff = self.normalize_angle(phi_p - phi_g)
                    rotation_diffs.append(diff)
                    used_gallery.add(g_idx)
                    break

                elif swapped_match:
                    diff = self.normalize_angle(phi_p - (phi_g + np.pi))
                    rotation_diffs.append(diff)
                    used_gallery.add(g_idx)
                    break

        # -----------------------------
        # Step 2: Consensus voting
        # -----------------------------
        final_score = 0

        for r in rotation_diffs:
            count = sum(
                abs(self.normalize_angle(r - other)) < rot_tol
                for other in rotation_diffs
            )
            final_score = max(final_score, count)

        return final_score



    def get_match_label(self, filename_a, filename_b):
        """
        Returns 'Genuine' if files are from the same finger of the same subject,
        else 'Imposter'.
        """

        # 1. Remove extension
        name_a = os.path.splitext(filename_a)[0]
        name_b = os.path.splitext(filename_b)[0]

        # 2. Split components
        # Expected: SIRE-SubjectID_FingerID_CaptureID
        parts_a = name_a.split('_')
        parts_b = name_b.split('_')

        # Safety check
        if len(parts_a) < 3 or len(parts_b) < 3:
            return "Imposter"

        # 3. Extract Subject ID and Finger ID
        # parts_a[0] = "SIRE-1"
        subject_a = parts_a[0]      # includes "SIRE-"
        finger_a = parts_a[1]

        subject_b = parts_b[0]
        finger_b = parts_b[1]

        # 4. Decide label
        if subject_a == subject_b and finger_a == finger_b:
            return "Genuine"
        else:
            return "Imposter"

    def run_experiment(self, tolerance=6):
        """
        Run an all-vs-all fingerprint matching experiment.
        Returns genuine and imposter score lists.
        """

        # -------------------------------
        # Part 1: Precompute graphs
        # -------------------------------
        graph_database = {}

        for filename, minutiae_list in self.all_features.items():
            _, _, edge_lengths = self.build_graph(minutiae_list)
            graph_database[filename] = edge_lengths

        # -------------------------------
        # Part 2: All-vs-All Matching
        # -------------------------------
        genuine_scores = []
        imposter_scores = []

        filenames = list(graph_database.keys())
        N = len(filenames)

        for i in range(N):
            for j in range(i + 1, N):

                file_a = filenames[i]
                file_b = filenames[j]

                # Retrieve precomputed graphs
                edges_a = graph_database[file_a]
                edges_b = graph_database[file_b]

                # Compute match score
                raw_matches = score = self.match_graphs(edges_a,edges_b,dist_tol=6,angle_tol=0.17, rot_tol=0.1)
                score = raw_matches / min(len(edges_a), len(edges_b))

                # Determine label
                label = self.get_match_label(file_a, file_b)

                # Store score
                if label == "Genuine":
                    genuine_scores.append(score)
                else:
                    imposter_scores.append(score)

        return genuine_scores, imposter_scores


    def calculate_error_rates(self, threshold, genuine_scores, imposter_scores):
        """
        Calculate FAR and FRR for a given threshold.

        Args:
            threshold: decision threshold
            genuine_scores: list of genuine match scores
            imposter_scores: list of imposter match scores

        Returns:
            FAR (%), FRR (%)
        """

        # Safety checks
        if len(genuine_scores) == 0 or len(imposter_scores) == 0:
            return 0.0, 0.0

        # False Acceptances (Impostors incorrectly accepted)
        false_accepts = sum(score >= threshold for score in imposter_scores)
        FAR = (false_accepts / len(imposter_scores)) * 100

        # False Rejections (Genuine users incorrectly rejected)
        false_rejects = sum(score < threshold for score in genuine_scores)
        FRR = (false_rejects / len(genuine_scores)) * 100

        return FAR, FRR


    def find_eer(self, genuine_scores, imposter_scores):
        best_threshold = None
        min_diff = float('inf')
        eer = None

        # Sweep thresholds in [0, 1]
        for threshold in np.linspace(0, 1, 200):

            FAR, FRR = self.calculate_error_rates(
                threshold,
                genuine_scores,
                imposter_scores
            )

            diff = abs(FAR - FRR)

            if diff < min_diff:
                min_diff = diff
                best_threshold = threshold
                eer = (FAR + FRR) / 2.0

        return eer, best_threshold

    def visualize_match(self, filename_a, filename_b):
        """
        Visualizes the consensus matches between two fingerprints.
        """
        # 1. Load images
        img_a = cv2.imread(os.path.join(self.output_path, filename_a.replace(".bmp", "_enhanced.png")))
        img_b = cv2.imread(os.path.join(self.output_path, filename_b.replace(".bmp", "_enhanced.png")))
        
        # 2. Get Features
        edges_a = self.build_graph(self.all_features[filename_a])[2]
        edges_b = self.build_graph(self.all_features[filename_b])[2]
        
        # 3. Find Consensus Matches (Re-run logic to get specific edges)
        # We need to adapt the matching logic slightly to return the PAIRS
        # (This uses your hard-coded tolerances)
        dist_tol = 6
        angle_tol = 0.25
        rot_tol = 0.15
        
        probe_edges = list(edges_a.values())
        gallery_edges = list(edges_b.values())
        
        # Store (probe_idx, gallery_idx, rotation_diff)
        candidates = []
        
        # --- Copy-Paste of your Match Logic (Step 1) ---
        for p_idx, (dp, a1p, a2p, phi_p) in enumerate(probe_edges):
            for g_idx, (dg, a1g, a2g, phi_g) in enumerate(gallery_edges):
                
                if abs(dp - dg) > dist_tol: continue

                direct_match = (abs(a1p - a1g) < angle_tol and abs(a2p - a2g) < angle_tol)
                swapped_match = (abs(a1p - a2g) < angle_tol and abs(a2p - a1g) < angle_tol)
                
                if direct_match:
                    diff = self.normalize_angle(phi_p - phi_g)
                    candidates.append((p_idx, g_idx, diff))
                elif swapped_match:
                    diff = self.normalize_angle(phi_p - (phi_g + np.pi))
                    candidates.append((p_idx, g_idx, diff))

        # --- Consensus Logic (Step 2) ---
        best_count = 0
        best_rotation = 0
        
        # Find the best rotation cluster
        for _, _, r in candidates:
            count = sum(abs(self.normalize_angle(r - other[2])) < rot_tol for other in candidates)
            if count > best_count:
                best_count = count
                best_rotation = r
                
        # Filter: Keep only edges that agree with best_rotation
        final_matches = []
        for p_idx, g_idx, r in candidates:
            if abs(self.normalize_angle(r - best_rotation)) < rot_tol:
                # We need the original edge indices (i, j) to get coordinates
                edge_keys_a = list(edges_a.keys())
                edge_keys_b = list(edges_b.keys())
                final_matches.append((edge_keys_a[p_idx], edge_keys_b[g_idx]))

        print(f"Visualizing {len(final_matches)} consensus matches (Rotation: {np.degrees(best_rotation):.1f}°)")

        # 4. Draw!
        # Stack images side-by-side
        h1, w1 = img_a.shape[:2]
        h2, w2 = img_b.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = img_a
        vis[:h2, w1:w1+w2] = img_b
        
        points_a = self.build_graph(self.all_features[filename_a])[0]
        points_b = self.build_graph(self.all_features[filename_b])[0]

        for (ia, ja), (ib, jb) in final_matches:
            # Get coordinates for the edge midpoints or endpoints
            # Let's just draw the first point of the edge to keep it simple
            pt_a = tuple(points_a[ia].astype(int))
            pt_b = tuple(points_b[ib].astype(int))
            
            # Shift point B by width of image A
            pt_b_shifted = (pt_b[0] + w1, pt_b[1])
            
            # Draw line
            cv2.line(vis, pt_a, pt_b_shifted, (0, 255, 0), 1)
            cv2.circle(vis, pt_a, 4, (0, 0, 255), -1)
            cv2.circle(vis, pt_b_shifted, 4, (0, 0, 255), -1)

        cv2.imwrite(os.path.join(self.output_path, "match_vis.png"), vis)
        print("Saved match_vis.png")

    def get_roc_data(self, genuine_scores, imposter_scores, num_points=100):
        """
        Generates (FAR, TAR) pairs for plotting the ROC curve.
        """
        # Generate thresholds from 0.0 to 1.0
        thresholds = np.linspace(0, 1, num_points)
        
        far_list = []
        tar_list = []

        for t in thresholds:
            # Use your existing helper to get errors
            far, frr = self.calculate_error_rates(t, genuine_scores, imposter_scores)
            
            # TAR is the opposite of FRR (Accepted = 100% - Rejected)
            tar = 100 - frr
            
            far_list.append(far)
            tar_list.append(tar)

        return far_list, tar_list

    def tune_parameters(self, log_filename="tuning_results.csv"):
        """
        Runs a Grid Search to find the best (dist, rot, angle) combination.
        Logs results to a CSV file.
        """
        print(f"\n--- Starting 3-Parameter Tuning (Grid Search) ---")
        
        # 1. Precompute Graphs
        print("Precomputing all graphs...")
        graph_database = {}
        for filename, minutiae_list in self.all_features.items():
            _, _, edge_features = self.build_graph(minutiae_list)
            graph_database[filename] = edge_features

        filenames = list(graph_database.keys())
        N = len(filenames)
        
        # 2. Define Parameter Ranges
        dist_range = [4, 6, 8]
        rot_range = [0.10, 0.15, 0.20, 0.25]
        # We include 0.17 specifically to see if we can beat your record!
        angle_range = [0.10, 0.15, 0.17, 0.20, 0.25] 

        best_eer = 100.0
        best_params = (None, None, None)

        # 3. Open Log File
        with open(os.path.join(self.output_path, log_filename), "w") as f:
            # Update Header
            f.write("Dist_Tol,Rot_Tol,Angle_Tol,EER,Threshold\n")
            
            total_runs = len(dist_range) * len(rot_range) * len(angle_range)
            run_count = 0
            
            for d_tol in dist_range:
                for r_tol in rot_range:
                    for a_tol in angle_range:  # <--- New Loop!
                        run_count += 1
                        
                        genuine_scores = []
                        imposter_scores = []

                        # Run Matching Loop
                        for i in range(N):
                            for j in range(i + 1, N):
                                file_a = filenames[i]
                                file_b = filenames[j]
                                
                                edges_a = graph_database[file_a]
                                edges_b = graph_database[file_b]

                                # Call match_graphs with ALL parameters dynamic
                                raw_score = self.match_graphs(
                                    edges_a, edges_b, 
                                    dist_tol=d_tol, 
                                    angle_tol=a_tol,  # <--- Now using the loop variable
                                    rot_tol=r_tol
                                )
                                
                                denom = min(len(edges_a), len(edges_b))
                                score = raw_score / denom if denom > 0 else 0

                                label = self.get_match_label(file_a, file_b)
                                if label == "Genuine":
                                    genuine_scores.append(score)
                                else:
                                    imposter_scores.append(score)

                        # Calculate EER
                        eer, thresh = self.find_eer(genuine_scores, imposter_scores)
                        
                        # Log to file
                        f.write(f"{d_tol},{r_tol},{a_tol},{eer:.4f},{thresh:.4f}\n")
                        f.flush()
                        
                        # Only print if we found a new best (to keep output clean)
                        if eer < best_eer:
                            print(f"[{run_count}/{total_runs}] New Best! Dist={d_tol}, Rot={r_tol:.2f}, Angle={a_tol:.2f} -> EER: {eer:.2f}%")
                            best_eer = eer
                            best_params = (d_tol, r_tol, a_tol)

        print(f"\n🏆 Final Best EER: {best_eer:.2f}%")
        print(f"   Parameters: dist={best_params[0]}, rot={best_params[1]}, angle={best_params[2]}")
        print(f"   Log saved to: {os.path.join(self.output_path, log_filename)}")
        
        return best_params

# --- Execution Block ---
# Update these paths to match your actual structure

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(BASE_DIR)
dataset_dir = os.path.join(BASE_DIR, "data", "DS1_sample")
processed_dir = os.path.join(BASE_DIR, "data", "processed_images")


preprocessor = FingerprintPreprocessor(dataset_dir, processed_dir)

# 1. Process images to extract minutiae (Step 1-4)
preprocessor.process_dataset() 

# 2. Run the Matching Experiment (Step 5-7)
print("\n--- Running All-vs-All Matching Experiment ---")
gen_scores, imp_scores = preprocessor.run_experiment()

# 3. Analyze Results
avg_gen = sum(gen_scores) / len(gen_scores) if gen_scores else 0
avg_imp = sum(imp_scores) / len(imp_scores) if imp_scores else 0

print(f"Genuine Pairs: {len(gen_scores)} | Average Score: {avg_gen:.2f}")
print(f"Imposter Pairs: {len(imp_scores)} | Average Score: {avg_imp:.2f}")

# Optional: Save scores to file for plotting later
results = {"genuine": gen_scores, "imposter": imp_scores}
with open(os.path.join(processed_dir, "scores.json"), 'w') as f:
    json.dump(results, f)


eer, best_threshold = preprocessor.find_eer(gen_scores, imp_scores)
print(f"EER : {eer}\nThreshold: {best_threshold}")

preprocessor.visualize_match("SIRE-3_1_1.bmp", "SIRE-3_1_2.bmp")

print("\n--- Generating ROC Curve ---")

# 1. Get the data points
far_points, tar_points = preprocessor.get_roc_data(gen_scores, imp_scores)

# 2. Plot the Curve
plt.figure(figsize=(6, 6))
plt.plot(far_points, tar_points, label=f"ROC (EER={eer:.2f}%)", color='blue', linewidth=2)

# 3. Add the 'Random Guess' line (diagonal)
plt.plot([0, 100], [0, 100], linestyle='--', color='gray', label="Random Guess")

# 4. Labels and Saving
plt.xlabel("False Acceptance Rate (FAR) %")
plt.ylabel("True Acceptance Rate (TAR) %")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.grid(True)
plt.xlim([0, 100])
plt.ylim([0, 100])

save_path = os.path.join(processed_dir, "roc_curve.png")
plt.savefig(save_path)
print(f"ROC Curve saved to: {save_path}")

# Run the Grid Search
best_dist, best_rot, best_ang = preprocessor.tune_parameters("tuning_log.csv")

# Optional: You can now run the final ROC/Visualization using these best params!
# print(f"Running final validation with optimized parameters...")
# ...