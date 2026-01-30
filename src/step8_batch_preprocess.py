import os
import argparse
import sys
import re
import cv2
import time

# Add src to path to allow importing modules
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import your steps as modules
import step1_standardization as s1
import step2_roi_extraction as s2
import step3_enhancement as s3
import step4_minutia_extraction as s4

def parse_jpg_filename(filename):
    # Matches: 1_10_2_0.jpg
    match = re.match(r"(\d+)_(\d+)_(\d+)_(\d+)", filename)
    if match: return match.groups()
    return None

def main(args):
    input_root = os.path.abspath(args.input_dir)
    processed_dir = os.path.join(input_root, "processed_jpgs")
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. INITIALIZE SDK (ONCE)
    print("="*40)
    print("INITIALIZING BIO-ENGINE...")
    try:
        # We use Step 2's init function to get the engine
        engine = s2.init_sdk()
        print("SDK Initialized. Starting Batch Processing...")
    except Exception as e:
        print(f"Critical SDK Error: {e}")
        sys.exit(1)
    print("="*40)

    # 2. Collect Tasks
    tasks = []
    for root, dirs, files in os.walk(input_root):
        if os.path.basename(root) == 'raw':
            for f in files:
                if f.lower().endswith('.jpg'):
                    meta = parse_jpg_filename(f)
                    if meta:
                        subj, fing, cap, seq = meta
                        full_path = os.path.join(root, f)
                        clean_name = f"{subj}_{fing}_{cap}_{seq}"
                        tasks.append((full_path, clean_name))

    # Sort by Subject ID
    tasks.sort(key=lambda x: int(x[1].split('_')[0]))
    total = len(tasks)
    print(f"Found {total} images to process.")
    
    success_count = 0
    start_time = time.time()
    
    try:
        # 3. PROCESSING LOOP
        for i, (input_path, base_name) in enumerate(tasks):
            path_std = os.path.join(processed_dir, f"{base_name}_std.jpg")
            path_roi = os.path.join(processed_dir, f"{base_name}_roi.jpg")
            path_enh = os.path.join(processed_dir, f"{base_name}_enhanced.jpg")
            path_json = os.path.join(processed_dir, f"{base_name}_minutiae.json")
            
            # Skip if output exists
            if os.path.exists(path_json) and os.path.exists(path_enh):
                continue

            print(f"[{i+1}/{total}] {base_name}...", end=" ", flush=True)

            try:
                # Step 1: Standardize
                s1.standardize_image(input_path, path_std)
                
                # Step 2: ROI (Pass Engine)
                if not s2.process_roi(engine, path_std, path_roi):
                    print("[Fail Step 2]")
                    continue
                    
                # Step 3: Enhance
                if not os.path.exists(path_roi):
                    print("[Fail Step 3: No Input]")
                    continue
                s3.enhance_fingerprint(path_roi, path_enh)
                
                # Step 4: Minutiae (Pass Engine)
                if not s4.process_minutiae(engine, path_enh, path_json):
                     print("[Fail Step 4]")
                     continue
                
                print("Done.")
                success_count += 1
                
            except Exception as e:
                print(f"[Error: {e}]")
                continue

    finally:
        # 4. CLEANUP (Crucial to prevent 'invalid_operation' crash)
        print("\nReleasing SDK Engine...")
        del engine
        print(f"Batch Complete. Success: {success_count}/{total}")
        print(f"Time Taken: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help="Path to data root (containing 'raw' folders)")
    args = parser.parse_args()
    main(args)