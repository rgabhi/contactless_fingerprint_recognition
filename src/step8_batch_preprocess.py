import os
import argparse
import sys
import re
import cv2
import time
import logging

# Add src to path to allow importing modules
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import your steps as modules
import step1_standardization as s1
import step2_roi_extraction as s2
import step3_enhancement as s3
import step4_minutia_extraction as s4

def setup_logging(log_file="batch_process.log"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_jpg_filename(filename):
    # Matches: 1_10_2_0.jpg
    match = re.match(r"(\d+)_(\d+)_(\d+)_(\d+)", filename)
    if match: return match.groups()
    return None

def main(args):
    setup_logging(args.log_file)
    input_root = os.path.abspath(args.input_dir)
    processed_dir = os.path.join(input_root, "processed_jpgs")
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. INITIALIZE SDK (ONCE)
    logging.info("="*40)
    logging.info("INITIALIZING BIO-ENGINE...")
    try:
        # We use Step 2's init function to get the engine
        engine = s2.init_sdk()
        logging.info("SDK Initialized. Starting Batch Processing...")
    except Exception as e:
        logging.critical(f"Critical SDK Error: {e}")
        sys.exit(1)
    logging.info("="*40)

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
    logging.info(f"Found {total} images to process.")
    
    success_count = 0
    start_time = time.time()
    
    try:
        # 3. PROCESSING LOOP
        for i, (input_path, base_name) in enumerate(tasks):
            path_std = os.path.join(processed_dir, f"{base_name}_std.jpg")
            path_roi = os.path.join(processed_dir, f"{base_name}_roi.jpg")
            path_enh = os.path.join(processed_dir, f"{base_name}_enhanced.jpg")
            path_json = os.path.join(processed_dir, f"{base_name}_minutiae.json")
            
            # Use --force to overwrite if needed, otherwise skip
            if not args.force and os.path.exists(path_json) and os.path.exists(path_enh):
                if i % 50 == 0: # Reduce spam for skipped files
                    logging.info(f"[{i+1}/{total}] {base_name}: Skipping (Already Exists)")
                continue

            try:
                # Step 1: Standardize
                s1.standardize_image(input_path, path_std)
                
                # Step 2: ROI (Pass Engine)
                if not s2.process_roi(engine, path_std, path_roi):
                    logging.warning(f"[{i+1}/{total}] {base_name}: Failed Step 2 (ROI)")
                    continue
                    
                # Step 3: Enhance
                if not os.path.exists(path_roi):
                    logging.warning(f"[{i+1}/{total}] {base_name}: Failed Step 3 (No Input)")
                    continue
                s3.enhance_fingerprint(path_roi, path_enh)
                
                # Step 4: Minutiae (Pass Engine)
                if not s4.process_minutiae(engine, path_enh, path_json):
                     logging.warning(f"[{i+1}/{total}] {base_name}: Failed Step 4 (Minutiae)")
                     continue
                
                success_count += 1
                # Log success periodically to keep shell clean
                if i % 10 == 0 or i == total - 1:
                    logging.info(f"[{i+1}/{total}] {base_name}: Success")
                
            except Exception as e:
                logging.error(f"[{i+1}/{total}] {base_name}: Error {e}")
                continue

    finally:
        # 4. CLEANUP (Crucial to prevent 'invalid_operation' crash)
        logging.info("Releasing SDK Engine...")
        del engine
        elapsed_min = (time.time() - start_time) / 60
        logging.info(f"Batch Complete. Success: {success_count}/{total}")
        logging.info(f"Time Taken: {elapsed_min:.2f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help="Path to data root (containing 'raw' folders)")
    parser.add_argument('--log_file', default='batch_process.log', help="Log file path")
    parser.add_argument('--force', action='store_true', help="Force overwrite of existing files")
    args = parser.parse_args()
    main(args)