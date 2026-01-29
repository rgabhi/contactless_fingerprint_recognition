import os
import argparse
import glob
import subprocess
import sys
import re

def run_command(cmd):
    """Runs a shell command and prints output if error occurs"""
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=False)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(e.output.decode())
        return False
    return True

def parse_raw_filename(filename):
    """
    Parses 'SI-1_g15_8_1_5.pgm'
    Format: SI-{Subject}_g{Group}_{Finger}_{Capture}_{Seq}.pgm
    Returns: (subject, finger, capture, seq)
    """
    # Regex to capture the digits
    # Matches: SI-1, g15, 8, 1, 5
    match = re.match(r"SI-(\d+)_g\d+_(\d+)_(\d+)_(\d+)", filename)
    
    if match:
        return match.groups() # (Subject, Finger, Capture, Seq)
    return None

def main(args):
    input_root = args.input_dir
    
    # Create centralized output directory
    processed_dir = os.path.join(input_root, "processed_raw")
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"Scanning for RAW .pgm images in {input_root}...")
    
    tasks = []
    
    # Recursive walk to find files in DS1/*/raw/
    for root, dirs, files in os.walk(input_root):
        # We only care about the 'raw' folder
        if os.path.basename(root) == 'raw':
            for f in files:
                if f.lower().endswith('.pgm'):
                    meta = parse_raw_filename(f)
                    if meta:
                        subj, fing, cap, seq = meta
                        full_path = os.path.join(root, f)
                        
                        # New Unique Name: Subject_Finger_Capture_Seq
                        clean_name = f"{subj}_{fing}_{cap}_{seq}"
                        tasks.append((full_path, clean_name))

    # Sort tasks numerically by subject for clean processing order
    tasks.sort(key=lambda x: int(x[1].split('_')[0]))
    
    print(f"Found {len(tasks)} raw samples.")
    
    for i, (input_path, base_name) in enumerate(tasks):
        # Define output paths
        path_std = os.path.join(processed_dir, f"{base_name}_std.jpg")
        path_roi = os.path.join(processed_dir, f"{base_name}_roi.png")
        path_enh = os.path.join(processed_dir, f"{base_name}_enhanced.jpg")
        path_json = os.path.join(processed_dir, f"{base_name}_minutiae.json")
        
        # Check if already done
        if os.path.exists(path_json) and os.path.exists(path_enh):
            if i % 10 == 0: 
                print(f"[{i+1}/{len(tasks)}] Skipping {base_name} (Already processed)")
            continue

        print(f"[{i+1}/{len(tasks)}] Processing {base_name}...")

        # 1. Standardization (Resize)
        if not run_command(["python3", "src/step1_standardization.py", "--input", input_path, "--output", path_std]): continue
        
        # 2. ROI Extraction
        if not run_command(["python3", "src/step2_roi_extraction.py", "--input", path_std, "--output", path_roi]): continue
        
        # 3. Enhancement (Intrinsic Decomposition + Guided Filter)
        if not run_command(["python3", "src/step3_enhancement.py", "--input", path_roi, "--output", path_enh]): continue
        
        # 4. Minutiae Extraction (VeriFinger)
        if not run_command(["python3", "src/step4_minutia_extraction.py", "--input", path_enh, "--output", path_json]): continue

    print("\nBatch preprocessing complete!")
    print(f"All processed files are stored in: {processed_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 8: Batch Preprocess RAW Dataset')
    parser.add_argument('--input_dir', required=True, help='Root path to DS1 folder')
    args = parser.parse_args()
    main(args)