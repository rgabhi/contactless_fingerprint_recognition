import os
import argparse
import glob
import subprocess
import sys
import re

def run_command(cmd):
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=False)
    except subprocess.CalledProcessError as e:
        # Decode error for readability
        err_msg = e.output.decode('utf-8', errors='ignore').strip()
        print(f"  [FAILED] {err_msg.splitlines()[-1]}") # Print last line of error
        return False
    return True

def parse_jpg_filename(filename):
    # Matches: 1_10_2_0.jpg -> (1, 10, 2, 0)
    match = re.match(r"(\d+)_(\d+)_(\d+)_(\d+)", filename)
    if match:
        return match.groups()
    return None

def main(args):
    input_root = os.path.abspath(args.input_dir)
    processed_dir = os.path.join(input_root, "processed_jpgs")
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"Scanning {input_root} for images...")
    tasks = []
    
    # Recursive search for .jpg in raw folders
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

    tasks.sort(key=lambda x: int(x[1].split('_')[0]))
    print(f"Found {len(tasks)} raw samples.")
    
    success_count = 0
    
    for i, (input_path, base_name) in enumerate(tasks):
        # All outputs as .jpg
        path_std = os.path.join(processed_dir, f"{base_name}_std.jpg")
        path_roi = os.path.join(processed_dir, f"{base_name}_roi.jpg") # Changed to .jpg
        path_enh = os.path.join(processed_dir, f"{base_name}_enhanced.jpg")
        path_json = os.path.join(processed_dir, f"{base_name}_minutiae.json")
        
        if os.path.exists(path_json) and os.path.exists(path_enh):
            continue

        print(f"[{i+1}/{len(tasks)}] Processing {base_name}...", end=" ", flush=True)

        # Pipeline
        if not run_command(["python3", "src/step1_standardization.py", "--input", input_path, "--output", path_std]): 
            print("")
            continue
        if not run_command(["python3", "src/step2_roi_extraction.py", "--input", path_std, "--output", path_roi]): 
            print("")
            continue
        if not run_command(["python3", "src/step3_enhancement.py", "--input", path_roi, "--output", path_enh]): 
            print("")
            continue
        if not run_command(["python3", "src/step4_minutia_extraction.py", "--input", path_enh, "--output", path_json]): 
            print("")
            continue
            
        print("Done.")
        success_count += 1

    print("\n" + "="*30)
    print(f"Preprocessing Complete. Success: {success_count}/{len(tasks)}")
    print(f"Output Directory: {processed_dir}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    args = parser.parse_args()
    main(args)