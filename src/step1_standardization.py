import argparse
from PIL import Image
def standardize_image(input_path, output_path, target_ppi=500):
    try:
        with Image.open(input_path) as img:
            current_ppi = img.info.get('dpi', (1000, 1000))[0]

            if current_ppi < target_ppi:
                print(f"Warning: Img ppi ({current_ppi}) < target ({target_ppi}). skipping resize.")
                scale_factor = 1
            else:
                scale_factor = target_ppi/current_ppi
            
            #calculate new dims
            new_w = int(img.width*scale_factor)
            new_h = int(img.height*scale_factor)
            print(f"Original Size: {img.size} @ {current_ppi} PPI")
            print(f"Target Size:   ({new_w}, {new_h}) @ {target_ppi} PPI")

            resized_image = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized_image.save(output_path, dpi=(target_ppi, target_ppi))
            print(f"Saved standardized image to: {output_path}")
    except Exception as e:
            print(f"Error processing image: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 1: Standardization (Down-sizing)')
    parser.add_argument('--input', type=str, required=True, help='Path to raw input image')
    parser.add_argument('--output', type=str, required=True, help='Path to save standardized image')
    args = parser.parse_args()

    standardize_image(args.input, args.output)