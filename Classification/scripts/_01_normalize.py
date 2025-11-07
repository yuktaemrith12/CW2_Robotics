# ==========================================================
# ============ NORMALISE IMAGE DATASET SCRIPT ==============
# ==========================================================
# This script prepares image data for training deep learning
# models by resizing and converting images into a consistent
# format and quality. It creates a clean, uniform dataset
# under the 'Processed_Dataset' folder for classification tasks.
# ==========================================================

# ---------- Imports ----------
import os                        
from PIL import Image           
from tqdm import tqdm              

# ---------- Define Input, Output and Resize Parameters ----------
INPUT_DIR = 'Classification/dataset/Main_Dataset'       
OUTPUT_DIR = 'Classification/dataset/Processed_Dataset' 
SIZE = (224, 224)                                  

# ==========================================================
# =============== IMAGE NORMALISATION FUNCTION =============
# ==========================================================
def resize_images():

    # --- Step 1: Create output directory if it doesn't exist ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 2: Collect all image paths from subfolders ---
    images = []
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                images.append(os.path.join(root, file))

    # --- Step 3: Process each image with a progress bar ---
    for img_path in tqdm(images, desc="Processing images"):
        # Preserve relative subfolder structure
        rel_path = os.path.relpath(img_path, INPUT_DIR)
        out_path = os.path.join(OUTPUT_DIR, rel_path)

        # Ensure destination folder exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # --- Step 4: Resize and convert each image -----
        try:
            img = Image.open(img_path)
            img = img.resize(SIZE, Image.Resampling.LANCZOS)    # High-quality downsampling
            img = img.convert('RGB')                            # Ensure 3-channel image
            img.save(out_path, 'JPEG', quality=95)              # Save with good compression-quality balance
        except Exception as e:
            # Log errors if an image fails to process
            print(f"Failed: {img_path} - {e}")


# ==========================================================
# =============== MAIN EXECUTION ENTRY POINT ===============
# ==========================================================
if __name__ == '__main__':
    resize_images()   
