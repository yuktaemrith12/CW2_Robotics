# ==========================================================
# ============ SPLIT IMAGE DATASET =========================
# ==========================================================
# This script divides the preprocessed dataset into
# Training, Validation, and Test sets in a 70/15/15 ratio.
# It preserves class-specific subfolder structures to ensure
# balanced sampling for each category.
# ==========================================================


# ---------- Imports ----------
import os               
import shutil          
import random           
from tqdm import tqdm    

# ---------- Define Directory Paths and Split Ratios ----------
SOURCE_DIR = 'Classification/dataset/Processed_Dataset'   
OUTPUT_DIR = 'Classification/dataset/Final_Dataset'       
SPLIT_RATIO = [0.70, 0.15, 0.15]                          # Train, Validation, Test ratios


# ==========================================================
# ================ SPLITTING FUNCTION ======================
# ==========================================================
def split_dataset():

    
    # --- Step 1: Validate the split ratio ---
    if sum(SPLIT_RATIO) != 1.0:
        print(f"Error: Split ratio must sum to 1.0, but it sums to {sum(SPLIT_RATIO)}")
        return
    
    # --- Step 2: Remove existing output folder if it already exists ---
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # --- Step 3: Create new clean directory structure ---
    print(f"Creating new directory structure at: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 4: Iterate through each class folder ---
    for class_name in tqdm(os.listdir(SOURCE_DIR), desc="Splitting classes"):
        class_path = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        # Get list of all image files for this class
        files = os.listdir(class_path)
        random.shuffle(files)  # Shuffle to ensure random distribution

        # Determine number of files per split
        train_end = int(SPLIT_RATIO[0] * len(files))
        val_end = train_end + int(SPLIT_RATIO[1] * len(files))

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # --- Step 5: Define subfolders for each split ---
        for split_name, split_files in zip(['train', 'validation', 'test'], [train_files, val_files, test_files]):
            split_class_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            # --- Step 6: Copy images into appropriate split folders ---
            for file in split_files:
                src_file = os.path.join(class_path, file)
                dst_file = os.path.join(split_class_dir, file)
                shutil.copy(src_file, dst_file)

    print("✅ Dataset successfully split into train, validation, and test sets.")


# ==========================================================
# ========================= MAIN ===========================
# ==========================================================
if __name__ == '__main__':
    split_dataset()  


# ==================== NEXT STEP ====================
# Zip Final_Dataset folder 
# Upload to Google Drive for training on Google Colab
# ==================================================