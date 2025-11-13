# ==========================================================
# ======= ADVANCED SYNTHETIC DATA GENERATION (YOLO) ========
# ==========================================================
# Purpose:
#   - Composites object cut-outs (RGBA PNGs) onto random office backgrounds
#   - Applies realistic augmentations (scale, rotation, lighting, noise, blur)
#   - Saves:
#       * JPEG image
#       * YOLO-format label (.txt) [class cx cy w h in relative coords]
#       * Visualization image with drawn boxes for quick QA
# Notes:
#   - Designed for object detection training (e.g., YOLOv8)
#   - Keeps background resolution; multiple objects per image
# ==========================================================

import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from tqdm import tqdm

# ----------------------------------------------------------
# 1) CONFIGURATION
# ----------------------------------------------------------

# --- Input Folders ---
# Folder of object PNGs with transparency (alpha channel)
OBJECT_DIR = 'Detection/synthetic_data_aug/seafood/seafood_cutout'
# Folder of room backgrounds (JPEG/PNG)
BACKGROUND_DIR = 'Detection/synthetic_data_aug/house_room_background/images'

# --- Output Folders (created if missing) ---
OUTPUT_IMAGE_DIR = 'Detection/synthetic_data_aug/seafood/seafood_synthetic/images'
OUTPUT_LABEL_DIR = 'Detection/synthetic_data_aug/seafood/seafood_synthetic/labels'
OUTPUT_VISUALIZE_DIR = 'Detection/synthetic_data_aug/seafood/seafood_synthetic/visualize'

# --- Generation Settings ---
NUM_IMAGES_TO_GENERATE = 1000          # Total synthetic images to produce
CLASS_ID = 4                           # Numeric class id for YOLO labels
CLASS_NAME = 'seafood'                # Human-readable name (for visualization only)
MIN_OBJECTS_PER_IMAGE = 1               # Range of objects pasted per background
MAX_OBJECTS_PER_IMAGE = 3

# --- Augmentation Strengths ---
MIN_SCALE = 0.2                         
MAX_SCALE = 0.8                          
MAX_ROTATION = 20                  
MIN_BRIGHTNESS = 0.5              
MAX_BRIGHTNESS = 1.3
MIN_CONTRAST = 0.7                     
MAX_CONTRAST = 1.2
NOISE_STRENGTH = 15                      

# ----------------------------------------------------------
# Utility: Add RGB noise without touching alpha channel
# ----------------------------------------------------------
def add_noise(pil_img, strength):
    """Adds uniform RGB noise to a PIL RGBA/RGB image while preserving alpha."""
    np_img = np.array(pil_img)

    # Random noise in [-strength, strength], shape (H, W, 3)
    noise = np.random.randint(-strength, strength + 1,
                              (np_img.shape[0], np_img.shape[1], 3))

    # Operate on RGB channels only (ignore alpha if present)
    rgb = np_img[:, :, :3]

    # Add noise in int16 to avoid uint8 overflow, then clip back to [0, 255]
    noisy_rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    np_img[:, :, :3] = noisy_rgb

    return Image.fromarray(np_img)

# ----------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------
def main():
    print("Starting ADVANCED synthetic data generation...")

    # 2) SETUP — ensure outputs exist
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VISUALIZE_DIR, exist_ok=True)

    # Resolve candidate files
    try:
        object_files = [f for f in os.listdir(OBJECT_DIR) if f.endswith('.png')]
        background_files = [f for f in os.listdir(BACKGROUND_DIR)
                            if f.endswith(('.jpg', '.jpeg', '.png'))]
    except FileNotFoundError as e:
        print(f" ERROR: Directory not found. Check your paths.\n{e}")
        return

    # Basic sanity checks
    if not object_files:
        print(f" ERROR: No .png object files found in {OBJECT_DIR}")
        return
    if not background_files:
        print(f" ERROR: No background images found in {BACKGROUND_DIR}")
        return

    print(f"Found {len(object_files)} objects and {len(background_files)} backgrounds.")

    # 3) MAIN GENERATION LOOP
    for i in tqdm(range(NUM_IMAGES_TO_GENERATE), desc="Generating Images"):
        # --- Pick and open a random background (RGBA for alpha-compositing) ---
        bg_path = os.path.join(BACKGROUND_DIR, random.choice(background_files))
        try:
            background = Image.open(bg_path).convert('RGBA')
            
            # ✅ Resize every background to 224x224 for consistent output
            TARGET_SIZE = (224, 224)
            background = background.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Warning: Could not open background {bg_path}. Skipping. Error: {e}")
            continue


        # Work on a copy so original background stays intact
        synthetic_img = background.copy()
        yolo_labels = []     # Accumulates YOLO lines per image
        boxes_to_draw = []   # For visualization overlay

        # Number of objects to paste on this background
        num_objects = random.randint(MIN_OBJECTS_PER_IMAGE, MAX_OBJECTS_PER_IMAGE)

        for _ in range(num_objects):
            # --- Pick and open a random object PNG (with transparency) ---
            obj_path = os.path.join(OBJECT_DIR, random.choice(object_files))
            try:
                obj = Image.open(obj_path).convert('RGBA')
            except Exception as e:
                print(f"Warning: Could not open object {obj_path}. Skipping. Error: {e}")
                continue

            # ----------------- AUGMENTATION PIPELINE -----------------
            # 1) Scale (preserve aspect ratio)
            scale = random.uniform(MIN_SCALE, MAX_SCALE)
            obj_w = int(obj.width * scale)
            obj_h = int(obj.height * scale)
            obj = obj.resize((obj_w, obj_h), Image.Resampling.LANCZOS)

            # 2) Rotation (with expansion to keep full bounds)
            angle = random.uniform(-MAX_ROTATION, MAX_ROTATION)
            obj = obj.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

            # 3) Lighting (Brightness then Contrast)
            enhancer = ImageEnhance.Brightness(obj)
            obj = enhancer.enhance(random.uniform(MIN_BRIGHTNESS, MAX_BRIGHTNESS))
            enhancer = ImageEnhance.Contrast(obj)
            obj = enhancer.enhance(random.uniform(MIN_CONTRAST, MAX_CONTRAST))

            # 4) Noise (RGB only, alpha preserved)
            obj = add_noise(obj, NOISE_STRENGTH)

            # 5) Optional soft blur to reduce cut-out edges
            if random.random() > 0.5:
                obj = obj.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.0)))

            # ----------------- PASTE & LABEL -----------------
            paste_w, paste_h = obj.size
            max_x = background.width - paste_w
            max_y = background.height - paste_h
            if max_x <= 0 or max_y <= 0:
                # Skip if object would not fit
                continue

            # Random top-left position that keeps object fully in frame
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            # Composite object onto background using its alpha as mask
            synthetic_img.paste(obj, (paste_x, paste_y), obj)

            # Compute bounding box in absolute pixels
            box_x1, box_y1 = paste_x, paste_y
            box_x2, box_y2 = paste_x + paste_w, paste_y + paste_h
            bg_w, bg_h = background.size

            # Convert to YOLO normalized [cx, cy, w, h] (relative to background size)
            yolo_x_center = ((box_x1 + box_x2) / 2) / bg_w
            yolo_y_center = ((box_y1 + box_y2) / 2) / bg_h
            yolo_w = (box_x2 - box_x1) / bg_w
            yolo_h = (box_y2 - box_y1) / bg_h

            # Append single line label string
            yolo_labels.append(
                f"{CLASS_ID} {yolo_x_center:.6f} {yolo_y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}\n"
            )

            # Store for visualization overlay
            boxes_to_draw.append([box_x1, box_y1, box_x2, box_y2])

        # If nothing was pasted, skip saving this frame
        if not yolo_labels:
            continue

        # 4) SAVE ALL OUTPUTS
        synthetic_img_rgb = synthetic_img.convert('RGB')  # Drop alpha for JPEG
        base_filename = f"synthetic_seafood_{i:05d}"

        # -- Save image (.jpg) --
        img_path = os.path.join(OUTPUT_IMAGE_DIR, base_filename + ".jpg")
        synthetic_img_rgb.save(img_path, "JPEG")

        # -- Save label (.txt, YOLO format) --
        label_path = os.path.join(OUTPUT_LABEL_DIR, base_filename + ".txt")
        with open(label_path, 'w') as f:
            f.writelines(yolo_labels)

        # -- Save visualization (drawn boxes + class text) --
        draw = ImageDraw.Draw(synthetic_img_rgb)
        for box in boxes_to_draw:
            draw.rectangle(box, outline="lime", width=3)
            draw.text((box[0], box[1] - 15), CLASS_NAME, fill="lime")

        viz_path = os.path.join(OUTPUT_VISUALIZE_DIR, base_filename + ".jpg")
        synthetic_img_rgb.save(viz_path, "JPEG")

    # Summary
    print("-" * 40)
    print(f"Generation complete!")
    print(f"Total images created: {NUM_IMAGES_TO_GENERATE}")
    print(f"Images: {OUTPUT_IMAGE_DIR}")
    print(f"Labels: {OUTPUT_LABEL_DIR}")
    print(f"Visualizations: {OUTPUT_VISUALIZE_DIR}")
    print("-" * 40)

# ----------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------
if __name__ == "__main__":
    main()
