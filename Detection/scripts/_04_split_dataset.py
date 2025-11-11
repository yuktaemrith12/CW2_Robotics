import os
import shutil
import random

# --- Configuration ---
# The source folder containing your 'images' and 'labels' subfolders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "..", "dataset", "_02_compiled_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "dataset", "_03_split_dataset")
SPLIT_RATIO = (0.7, 0.15, 0.15)


def split_data():
    """
    Splits a YOLO dataset into training, validation, and test sets.
    """
    print(f" Starting to split dataset from '{SOURCE_DIR}'...")

    source_images_dir = os.path.join(SOURCE_DIR, "images")
    source_labels_dir = os.path.join(SOURCE_DIR, "labels")

    if not os.path.exists(source_images_dir):
        print(f"Error: Source images directory not found at '{source_images_dir}'")
        return

    # 1. Create the destination folder structure
    sets = ['train', 'valid', 'test']
    for s in sets:
        os.makedirs(os.path.join(OUTPUT_DIR, s, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, s, 'labels'), exist_ok=True)

    # 2. Get a list of all image files and shuffle them
    image_files = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files)

    # 3. Calculate split points
    total_images = len(image_files)
    train_end = int(total_images * SPLIT_RATIO[0])
    valid_end = train_end + int(total_images * SPLIT_RATIO[1])

    # 4. Assign images to each set
    train_files = image_files[:train_end]
    valid_files = image_files[train_end:valid_end]
    test_files = image_files[valid_end:]

    file_sets = {
        'train': train_files,
        'valid': valid_files,
        'test': test_files
    }

    # 5. Move the files
    for set_name, files in file_sets.items():
        print(f"Moving {len(files)} files to '{set_name}' set...")
        for file_name in files:
            base_name = os.path.splitext(file_name)[0]
            label_name = base_name + ".txt"

            # Define source paths
            src_image_path = os.path.join(source_images_dir, file_name)
            src_label_path = os.path.join(source_labels_dir, label_name)

            # Define destination paths
            dest_image_path = os.path.join(OUTPUT_DIR, set_name, 'images', file_name)
            dest_label_path = os.path.join(OUTPUT_DIR, set_name, 'labels', label_name)

            # Move the image and its corresponding label
            if os.path.exists(src_image_path):
                shutil.move(src_image_path, dest_image_path)
            if os.path.exists(src_label_path):
                shutil.move(src_label_path, dest_label_path)

    print("\n Dataset splitting complete!")
    print(f"Results saved in '{OUTPUT_DIR}'.")
    print(f"  - Training set: {len(train_files)} images")
    print(f"  - Validation set: {len(valid_files)} images")
    print(f"  - Test set: {len(test_files)} images")

if __name__ == "__main__":
    split_data()