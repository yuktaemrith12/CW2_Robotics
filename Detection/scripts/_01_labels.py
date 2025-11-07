import os

# --- Configuration ---
# Set the path to the directory containing your images
images_dir = 'path to which your dumb ass images are stored' 

# Set the path to the directory containing your YOLO .txt label files
labels_dir = 'path to which your dumb ass labels are stored' 

# The new class ID you want for ALL objects (0 for a single class)
NEW_CLASS_ID = 'class id you want to set for all objects' 

# The image extensions to look for when deleting
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
# ---------------------

def process_yolo_labels(labels_directory, images_directory, new_id):
    """
    Remaps class IDs in non-empty YOLO label files.
    Deletes empty label files and their corresponding images.
    """
    
    # --- Directory Validation ---
    if not os.path.isdir(labels_directory):
        print(f" Error: Labels directory not found at {labels_directory}")
        return
    if not os.path.isdir(images_directory):
        print(f" Error: Images directory not found at {images_directory}")
        return

    # --- Counters for the final report ---
    remapped_count = 0
    deleted_txt_count = 0
    deleted_img_count = 0
    
    print(" Starting dataset processing...")

    # --- Iterate through all files in the labels directory ---
    for filename in os.listdir(labels_directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(labels_directory, filename)
            
            try:
                # --- 1. Check if the label file is empty ---
                if os.path.getsize(filepath) == 0:
                    print(f" Found empty label file: {filename}")
                    
                    # Delete the empty .txt file
                    os.remove(filepath)
                    deleted_txt_count += 1
                    print(f"   -> Deleted {filename}")

                    # --- 2. Find and delete the corresponding image ---
                    base_filename = os.path.splitext(filename)[0]
                    image_found = False
                    for ext in IMAGE_EXTENSIONS:
                        image_path = os.path.join(images_directory, base_filename + ext)
                        if os.path.exists(image_path):
                            os.remove(image_path)
                            deleted_img_count += 1
                            print(f"   -> Deleted corresponding image: {base_filename + ext}")
                            image_found = True
                            break # Stop after finding the first match
                    
                    if not image_found:
                         print(f"   -> No corresponding image found for {filename}")
                    
                    continue # Skip to the next file

                # --- 3. If not empty, remap the class ID ---
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5: 
                        coordinates = parts[1:] 
                        new_line = new_id + ' ' + ' '.join(coordinates) + '\n'
                        new_lines.append(new_line)
                    else:
                        new_lines.append(line)
                
                # Write the modified content back to the file
                with open(filepath, 'w') as f:
                    f.writelines(new_lines)
                
                remapped_count += 1

            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")

    # --- Final Summary ---
    print("-" * 50)
    print(" Dataset processing complete.")
    print(f" Summary:")
    print(f"   - Remapped {remapped_count} label files to Class ID: {NEW_CLASS_ID}.")
    print(f"   - Deleted {deleted_txt_count} empty label files.")
    print(f"   - Deleted {deleted_img_count} corresponding images.")
    print("-" * 50)


# --- Run the script ---
process_yolo_labels(labels_dir, images_dir, NEW_CLASS_ID)