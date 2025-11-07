import os

# --- Configuration ---
labels_dir = 'path to which your dumb ass labels are stored' 

# The single class ID you want for the final merged box
FINAL_CLASS_ID = 'class id you want to set for the final merged box' 
# ---------------------

def merge_annotations_to_single_box(directory, final_class_id):
    """
    Reads all .txt files in a directory.
    Merges ALL annotations (polygons or boxes) within a single file 
    into ONE single bounding box.
    Overwrites the file with the single bounding box line.
    """
    
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f" Error: Directory not found at {directory}")
        return

    merged_files_count = 0
    
    print(" Starting annotation merge process...")

    # Walk through the directory to find all .txt files
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            
            all_x_points = []
            all_y_points = []
            
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                if not lines: # Skip empty files
                    continue

                # --- 1. Collect all points from all shapes ---
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5: 
                        continue
                    
                    coords = parts[1:]

                    # Check if it's a polygon (x,y,x,y...)
                    if len(coords) > 4 and len(coords) % 2 == 0:
                        all_x_points.extend([float(coords[i]) for i in range(0, len(coords), 2)])
                        all_y_points.extend([float(coords[i]) for i in range(1, len(coords), 2)])
                    
                    # Check if it's already a box (x_c, y_c, w, h)
                    elif len(coords) == 4:
                        x_center, y_center, width, height = map(float, coords)
                        # Convert box to min/max coordinates
                        x1 = x_center - (width / 2)
                        y1 = y_center - (height / 2)
                        x2 = x_center + (width / 2)
                        y2 = y_center + (height / 2)
                        all_x_points.extend([x1, x2])
                        all_y_points.extend([y1, y2])
                
                if not all_x_points or not all_y_points:
                    print(f" Warning: No valid annotations found in {filename}. Skipping.")
                    continue

                # --- 2. Find the single bounding box ---
                min_x = min(all_x_points)
                max_x = max(all_x_points)
                min_y = min(all_y_points)
                max_y = max(all_y_points)
                
                # Convert the single large box to YOLO format
                final_x_center = (min_x + max_x) / 2
                final_y_center = (min_y + max_y) / 2
                final_width = max_x - min_x
                final_height = max_y - min_y
                
                # --- 3. Create the single line for the new file ---
                final_x_center = max(0.0, min(1.0, final_x_center))
                final_y_center = max(0.0, min(1.0, final_y_center))
                final_width = max(0.0, min(1.0, final_width))
                final_height = max(0.0, min(1.0, final_height))
                
                final_line = f"{final_class_id} {final_x_center} {final_y_center} {final_width} {final_height}\n"
                
                # --- 4. Write the single merged line back to the file ---
                with open(filepath, 'w') as f:
                    f.write(final_line)
                
                merged_files_count += 1

            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")

    print("-" * 50)
    print(f" Merging complete.")
    print(f"   - {merged_files_count} files were processed.")
    print(f"   - All annotations in each file have been merged into one box.")
    print("-" * 50)

# Run the function
merge_annotations_to_single_box(labels_dir, FINAL_CLASS_ID)