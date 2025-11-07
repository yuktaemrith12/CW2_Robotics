import os
import cv2 # Make sure to 'pip install opencv-python'

# --- Configuration ---
base_dir = 'path to your base directory containing images and labels- where everything is strored (segmented images and labels)'

# Set the names of your subdirectories
images_dir_name = 'images'
labels_dir_name = 'labels'

# Set the name for the new folder to save visualizations
output_dir_name = 'visualized_output' 

# Since your script remapped everything to '0', you only need one item.
CLASS_NAMES = ['Table'] 

# --- Drawing Configuration ---
BOX_COLOR = (0, 255, 0)  # Green (in BGR format for OpenCV)
TEXT_COLOR = (255, 255, 255) # White
BOX_THICKNESS = 1
FONT_SCALE = 0.5

# --- Image extensions to check ---
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
# ---------------------

def visualize_bounding_boxes(base_path, img_dir, lbl_dir, out_dir, class_names):
    """
    Reads images and their corresponding YOLO labels, draws the bounding boxes
    and class names, and saves them to a new output directory.
    """
    
    # --- 1. Define Full Paths ---
    images_path = os.path.join(base_path, img_dir)
    labels_path = os.path.join(base_path, lbl_dir)
    output_path = os.path.join(base_path, out_dir)

    # --- 2. Validate Paths ---
    if not os.path.isdir(images_path):
        print(f" Error: Images directory not found at {images_path}")
        return
    if not os.path.isdir(labels_path):
        print(f" Error: Labels directory not found at {labels_path}")
        return

    # --- 3. Create Output Directory ---
    os.makedirs(output_path, exist_ok=True)
    print(f" Saving visualized images to: {output_path}")

    processed_count = 0

    # --- 4. Loop Through All Images ---
    for image_filename in os.listdir(images_path):
        if not image_filename.lower().endswith(IMAGE_EXTENSIONS):
            continue  # Skip files that aren't images

        # --- 5. Define file paths ---
        image_filepath = os.path.join(images_path, image_filename)
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_filepath = os.path.join(labels_path, label_filename)
        output_filepath = os.path.join(output_path, image_filename)

        try:
            # --- 6. Load the image ---
            image = cv2.imread(image_filepath)
            if image is None:
                print(f"Warning: Could not read image {image_filename}. Skipping.")
                continue
                
            img_height, img_width, _ = image.shape

            # --- 7. Check for label file and draw boxes ---
            if os.path.exists(label_filepath):
                with open(label_filepath, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Warning: Skipping malformed line in {label_filename}: {line}")
                        continue

                    # Parse YOLO data
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])

                    # De-normalize coordinates to pixel values
                    box_width = width * img_width
                    box_height = height * img_height
                    x_center_pixels = x_center * img_width
                    y_center_pixels = y_center * img_height

                    # Calculate top-left (x1, y1) and bottom-right (x2, y2)
                    x1 = int(x_center_pixels - (box_width / 2))
                    y1 = int(y_center_pixels - (box_height / 2))
                    x2 = int(x_center_pixels + (box_width / 2))
                    y2 = int(y_center_pixels + (box_height / 2))

                    # --- 8. Draw the rectangle on the image ---
                    cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

                    # --- 9. Draw the class label text ---
                    try:
                        label_text = class_names[class_id]
                    except IndexError:
                        label_text = f"Class {class_id}" # Fallback
                    
                    # Get text size to create a filled background
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, BOX_THICKNESS)
                    
                    # Draw a filled rectangle for the text background
                    cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), BOX_COLOR, -1)
                    
                    # Put the white text on top of the filled rectangle
                    cv2.putText(image, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, BOX_THICKNESS, cv2.LINE_AA)

            # --- 10. Save the modified image ---
            cv2.imwrite(output_filepath, image)
            processed_count += 1

        except Exception as e:
            print(f"An error occurred while processing {image_filename}: {e}")

    # --- Final Summary ---
    print("-" * 50)
    print(f" Visualization complete.")
    print(f"   - {processed_count} images were processed and saved to:")
    print(f"   - {output_path}")
    print("-" * 50)


# --- Run the script ---
visualize_bounding_boxes(
    base_dir, 
    images_dir_name, 
    labels_dir_name, 
    output_dir_name,
    CLASS_NAMES
)