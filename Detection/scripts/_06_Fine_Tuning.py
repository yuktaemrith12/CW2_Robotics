from ultralytics import YOLO

def fine_tune_model():
    """
    Fine-tunes the previously trained specialist model.
    """
    print(" Starting fine-tuning run...")

    # --- 1. LOAD YOUR BEST MODEL from the last run ---
    # Make sure this path is correct!
    model = YOLO('training_runs/specialist_model_v2/weights/best.pt') 

    # --- 2. START THE FINE-TUNING PROCESS ---
    results = model.train(
        # --- Core Parameters ---
        data='v1_dataset.yaml',
        epochs=25,              # Shorter run: 20-30 epochs is plenty.
        patience=10,            # Stop if no improvement
        project='training_runs',
        name='specialist_model_v3', # Give it a new name

        # --- 3. CRITICAL: Lower the Learning Rate ---
        lr0=0.001,              # Use a much smaller learning rate

        # --- EFFICIENCY PARAMETERS (KEEP THESE) ---
        batch=16,               # Use the *maximized* batch size
        half=True,
        workers=4,
        cache=False,

        # --- 4. DATA AUGMENTATION (TURNED OFF) ---
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,             # Critically, mosaic must be 0
    )

    print(f"\n Fine-tuning complete!")
    print(f"Your new fine-tuned model is saved in the '{results.save_dir}' folder.")

if __name__ == '__main__':
    fine_tune_model()