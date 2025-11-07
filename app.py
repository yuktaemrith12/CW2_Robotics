# app.py
# Unified Flask app: Image Classification (upload/webcam snapshot) + YOLO Object Detection (live stream)

import os
import io
import base64
import threading
import time
from collections import deque

# --- Flask / HTTP ---
from flask import Flask, request, render_template, jsonify, Response

# --- Classification stack (PyTorch / torchvision) ---
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- Detection stack (OpenCV / Ultralytics YOLO) ---
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)


# =========================================================
# =============== 1) CLASSIFICATION SECTION ===============
# =========================================================


# --- Path to your YOLOv8 classification checkpoint ---
YOLO_CLS_WEIGHTS = 'office_item_classifier_yolov8cls.pt'

def load_classifier():
    """
    Load YOLOv8 classification model once at startup.
    """
    model = YOLO(YOLO_CLS_WEIGHTS)      # Ultralytics handles preprocessing internally
    model.fuse()                        # small speed boost at inference
    return model

clf_model = load_classifier()

# Class names come from the YOLO model itself
CLASSES = [name for _, name in sorted(clf_model.names.items(), key=lambda kv: kv[0])]

def preprocess_image(img_bytes: bytes):
    """
    Decode raw bytes to an RGB numpy array.
    YOLOv8 will handle resize/normalization internally.
    """
    pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return np.array(pil)   

def run_cls_inference(image_rgb: np.ndarray):
    """
    Run YOLOv8-CLS on an RGB image (numpy array).
    Returns the top-1 prediction + (optional) top-3 list when confidence < 95%.
    """
    results = clf_model(image_rgb, verbose=False)
    res = results[0]

    # Top-1
    top1_idx = int(res.probs.top1)
    top1_conf = float(res.probs.top1conf) * 100.0
    main_class = CLASSES[top1_idx]
    main_conf = top1_conf

    # Top-3  
    other_preds = None
    if main_conf < 95.0:
        # res.probs.data is a tensor of class probabilities
        probs = res.probs.data.cpu().numpy()
        top3_idx = probs.argsort()[-3:][::-1]  # indices of top-3 probs
        other_preds = []
        for idx in top3_idx:
            cls_name = CLASSES[int(idx)]
            conf_pct = probs[int(idx)] * 100.0
            other_preds.append({'class': cls_name, 'confidence': f"{conf_pct:.2f}"})
        # Ensure main class is listed first and not duplicated
        other_preds = [p for p in other_preds if p['class'] != main_class]

    return main_class, main_conf, other_preds

# ---- Classification routes ----

@app.route('/', methods=['GET'])
def home():
    """Landing page for classification UI """
    return render_template('index.html', prediction=None, confidence=None)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Case 1: multipart file upload -> HTML render or JSON (AJAX).
    Case 2: webcam/dataURL base64 -> JSON.
    """
    # ---- Case 1: file upload ----
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        raw_bytes = file.read()

        img_rgb = preprocess_image(raw_bytes)
        pred, conf, others = run_cls_inference(img_rgb)

        # Preview (base64 PNG) for UI
        try:
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception:
            data_url = None

        # AJAX -> JSON (no refresh)
        if request.form.get('ajax') == '1' or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'prediction': pred,
                'confidence': f"{conf:.2f}",
                'other_predictions': others or [],
                'uploaded_preview': data_url
            })

        # Fallback -> server-rendered page
        return render_template(
            'index.html',
            prediction=pred,
            confidence=f"{conf:.2f}",
            other_predictions=others,
            uploaded_preview=data_url
        )

    # ---- Case 2: base64 from webcam/snapshot ----
    b64_str = request.json.get('image_data') if request.is_json else request.form.get('image_data')
    if not b64_str:
        return jsonify({'error': 'No image provided'}), 400

    if ',' in b64_str:  # strip data URL header if present
        b64_str = b64_str.split(',', 1)[1]

    try:
        img_bytes = base64.b64decode(b64_str)
        img_rgb = preprocess_image(img_bytes)
        pred, conf, others = run_cls_inference(img_rgb)

        return jsonify({
            'prediction': pred,
            'confidence': f"{conf:.2f}",
            'other_predictions': others or []
        })
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {e}'}), 400


# ======================================================
# =============== 2) DETECTION SECTION ================
# ======================================================

# Load your custom-trained YOLO detection model

det_model = YOLO('best.pt')

# Thread-safe buffers using deque
raw_frames_buffer = deque(maxlen=1)
processed_frames_buffer = deque(maxlen=1)
stop_event = threading.Event()

camera = None
capture_thread = None
processing_thread = None

def create_placeholder_frame(text):
    """Creates a black frame with centered text (used before first detection frame)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

def capture_frames(cam):
    """Continuously capture frames and push to raw buffer."""
    while not stop_event.is_set():
        success, frame = cam.read()
        if success:
            raw_frames_buffer.append(frame)
        else:
            time.sleep(0.1)

def process_frames():
    """Pop raw frames, run YOLO, push annotated frames."""
    while not stop_event.is_set():
        try:
            raw = raw_frames_buffer.popleft()

            # Process with YOLO (YOLO handles resize for speed)
            results = det_model(raw, imgsz=640, verbose=False)
            
            # Use default confidence (0.25)
            annotated = results[0].plot()
            
            # ---  Test: Lower confidence to see weak detections ---
            # annotated = results[0].plot(conf=0.1) 

            processed_frames_buffer.append(annotated)
        except IndexError:
            # Buffer was empty, wait a moment
            time.sleep(0.01)

def generate_web_frames():
    """MJPEG generator for /video_feed."""
    processed_frames_buffer.append(create_placeholder_frame("Initializing..."))
    
    try:
        while not stop_event.is_set():
            try:
                # Peek at the last frame, don't pop (for smooth FPS)
                frame = processed_frames_buffer[0]

                ok, buf = cv2.imencode('.jpg', frame)
                if not ok:
                    continue
                frame_bytes = buf.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # Target ~30 FPS on the browser side
                time.sleep(0.03) 
                
            except IndexError:
                # Buffer was empty, wait a moment
                time.sleep(0.01)
    finally:
        # This 'finally' block runs when the client disconnects
        print("[YOLO] Client disconnected from video feed.")
        stop_detection_threads()


# ---- Detection routes ----
@app.route('/detect')
def detect_page():
    """Simple page that shows the live YOLO stream (create templates/detect.html)."""
    return render_template('detect.html')

@app.route('/video_feed')
def video_feed():
    """Live MJPEG stream for detection."""
    #  Start threads only when this page is requested
    start_detection_threads(camera_index=0) 
    return Response(generate_web_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ======================================================
# ================== 3) APP START/STOP =================
# ======================================================

def start_detection_threads(camera_index=0):
    global camera, capture_thread, processing_thread
    if capture_thread and capture_thread.is_alive():
        return  

    # Clear the stop event in case it was set by a previous run
    stop_event.clear()

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise IOError("Cannot open webcam.")

    capture_thread = threading.Thread(target=capture_frames, args=(camera,), daemon=True)
    processing_thread = threading.Thread(target=process_frames, daemon=True)
    capture_thread.start()
    processing_thread.start()
    print(f"[YOLO] Camera opened on index {camera_index}. Threads started.")

def stop_detection_threads():
    global camera, capture_thread, processing_thread
    stop_event.set()
    if capture_thread:
        capture_thread.join(timeout=1)
    if processing_thread:
        processing_thread.join(timeout=1)
    if camera:
        camera.release()
        camera = None # Set camera to None after release
        print("[YOLO] Camera released.")

@app.route('/healthz')
def health():
    """Optional: quick health endpoint."""
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    try:

        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except (KeyboardInterrupt, SystemExit):
        print("Shutdown signal received.")
    finally:
        print("Stopping threads and releasing resources...")
        stop_detection_threads()
        print("Shutdown complete.")