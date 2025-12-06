import cv2
import time
import threading
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO

# Try importing the Arm library (Handles case if testing off-robot)
try:
    from Arm_Lib import Arm_Device
    Arm = Arm_Device()
    time.sleep(0.1)
    print("✅ DOFBOT connected")
    HAS_ROBOT = True
except ImportError:
    print("⚠️ Arm_Lib not found. Running in simulation/mock mode.")
    HAS_ROBOT = False
    Arm = None

app = Flask(__name__)

# ==========================================
# CONFIGURATION & GLOBALS
# ==========================================
DETECTION_WEIGHTS = "best.pt"  # Ensure this file is in the same directory
try:
    model = YOLO(DETECTION_WEIGHTS)
    print(f"✅ YOLO detection model loaded: {DETECTION_WEIGHTS}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Global state variables
output_frame = None
lock = threading.Lock()
is_sorting_active = False
system_status = "Idle"
last_detected = "None"
streak_counter = 0

# ==========================================
# ROBOT COORDINATES & FUNCTIONS
# ==========================================
p_mould  = [90, 50, 50, 10, 90]      # Resting
p_top    = [90, 80, 50, 50, 270]     # High clearance
p_Brown  = [90, 30, 61, 30, 270]     # Pick area

# Drop zones
CLASS_TO_TARGET_POSE = {
    "bottle":    [65, 22, 64, 56, 270],   # Yellow
    "can":       [124, 22, 66, 45, 270],  # Red
    "detergent": [136, 66, 20, 29, 270],  # Purple
    "fruit":     [44, 66, 20, 28, 270],   # Blue
    "pulses":    [8, 72, 27, 6, 270],     # Pink
    "seafood":   [178, 79, 19, 0, 270],   # Navy
}

def arm_move(p, s_time=500):
    """Move robot arm."""
    if not HAS_ROBOT: return
    for i in range(5):
        id = i + 1
        if id == 5:
            time.sleep(.1)
            Arm.Arm_serial_servo_write(id, p[i], int(s_time * 1.2))
        else:
            Arm.Arm_serial_servo_write(id, p[i], s_time)
        time.sleep(.01)
    time.sleep(s_time / 1000.0)

def arm_clamp_block(enable):
    """0 = Open, 1 = Close"""
    if not HAS_ROBOT: return
    if enable == 0:
        Arm.Arm_serial_servo_write(6, 60, 400)
    else:
        Arm.Arm_serial_servo_write(6, 135, 400)
    time.sleep(0.5)

def arm_move_up():
    if not HAS_ROBOT: return
    Arm.Arm_serial_servo_write(2, 90, 1500)
    Arm.Arm_serial_servo_write(3, 90, 1500)
    Arm.Arm_serial_servo_write(4, 90, 1500)
    time.sleep(0.1)

def go_idle():
    """Return to resting pose."""
    global system_status
    system_status = "Returning to Home"
    arm_clamp_block(0)
    arm_move_up()
    arm_move(p_mould, 1000)
    system_status = "Idle - Waiting for object"

def perform_pick_and_place(target_pose, item_name):
    global system_status
    
    system_status = f"Picking up {item_name}..."
    # Go above pick
    arm_move(p_top, 1000)
    arm_move(p_Brown, 1000)
    arm_clamp_block(1) # Grab
    
    system_status = f"Moving {item_name} to bin..."
    # Move to bin
    arm_move(p_top, 1000)
    arm_move(target_pose, 1000)
    arm_clamp_block(0) # Release
    
    # Return home
    go_idle()

# ==========================================
# BACKGROUND THREADS
# ==========================================

def camera_loop():
    """Captures video frames constantly."""
    global output_frame, lock
    cap = cv2.VideoCapture(1) # Try index 0, change to 1 if needed
    
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue
            
        # Resize for faster processing/streaming
        frame = cv2.resize(frame, (640, 480))
        
        with lock:
            output_frame = frame.copy()
            
        time.sleep(0.03) # Limit FPS slightly

def sorting_logic_loop():
    """Runs YOLO and controls the arm."""
    global is_sorting_active, output_frame, lock, system_status, last_detected, streak_counter
    
    candidate = None
    streak = 0
    FRAMES_TO_CONFIRM = 5 # Reduced slightly for responsiveness
    
    # Ensure robot is home at start
    go_idle()
    
    while True:
        if not is_sorting_active:
            time.sleep(0.5)
            continue
            
        # Get latest frame safely
        with lock:
            if output_frame is None:
                continue
            frame_to_process = output_frame.copy()
            
        # Run YOLO
        if model:
            results = model(frame_to_process, conf=0.5, verbose=False)
            result = results[0]
            
            # Find best detection
            top_conf = 0.0
            top_name = None
            
            if result.boxes:
                for box in result.boxes:
                    conf = float(box.conf.item())
                    cls_id = int(box.cls.item())
                    name = model.names[cls_id]
                    if conf > top_conf:
                        top_conf = conf
                        top_name = name
            
            last_detected = top_name if top_name else "Scanning..."
            
            # Confirmation Logic
            if top_name:
                if top_name == candidate:
                    streak += 1
                else:
                    candidate = top_name
                    streak = 1
                
                if streak >= FRAMES_TO_CONFIRM:
                    # Confirmed!
                    target = CLASS_TO_TARGET_POSE.get(candidate)
                    if target:
                        print(f"Confirmed {candidate}")
                        # Pause detection while moving
                        perform_pick_and_place(target, candidate)
                        # Reset
                        streak = 0
                        candidate = None
                        time.sleep(1.0) # Wait a bit before next scan
                    else:
                        system_status = f"Unknown bin for {candidate}"
            else:
                streak = 0
                candidate = None
                
        time.sleep(0.1)

# ==========================================
# FLASK ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

def generate_feed():
    """Generates the MJPEG stream with bounding boxes drawn."""
    global output_frame, lock, model
    while True:
        with lock:
            if output_frame is None:
                continue
            # Draw current detection on the frame for the user to see
            frame = output_frame.copy()

        # Simple annotation (optional: use YOLO plot if preferred)
        if model and output_frame is not None:
            # We re-run inference purely for visualization here, 
            # or rely on the sorting loop. For smoother video, 
            # let's just stream the raw frame or basic text.
            # To keep it fast, we won't run full YOLO again here.
            # We just add the text of what the system "thinks" it sees.
            cv2.putText(frame, f"Status: {system_status}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Seeing: {last_detected}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/toggle', methods=['POST'])
def toggle_sorting():
    global is_sorting_active
    data = request.json
    action = data.get('action')
    
    if action == 'start':
        is_sorting_active = True
        return jsonify({"status": "started", "message": "Sorting started"})
    elif action == 'stop':
        is_sorting_active = False
        return jsonify({"status": "stopped", "message": "Sorting stopped"})
    
    return jsonify({"error": "Invalid action"}), 400

@app.route('/api/status')
def get_status():
    return jsonify({
        "active": is_sorting_active,
        "status": system_status,
        "detected": last_detected
    })

# ==========================================
# MAIN ENTRY POINT
# ==========================================
if __name__ == '__main__':
    # Start background threads
    t_cam = threading.Thread(target=camera_loop)
    t_cam.daemon = True
    t_cam.start()
    
    t_sort = threading.Thread(target=sorting_logic_loop)
    t_sort.daemon = True
    t_sort.start()
    
    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)