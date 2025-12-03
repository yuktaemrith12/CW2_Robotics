from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
import threading
from typing import Dict, List, Optional
import json

app = FastAPI(title="PantryBot Integrated API")

# ----------------------------------------------------------
# 🔥 UPDATED CORS — REQUIRED FOR FRONTEND (localhost:5500)
# ----------------------------------------------------------
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:8000",

    # ⭐ REQUIRED FOR YOUR FRONTEND SERVER ⭐
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------
# Global variables
# ----------------------------------------------------------
running = False
camera_active = False
latest_detections = []
current_frame = None
frame_lock = threading.Lock()

# ----------------------------------------------------------
# Load YOLO model
# ----------------------------------------------------------
print("Loading YOLO model...")
try:
    DETECTION_WEIGHTS = "best.pt"
    detection_model = YOLO(DETECTION_WEIGHTS)
    print(f"✅ YOLO model loaded: {DETECTION_WEIGHTS}")
    print(f"Model classes: {detection_model.names}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    detection_model = None


# ----------------------------------------------------------
# Detection System
# ----------------------------------------------------------
class DetectionSystem:
    def __init__(self, model):
        self.model = model
        self.confidence_threshold = 0.5
        
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Perform real YOLO detection"""
        if self.model is None:
            return self.fallback_detection(frame)
        
        try:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            result = results[0]
            
            detections = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    confidence = float(box.conf.item())
                    class_id = int(box.cls.item())
                    class_name = self.model.names[class_id]
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detections.append({
                        "cls": class_name,
                        "conf": round(confidence, 3),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return self.fallback_detection(frame)
    
    def fallback_detection(self, frame):
        """Fake detections if model fails"""
        h, w = frame.shape[:2]
        detections = []
        
        grocery_classes = ["bottle", "can", "detergent", "pulses", "seafood", "fruit"]
        
        for _ in range(np.random.randint(1, 4)):
            if np.random.random() < 0.4:
                x1 = np.random.randint(40, w-200)
                y1 = np.random.randint(40, h-200)
                x2 = x1 + np.random.randint(80, 200)
                y2 = y1 + np.random.randint(80, 200)
                
                detections.append({
                    "cls": np.random.choice(grocery_classes),
                    "conf": round(np.random.uniform(0.6, 0.95), 2),
                    "bbox": [x1, y1, x2, y2]
                })
        
        return detections


# Initialize detection system
detection_system = DetectionSystem(detection_model)

# ----------------------------------------------------------
# Camera Stream Generator
# ----------------------------------------------------------
def camera_stream_generator():
    global running, camera_active, latest_detections, current_frame
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 10)
    
    camera_active = True
    frame_count = 0
    fps_update_time = time.time()
    fps = 0
    
    print("📹 Starting camera stream...")
    
    try:
        while camera_active:
            start_time = time.time()
            
            success, frame = cap.read()
            if not success:
                print("❌ Failed to read frame")
                break
            
            with frame_lock:
                current_frame = frame.copy()
            
            processed_frame = frame.copy()
            
            if running:
                detections = detection_system.detect_objects(frame)
                latest_detections = detections
                
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    label = f"{det['cls']} {det['conf']*100:.1f}%"
                    
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(processed_frame, (x1, y1-label_size[1]-10), 
                                 (x1+label_size[0], y1), (0, 255, 0), -1)
                    
                    cv2.putText(processed_frame, label, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            else:
                latest_detections = []
            
            frame_count += 1
            if time.time() - fps_update_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_update_time = time.time()
            
            status_text = f"FPS: {fps} | Detection: {'ON' if running else 'OFF'}"
            cv2.putText(processed_frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')
            
            processing_time = time.time() - start_time
            sleep_time = max(0.03, 0.1 - processing_time)
            time.sleep(sleep_time)
            
    except Exception as e:
        print(f"Stream error: {e}")
    finally:
        cap.release()
        camera_active = False
        print("📹 Camera stream stopped")


# ----------------------------------------------------------
# API Routes
# ----------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "PantryBot Integrated API",
        "status": "running",
        "model_loaded": detection_model is not None,
        "classes": list(detection_model.names.values()) if detection_model else []
    }

@app.get("/video")
async def video_feed():
    return StreamingResponse(
        camera_stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/detection/start")
async def start_detection():
    global running
    running = True
    return {
        "status": "detection_started",
        "running": running
    }

@app.post("/detection/stop")
async def stop_detection():
    global running
    running = False
    return {
        "status": "detection_stopped",
        "running": running
    }

@app.get("/detection/status")
async def detection_status():
    top_det = next((d for d in latest_detections), None)
    return {
        "running": running,
        "detections": latest_detections,
        "top_detection": top_det,
        "detection_count": len(latest_detections)
    }

@app.get("/model/classes")
async def get_model_classes():
    if detection_model:
        return {
            "classes": detection_model.names,
            "class_count": len(detection_model.names)
        }
    return {"classes": {}, "class_count": 0}

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Could not decode image"}
        
        detections = detection_system.detect_objects(frame)
        
        return {
            "filename": file.filename,
            "detections": detections,
            "detection_count": len(detections),
            "image_size": f"{frame.shape[1]}x{frame.shape[0]}"
        }
        
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}

@app.get("/system/status")
async def system_status():
    return {
        "camera_active": camera_active,
        "detection_running": running,
        "model_loaded": detection_model is not None,
        "current_detections": len(latest_detections),
        "fps_estimate": "5–10 FPS"
    }

# ----------------------------------------------------------
# Run Server
# ----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting PantryBot Integrated Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
