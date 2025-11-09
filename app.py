# app.py — Classification-only Flask app (YOLOv8-CLS)
import io
import base64

from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np

# Ultralytics YOLO (classification)
from ultralytics import YOLO

app = Flask(__name__)

# -------------------- Model --------------------
YOLO_CLS_WEIGHTS = "grocery_yolov8cls.pt"

def load_classifier():
    model = YOLO(YOLO_CLS_WEIGHTS)  # YOLOv8 classification checkpoint
    try:
        model.fuse()                # small speed boost when supported
    except Exception:
        pass
    return model

clf_model = load_classifier()
CLASSES = [name for _, name in sorted(clf_model.names.items(), key=lambda kv: kv[0])]

# -------------------- Helpers --------------------
def _bytes_to_rgb_array(img_bytes: bytes) -> np.ndarray:
    """Decode bytes → RGB numpy array (YOLO handles resize/normalization)."""
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(pil)

def run_cls(image_rgb: np.ndarray):
    """Return (top1_class, top1_conf_pct, top_k list excluding top1 when needed)."""
    results = clf_model(image_rgb, verbose=False)
    res = results[0]
    top1_idx = int(res.probs.top1)
    top1_conf = float(res.probs.top1conf) * 100.0
    main_class = CLASSES[top1_idx]

    other_preds = []
    if top1_conf < 95.0 and hasattr(res.probs, "data"):
        probs = res.probs.data.cpu().numpy()
        topk_idx = probs.argsort()[-3:][::-1]
        for idx in topk_idx:
            name = CLASSES[int(idx)]
            if name == main_class:
                continue
            other_preds.append({"class": name, "confidence": f"{probs[int(idx)]*100:.2f}"})

    return main_class, f"{top1_conf:.2f}", other_preds

# -------------------- Routes --------------------
@app.route("/", methods=["GET"])
def home():
    """Landing page for classification UI."""
    return render_template("index.html", prediction=None, confidence=None)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Supports:
      - Multipart file upload (form file input)
      - Base64 data URL (webcam snapshot via canvas)
    Returns JSON for AJAX; falls back to server-render when not AJAX.
    """
    # Case A: file upload
    if "file" in request.files and request.files["file"].filename:
        raw = request.files["file"].read()
        img_rgb = _bytes_to_rgb_array(raw)
        pred, conf, others = run_cls(img_rgb)

        # Optional preview for UI
        try:
            buf = io.BytesIO()
            Image.open(io.BytesIO(raw)).convert("RGB").save(buf, format="PNG")
            data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            data_url = None

        # AJAX → JSON
        if request.form.get("ajax") == "1" or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({
                "prediction": pred,
                "confidence": conf,
                "other_predictions": others,
                "uploaded_preview": data_url
            })

        # Server-rendered fallback
        return render_template(
            "index.html",
            prediction=pred,
            confidence=conf,
            other_predictions=others,
            uploaded_preview=data_url
        )

    # Case B: base64 image (JSON)
    b64_str = request.json.get("image_data") if request.is_json else request.form.get("image_data")
    if not b64_str:
        return jsonify({"error": "No image provided"}), 400

    if "," in b64_str:  # strip 'data:image/...;base64,' if present
        b64_str = b64_str.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(b64_str)
        img_rgb = _bytes_to_rgb_array(img_bytes)
        pred, conf, others = run_cls(img_rgb)
        return jsonify({
            "prediction": pred,
            "confidence": conf,
            "other_predictions": others
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {e}"}), 400

@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200

# -------------------- Entrypoint --------------------
if __name__ == "__main__":
    # threaded=True so multiple requests (upload + webcam snapshot) are handled smoothly
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
