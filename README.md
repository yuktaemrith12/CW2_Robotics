

# **SortiFy – Grocery Sorting Robot **

SortiFy is a vision-based robotic system built for the **AI in Robotics (PDE3802)** module.
The system combines:

* A **YOLOv8 detection model**,
* A **Raspberry Pi**,
* A **Yahboom DOFBOT 6-DOF robotic arm**,

to automatically **detect**, **pick**, and **sort grocery items** into colour-coded bins.

---

## 📦 1. Project Structure

```
CW2_ROBOTICS/
│
├── Classification/              # Early experiments using YOLOv8-CLS
│
├── Detection/                   # Final detection model pipeline
│   ├── dataset/                 # Real + synthetic images
│   ├── metric_results/          # Precision, recall, mAP, confusion matrix
│   └── scripts/
│       ├── _00_normalize.py
│       ├── _01_labels.py
│       ├── _02_boundary.py
│       ├── _03_draw_boundary.py
│       ├── _04_split_dataset.py
│       ├── _05_Grocery_YOLO_Detection_Model.ipynb
│       └── _06_Fine_Tuning.py
│
├── synthetic_data_aug/          # Synthetic data generation
│   ├── class_folders/
│   ├── house_room_background/
│   ├── _01_normalize_background.py
│   └── _02_synthetic_data.py
│
└── Dofbot/                      # Final deployed robot system
    ├── app.py                   # YOLO inference + robot control + camera stream
    ├── best.pt                  # Final trained YOLOv8 weights
    ├── index.html               # Web dashboard (live feed + control)
    └── images/
```

---

## 🚀 2. Features

### Vision System

* Real-time YOLOv8 detection (>5 FPS on Raspberry Pi)
* 6 trained classes: **bottle, can, detergent, fruit, pulses, seafood**
* Confirm detection over several frames to avoid errors

### Robotic System

* DOFBOT performs **pick → move → drop → return home**
* Pre-calibrated joint coordinates for each bin
* Safe idle position between tasks

### Web Interface

* Live camera feed (MJPEG)
* Start/Stop sorting button
* Detected object + confidence
* System status and robot action updates

---

## 🧠 3. Data & Model Training

### Real Dataset

Prepared using:

* `_00_normalize.py`
* `_01_labels.py`
* `_02_boundary.py`
* `_04_split_dataset.py`

### Synthetic Dataset

Generated using:

* `_01_normalize_background.py`
* `_02_synthetic_data.py`

Synthetic augmentation includes:

* random placement
* rotation, scale, noise, blur
* automatic YOLO label generation

### Training

The model was trained and fine-tuned in:
`_05_Grocery_YOLO_Detection_Model.ipynb`

Evaluation results are in `metric_results/`.

---

## 🔧 4. Installation

### Install dependencies

```
pip install -r requirements.txt
```

### Install DOFBOT Arm Library

Follow Yahboom’s setup guide for `Arm_Lib`.

---

## ▶️ 5. Running the System

### On Raspberry Pi (Robot):

```
cd CW2_ROBOTICS/Dofbot
python app.py
```

Then open in a browser:

```
http://<raspberry-pi-ip>:5000
```

### On Laptop (Model Testing):

```
cd Model_Testing
python app.py
```

---

## ⚙️ 6. How It Works

1. Camera sends frames to the Raspberry Pi.
2. YOLOv8 detects the grocery item.
3. If the same class is detected for several frames:

   * DOFBOT moves to pickup position
   * Grabs the object
   * Places it in the correct bin
4. System returns to idle and waits for the next item.

---



