# Sortify – Grocery Sorting Robot

Sortify is a Dofbot with a camera that automatically recognises grocery items and places them into coloured boxes.  
It is designed to help elderly users or people with limited mobility organise their groceries with minimal effort.

---

## 1. Project Overview

**Main components**

- **Raspberry Pi / PC** – runs the backend API and the detection model.
- **Camera** – points at the pickup area where the grocery item is placed.
- **Robotic Arm (DOFBOT)** – moves the item from the pickup area to the correct box.
- **YOLO detection model (`best.pt`)** – recognises the grocery class (Bottle, Can, Detergent, Fruit, Pulses, Seafood, …).
- **Web Frontend (`index.html`)** – simple dashboard called *Sortify* for elderly-friendly control.

**Basic workflow**

1. User opens the Sortify dashboard in a browser.
2. User presses **START** and places **one grocery item** in the pickup area.
3. The camera sends frames to the YOLO detection model.
4. When a class is stable with good confidence, the robot moves the item to its assigned box.
5. System returns to idle and asks for the next item.
6. User presses **STOP** when finished.

---

## 2. Folder Structure

```text
project-root/
  app.py             # FastAPI / Flask backend (detection & robot control)
  best.pt            # Trained YOLO detection model
  index.html         # Frontend UI (Sortify dashboard)
  images/
    background.jpg   # Background image for the UI (optional)
    logo.png         # Sortify logo shown in the header
