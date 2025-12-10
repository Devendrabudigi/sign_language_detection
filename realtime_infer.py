# realtime_infer.py
import cv2
import numpy as np
import tensorflow as tf
import json
from pathlib import Path

MODEL_PATH = "models/sign_cnn.h5"
METADATA = "models/metadata.json"
IMG_SIZE = (128,128)  # same as training

# load model and metadata
model = tf.keras.models.load_model(MODEL_PATH)
with open(METADATA, "r") as f:
    meta = json.load(f)
class_names = meta["class_names"]

def preprocess_frame(frame):
    # frame is BGR from OpenCV, convert to RGB and resize
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return img

cap = cv2.VideoCapture(0)  # change index if multiple cameras
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: define ROI to the center where the hand should be
    h, w, _ = frame.shape
    # square region in center
    size = min(h, w) // 2
    cx, cy = w//2, h//2
    x1, y1 = cx - size//2, cy - size//2
    x2, y2 = cx + size//2, cy + size//2
    roi = frame[y1:y2, x1:x2]

    img = preprocess_frame(roi)
    input_tensor = np.expand_dims(img, axis=0)  # shape (1, H, W, 3)
    probs = model.predict(input_tensor)[0]
    idx = np.argmax(probs)
    label = class_names[idx]
    prob = probs[idx]

    # display results
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    text = f"{label}: {prob:.2f}"
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    cv2.imshow("Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
