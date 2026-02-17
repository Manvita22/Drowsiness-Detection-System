import cv2
import numpy as np
from keras.models import load_model
from pygame import mixer
import os
import sys
import time
import tkinter as tk
from PIL import Image, ImageTk

# Initialize mixer for sound
mixer.init()

# Configuration
IMG_SIZE = 24
DROWSINESS_THRESHOLD = 8   # lower threshold so drowsy triggers faster
CLOSED_PROB_THRESHOLD = 0.5  # lower cutoff to accept closed prediction sooner
REQUIRED_CLOSED_EYES = 2     # require both eyes closed to increment score
ALARM_ALERT_PATH = 'alarm_alert.wav'   # short beep for alert
ALARM_DROWSY_PATH = 'alarm_drowsy.wav' # louder alarm for drowsy
MODEL_PATH = 'models/cnnCat2.h5'

# Load cascade classifiers
cascade_path = cv2.data.haarcascades

try:
    face_cascade = cv2.CascadeClassifier(
        cascade_path + 'haarcascade_frontalface_default.xml'
    )
    left_eye_cascade = cv2.CascadeClassifier(
        cascade_path + 'haarcascade_eye_tree_eyeglasses.xml'
    )
    right_eye_cascade = cv2.CascadeClassifier(
        cascade_path + 'haarcascade_eye.xml'
    )
except Exception as e:
    print(f"Error loading cascade classifiers: {e}")
    sys.exit(1)

# Load pre-trained model
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    print("Please run train_model.py first to train the model.")
    sys.exit(1)

# Load alarm sounds if they exist
alarm_alert = None
alarm_drowsy = None

if os.path.exists(ALARM_ALERT_PATH):
    try:
        alarm_alert = mixer.Sound(ALARM_ALERT_PATH)
        print(f"Alert sound loaded: {ALARM_ALERT_PATH}")
    except Exception as e:
        print(f"Warning: Could not load alert sound: {e}")
else:
    print(f"Warning: Alert sound not found at {ALARM_ALERT_PATH}")

if os.path.exists(ALARM_DROWSY_PATH):
    try:
        alarm_drowsy = mixer.Sound(ALARM_DROWSY_PATH)
        print(f"Drowsy sound loaded: {ALARM_DROWSY_PATH}")
    except Exception as e:
        print(f"Warning: Could not load drowsy sound: {e}")
else:
    print(f"Warning: Drowsy sound not found at {ALARM_DROWSY_PATH}")

if not alarm_alert and not alarm_drowsy:
    print("The system will work without sound alerts.")

# Label mapping
labels = ['Closed', 'Open']

def predict_eye_state(eye_image):
    """Predict if eye is open or closed"""
    try:
        # Convert to grayscale
        eye_gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        # Resize to model input size
        eye_resized = cv2.resize(eye_gray, (IMG_SIZE, IMG_SIZE))
        # Normalize
        eye_normalized = eye_resized / 255.0
        # Expand dimensions for model input
        eye_input = np.expand_dims(eye_normalized, axis=(0, -1))
        # Predict
        prediction = model.predict(eye_input, verbose=0)
        return prediction[0]
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def detect_drowsiness():
    """Main drowsiness detection loop with Tkinter GUI (no cv2.imshow)."""
    print("=" * 60)
    print("DROWSINESS DETECTION SYSTEM")
    print("=" * 60)
    print(f"Threshold for drowsiness alert: {DROWSINESS_THRESHOLD} frames")
    print("Press 'q' or click Quit to exit")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    drowsiness_score = 0
    alarm_played = False
    frame_count = 0
    fps_time = time.time()
    fps = 0

    # Tkinter window setup
    root = tk.Tk()
    root.title("Drowsiness Detection System")
    root.geometry("900x700")

    video_label = tk.Label(root)
    video_label.pack(pady=10)

    status_var = tk.StringVar(value="Initializing camera...")
    score_var = tk.StringVar(value="Score: 0")
    fps_var = tk.StringVar(value="FPS: --")

    status_label = tk.Label(root, textvariable=status_var, font=("Arial", 16, "bold"))
    status_label.pack(pady=5)

    info_frame = tk.Frame(root)
    info_frame.pack(pady=5)
    tk.Label(info_frame, textvariable=score_var, font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
    tk.Label(info_frame, textvariable=fps_var, font=("Arial", 12)).pack(side=tk.LEFT, padx=10)

    running = {"value": True}
    update_job = None

    def on_close(event=None):
        running["value"] = False
        if update_job is not None:
            try:
                root.after_cancel(update_job)
            except Exception:
                pass
        cap.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("q", on_close)

    def process_frame():
        nonlocal drowsiness_score, alarm_played, frame_count, fps_time, fps

        ret, frame = cap.read()
        if not ret:
            status_var.set("Error: Failed to read frame. Check webcam access.")
            score_var.set("Score: --")
            fps_var.set("FPS: --")
            running["value"] = False
            root.after(1500, on_close)
            return None

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            drowsiness_score = 0
            status = "No face detected"
            color = (0, 255, 0)
        else:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            left_eyes = left_eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
            right_eyes = right_eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
            eyes = list(left_eyes) + list(right_eyes)

            if len(eyes) < 2:
                # If face is present but eyes aren't detected, treat as potentially closed
                drowsiness_score = min(drowsiness_score + 1, DROWSINESS_THRESHOLD + 3)
                status = "Eyes not detected (counting as closed)"
                color = (255, 165, 0)
            else:
                eye_states = []
                # Keep the two largest detections to avoid eyebrow/noise
                eyes_sorted = sorted(eyes, key=lambda a: a[2] * a[3], reverse=True)[:2]
                for (ex, ey, ew, eh) in eyes_sorted:
                    eye_roi = roi_color[ey:ey + eh, ex:ex + ew]
                    if eye_roi.size == 0:
                        continue
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    prediction = predict_eye_state(eye_roi)
                    if prediction is not None:
                        closed_prob = float(prediction[0])
                        open_prob = float(prediction[1])
                        eye_state = 'Closed' if closed_prob >= open_prob else 'Open'
                        confidence = max(closed_prob, open_prob) * 100
                        eye_states.append((eye_state, confidence, closed_prob, open_prob))

                if len(eye_states) > 0:
                    closed_eyes = 0
                    for state, conf, closed_prob, open_prob in eye_states:
                        if closed_prob >= CLOSED_PROB_THRESHOLD:
                            closed_eyes += 1

                    if closed_eyes >= REQUIRED_CLOSED_EYES:
                        drowsiness_score += 1
                    else:
                        drowsiness_score = max(0, drowsiness_score - 1)

                    if drowsiness_score > DROWSINESS_THRESHOLD:
                        status = f"DROWSY! Score: {drowsiness_score}"
                        color = (0, 0, 255)
                        if alarm_drowsy and not alarm_played:
                            try:
                                alarm_drowsy.play()
                                alarm_played = True
                            except Exception:
                                pass
                    else:
                        if drowsiness_score > DROWSINESS_THRESHOLD * 0.5:
                            status = f"Alert! Score: {drowsiness_score}"
                            color = (0, 165, 255)
                            if alarm_alert:
                                try:
                                    alarm_alert.play()
                                except Exception:
                                    pass
                            alarm_played = False
                        else:
                            status = f"Awake. Score: {drowsiness_score}"
                            color = (0, 255, 0)
                            alarm_played = False
                else:
                    drowsiness_score = max(0, drowsiness_score - 1)
                    status = "Unable to process eyes"
                    color = (0, 165, 255)

        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_time)
            fps_time = time.time()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        cv2.putText(frame, status, (10, height - 40), font, font_scale, color, thickness)
        cv2.putText(frame, f"FPS: {fps:.1f}", (width - 150, 30), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'q' or close window to quit", (10, 30), font, 0.6, (255, 255, 255), 1)

        # Convert to Tkinter-compatible image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        status_var.set(status)
        score_var.set(f"Score: {drowsiness_score}")
        fps_var.set(f"FPS: {fps:.1f}")

        return status

    def update_loop():
        nonlocal update_job
        if not running["value"]:
            return
        process_frame()
        update_job = root.after(10, update_loop)

    print("Starting webcam... Close window or press 'q' to exit")
    update_job = root.after(10, update_loop)
    root.mainloop()

    cap.release()
    print("Program terminated successfully")

if __name__ == "__main__":
    detect_drowsiness()
