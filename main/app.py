import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import pyautogui
import collections
import threading
import queue
import tkinter as tk

# -------------------------
# 1) Global & Initial Variables
# -------------------------
is_running = False  # Flag to stop the gesture loop
frame_queue = queue.Queue(maxsize=2)

# ----- Parameters (default values) -----
speed_factor = 4
vel_alpha = 0.1
sudden_change_threshold = 30
small_threshold = 4

# Background variables
bg = None
bg_frames = 30
bg_counter = 0
gesture_buffer = collections.deque(maxlen=7)

# Kalman filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.04
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.02
kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
kalman.statePre = np.array([[0], [0], [0], [0]], dtype=np.float32)

# Gesture/Mouse state
mode = None
locked = False
in_drag = False
prev_cx, prev_cy = None, None
velocity_x, velocity_y = 0, 0

# PyAutoGUI failsafe off (optional)
pyautogui.FAILSAFE = False

# -------------------------
# 2) Resource Path Setup
# -------------------------
def resource_base_path():
    """Helper to find base path for PyInstaller or normal script."""
    if hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.abspath(".")

base_path = resource_base_path()

# -------------------------
# 3) Load Model and Icons
# -------------------------
model_path = os.path.join(base_path, "model.h5")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

class_labels = ["doubleClick", "drag", "drop", "leftClick", "moveCursor", "rightClick"]

gesture_icons_info = {
    "doubleClick": (os.path.join(base_path, "icons", "doubleClick.png"), "Double Click"),
    "drag":        (os.path.join(base_path, "icons", "drag.png"),        "Click and hold (Drag)"),
    "drop":        (os.path.join(base_path, "icons", "drop.png"),        "Release mouse (Drop)"),
    "leftClick":   (os.path.join(base_path, "icons", "leftClick.png"),   "Left Mouse Click"),
    "moveCursor":  (os.path.join(base_path, "icons", "moveCursor.png"),  "Move Mouse Pointer"),
    "rightClick":  (os.path.join(base_path, "icons", "rightClick.png"),  "Right Mouse Click"),
}

gesture_icons = {}
icon_size = (50, 50)
for gesture, (filename, desc) in gesture_icons_info.items():
    if os.path.exists(filename):
        icon = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if icon is not None:
            icon = cv2.resize(icon, icon_size, interpolation=cv2.INTER_AREA)
            gesture_icons[gesture] = icon
        else:
            print(f"Warning: Could not read file {filename}")
            gesture_icons[gesture] = None
    else:
        print(f"Warning: File {filename} does not exist.")
        gesture_icons[gesture] = None

# -------------------------
# 4) Helper Functions
# -------------------------
def run_avg(image, accumWeight=0.5):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
    else:
        cv2.accumulateWeighted(image, bg, accumWeight)

def segment(image, threshold=25, min_area=1000):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(thresholded, 50, 150)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    segmented = max(contours, key=cv2.contourArea)
    if cv2.contourArea(segmented) < min_area:
        return None, None
    return thresholded, (segmented, edges)

def kalman_predict_and_correct(cx, cy):
    global kalman
    measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
    kalman.correct(measurement)
    prediction = kalman.predict()
    return prediction[0][0], prediction[1][0]

def perform_mouse_action(gesture):
    if gesture == "doubleClick":
        pyautogui.doubleClick()
    elif gesture == "leftClick":
        pyautogui.click()
    elif gesture == "rightClick":
        pyautogui.rightClick()

def move_cursor(dx, dy):
    """If mode == 'drag', hold left mouse and moveRel; else normal move."""
    global mode, speed_factor
    offset_x = int(dx * speed_factor)
    offset_y = int(dy * speed_factor)
    
    if mode == "drag":
        # OS-level drag approach: single mouseDown + repeated moveRel
        pyautogui.moveRel(offset_x, offset_y, duration=0)
    else:
        # Normal movement
        current_x, current_y = pyautogui.position()
        new_x = current_x + offset_x
        new_y = current_y + offset_y
        
        screen_width, screen_height = pyautogui.size()
        margin = 10
        new_x = max(margin, min(screen_width - margin, new_x))
        new_y = max(margin, min(screen_height - margin, new_y))
        
        pyautogui.moveTo(new_x, new_y, duration=0)

# -------------------------
# 5) Gesture Recognition Thread
# -------------------------
def gesture_loop():
    """Main loop for gesture recognition in a separate thread."""
    global is_running
    global cap
    global mode, locked, in_drag
    global bg, bg_counter, velocity_x, velocity_y, prev_cx, prev_cy
    global speed_factor, vel_alpha, sudden_change_threshold, small_threshold

    # Re-init some variables each time we start
    bg_counter = 0
    bg = None

    while is_running:
        if frame_queue.empty():
            cv2.waitKey(1)
            continue

        frame = frame_queue.get()
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        height, width = clone.shape[:2]

        # ROI
        roi_top, roi_right, roi_bottom, roi_left = 20, 300, 350, 640
        roi = clone[roi_top:roi_bottom, roi_right:roi_left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Background Calibration
        if bg_counter < bg_frames:
            run_avg(gray)
            bg_counter += 1
            cv2.putText(clone, f"Calibrating background: {bg_counter}/{bg_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(clone, (roi_right, roi_top), (roi_left, roi_bottom), (0, 255, 0), 2)
            cv2.imshow("Gesture Recognition", clone)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                is_running = False
                break
            continue

        # Hand Segmentation & Gesture Prediction
        thresholded, seg_result = segment(gray)
        current_gesture = None
        cx = cy = None

        if seg_result is not None:
            segmented, edges = seg_result
            roi[edges > 0] = [0, 255, 0]

            # Centroid
            M = cv2.moments(segmented)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(roi, (cx, cy), 5, (255, 0, 0), -1)

            # Prepare thresholded image for model
            thresh_copy = cv2.resize(thresholded, (200, 200))
            thresh_copy = thresh_copy.astype("float32") / 255.0
            thresh_copy = np.expand_dims(thresh_copy, axis=-1)
            thresh_copy = np.expand_dims(thresh_copy, axis=0)

            predictions = model.predict(thresh_copy)
            confidence = np.max(predictions)
            predicted_class = np.argmax(predictions)

            if confidence > 0.75:
                gesture_buffer.append(predicted_class)
            else:
                gesture_buffer.append(-1)

            most_common, count = collections.Counter(gesture_buffer).most_common(1)[0]
            if most_common != -1 and count >= 3:
                current_gesture = class_labels[most_common]

            cv2.imshow("Thresholded", thresholded)
        else:
            cv2.putText(clone, "No hand detected", (width - 220, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Mode & Gesture Logic
        global mode
        if current_gesture == "moveCursor":
            mode = "moveCursor"
            locked = False
        elif current_gesture == "drag":
            if not in_drag:
                mode = "drag"
                in_drag = True
                pyautogui.mouseDown(button='left')
            else:
                mode = "drag"
        elif current_gesture == "drop":
            if in_drag:
                pyautogui.mouseUp(button='left')
                in_drag = False
                mode = "drop"
                locked = False
            elif locked:
                locked = False
                mode = "drop"
            else:
                mode = None
        elif current_gesture in ["leftClick", "rightClick", "doubleClick"]:
            if not locked:
                mode = current_gesture
                perform_mouse_action(current_gesture)
                locked = True
        else:
            if not locked and not in_drag:
                mode = None

        # Movement with Kalman Filter
        global prev_cx, prev_cy, velocity_x, velocity_y
        if mode in ["moveCursor", "drag"] and cx is not None and cy is not None:
            cx_smooth, cy_smooth = kalman_predict_and_correct(cx, cy)
            if prev_cx is not None and prev_cy is not None:
                dx = cx_smooth - prev_cx
                dy = cy_smooth - prev_cy

                # Filter out sudden large jumps
                if abs(dx) > sudden_change_threshold or abs(dy) > sudden_change_threshold:
                    dx, dy = 0, 0

                # Update velocity with a lower alpha for smoother changes
                if abs(dx) >= small_threshold or abs(dy) >= small_threshold:
                    velocity_x = vel_alpha * dx + (1 - vel_alpha) * velocity_x
                    velocity_y = vel_alpha * dy + (1 - vel_alpha) * velocity_y
            else:
                dx, dy = 0, 0

            prev_cx, prev_cy = cx_smooth, cy_smooth
            move_cursor(velocity_x, velocity_y)
        else:
            velocity_x, velocity_y = 0, 0

        # Display
        x_icon = 10
        y_icon = 10
        spacing = 70

        for i, gesture in enumerate(class_labels):
            icon = gesture_icons.get(gesture, None)
            _, gesture_desc = gesture_icons_info[gesture]
            y_offset = y_icon + i * spacing

            if icon is not None:
                h, w_ = icon.shape[:2]
                roi_icon = clone[y_offset:y_offset+h, x_icon:x_icon+w_]
                if icon.shape[2] == 4:  # alpha channel
                    icon_bgr = icon[:, :, :3]
                    alpha_mask = icon[:, :, 3] / 255.0
                    for c in range(3):
                        roi_icon[:, :, c] = (alpha_mask * icon_bgr[:, :, c] +
                                             (1 - alpha_mask) * roi_icon[:, :, c])
                else:
                    clone[y_offset:y_offset+h, x_icon:x_icon+w_] = icon

                cv2.putText(clone, gesture_desc, (x_icon + w_ + 10, y_offset + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(clone, f"{gesture} - {gesture_desc}",
                            (x_icon, y_offset + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(clone, f"Mode: {mode}", (width - 220, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.rectangle(clone, (roi_right, roi_top), (roi_left, roi_bottom), (0, 255, 0), 2)
        cv2.imshow("Gesture Recognition", clone)

        # Check for 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            break

    print("Gesture loop ended.")
    cv2.destroyAllWindows()

# -------------------------
# 6) Webcam Initialization
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

def capture_frames():
    """Continuously capture frames from webcam to frame_queue."""
    global is_running, cap
    while is_running:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)

# -------------------------
# 7) Tkinter UI
# -------------------------
def on_speed_factor(val):
    global speed_factor
    speed_factor = float(val)

def on_vel_alpha(val):
    global vel_alpha
    vel_alpha = float(val)

def on_sudden_threshold(val):
    global sudden_change_threshold
    sudden_change_threshold = float(val)

def on_small_threshold(val):
    global small_threshold
    small_threshold = float(val)

def start_gesture():
    global is_running
    if is_running:
        return  # already running
    is_running = True

    # Start capture thread
    threading.Thread(target=capture_frames, daemon=True).start()

    # Start gesture thread
    threading.Thread(target=gesture_loop, daemon=True).start()

def stop_gesture():
    global is_running
    is_running = False

def build_ui():
    window = tk.Tk()
    window.title("Gesture Settings")

    # Speed Factor
    tk.Label(window, text="Speed Factor").pack()
    speed_scale = tk.Scale(window, from_=1, to=10, resolution=1, orient=tk.HORIZONTAL, command=on_speed_factor)
    speed_scale.set(speed_factor)
    speed_scale.pack()

    # Velocity Alpha
    tk.Label(window, text="Velocity Alpha").pack()
    vel_scale = tk.Scale(window, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, command=on_vel_alpha)
    vel_scale.set(vel_alpha)
    vel_scale.pack()

    # Sudden Change Threshold
    tk.Label(window, text="Sudden Change Threshold").pack()
    sudden_scale = tk.Scale(window, from_=5, to=100, resolution=5, orient=tk.HORIZONTAL, command=on_sudden_threshold)
    sudden_scale.set(sudden_change_threshold)
    sudden_scale.pack()

    # Small Threshold
    tk.Label(window, text="Small Movement Threshold").pack()
    small_scale = tk.Scale(window, from_=1, to=20, resolution=1, orient=tk.HORIZONTAL, command=on_small_threshold)
    small_scale.set(small_threshold)
    small_scale.pack()

    # Start/Stop
    btn_frame = tk.Frame(window)
    btn_frame.pack(pady=10)

    start_btn = tk.Button(btn_frame, text="Start Gesture", command=start_gesture)
    start_btn.pack(side=tk.LEFT, padx=5)

    stop_btn = tk.Button(btn_frame, text="Stop Gesture", command=stop_gesture)
    stop_btn.pack(side=tk.LEFT, padx=5)

    window.protocol("WM_DELETE_WINDOW", on_close(window))
    window.mainloop()

def on_close(window):
    """Callback to stop threads & close window."""
    def inner():
        stop_gesture()
        window.destroy()
    return inner

# -------------------------
# 8) Main Entry
# -------------------------
if __name__ == "__main__":
    # We'll just build the UI. The user can Start/Stop gesture from there.
    build_ui()

    # Once the UI is closed, release webcam if open
    cap.release()
    cv2.destroyAllWindows()
