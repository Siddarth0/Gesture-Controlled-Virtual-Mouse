import cv2
import numpy as np
import tensorflow as tf

# Load your trained model.
model = tf.keras.models.load_model('model.h5')
print("Model loaded successfully.")

# The order of class labels must match your model's output order.
class_labels = ["doubleClick", "drag", "drop", "leftClick", "moveCursor", "rightClick"]

# Initialize webcam.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define ROI coordinates (adjust as needed).
roi_top, roi_right, roi_bottom, roi_left = 20, 300, 350, 640

# Background model variables.
bg = None
bg_frames = 30  # Number of frames to calibrate background.
bg_counter = 0

def run_avg(image, accumWeight=0.5):
    """Update the background model using a running average."""
    global bg
    if bg is None:
        bg = image.copy().astype("float")
    else:
        cv2.accumulateWeighted(image, bg, accumWeight)

def segment(image, threshold=25, min_area=1000):
    """
    Segments the hand region by computing the absolute difference
    between the background and the current frame.
    Returns (thresholded image, segmented contour) if found, else None.
    """
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Clean up the thresholded image with morphological operations.
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    seg = max(contours, key=cv2.contourArea)
    if cv2.contourArea(seg) < min_area:
        return None
    return thresh, seg

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for mirror effect.
    frame = cv2.flip(frame, 1)
    clone = frame.copy()

    # Extract ROI and preprocess.
    roi = clone[roi_top:roi_bottom, roi_right:roi_left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Background calibration (first few frames).
    if bg_counter < bg_frames:
        run_avg(gray)
        bg_counter += 1
        cv2.putText(clone, f"Calibrating background {bg_counter}/{bg_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.rectangle(clone, (roi_right, roi_top), (roi_left, roi_bottom), (0, 255, 0), 2)
        cv2.imshow("Gesture Recognition", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue

    # Segment the hand region.
    seg_result = segment(gray)
    if seg_result is not None:
        thresh, seg = seg_result

        # Draw the segmented contour on the ROI.
        cv2.drawContours(roi, [seg], -1, (0, 255, 0), 2)
        
        # Prepare the thresholded image for prediction.
        thresh_resized = cv2.resize(thresh, (200, 200))
        thresh_normalized = thresh_resized.astype("float32") / 255.0
        thresh_normalized = np.expand_dims(thresh_normalized, axis=-1)
        thresh_normalized = np.expand_dims(thresh_normalized, axis=0)

        # Run the model prediction.
        predictions = model.predict(thresh_normalized)
        confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)
        gesture = class_labels[predicted_class]

        # Display the recognized gesture and its confidence.
        text = f"Gesture: {gesture} ({confidence:.2f})"
        cv2.putText(clone, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Thresholded", thresh)
    else:
        cv2.putText(clone, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw the ROI on the main frame.
    cv2.rectangle(clone, (roi_right, roi_top), (roi_left, roi_bottom), (0, 255, 0), 2)
    cv2.imshow("Gesture Recognition", clone)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
