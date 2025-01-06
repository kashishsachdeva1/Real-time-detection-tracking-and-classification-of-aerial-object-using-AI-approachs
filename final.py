import cv2
import numpy as np
import joblib
from ultralytics import YOLO

# Constants
MAX_DISTANCE = 100
CONFIDENCE_THRESHOLD = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_COLOR = (255, 0, 0)  # Blue for text
FONT_THICKNESS = 2
LINE_COLOR = (0, 0, 255)  # Red for tracking lines
SEQ_LENGTH = 5  # Length of sequence for using the last 5 frames

# Load models
yolo_model = YOLO("best.pt")
rf_model = joblib.load('random_forest_model1.pkl')  # Load the Random Forest model
scaler = joblib.load('scaler.pkl')  # Load the scaler for feature normalization

# Initialize tracking variables
next_id = 0
tracked_objects = {}
object_paths = {}
object_velocities = {}
object_accelerations = {}
object_frequencies = {}
frame_data = {}  # Dictionary to store features for each object over frames

# Initialize a deque for holding previous frame data for each tracked object
from collections import deque
previous_frames = {i: deque(maxlen=SEQ_LENGTH) for i in range(1000)}  # Assume max 1000 objects

# Initialize the video capture from the laptop camera
cap = cv2.VideoCapture("D4.mp4")  # 0 is the default camera index
if not cap.isOpened():
    raise ValueError("Error: Cannot open camera")

# Set the desired frame width and height (e.g., 1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set the height

# Get frame rate for calculations
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_time = 1 / fps if fps > 0 else 1 / 30  # Time between frames in seconds (default to 30 FPS if unavailable)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def associate_detections(tracked_objects, current_frame_objects):
    updated_tracked_objects = {}
    unmatched_current_objects = current_frame_objects.copy()

    for obj_id, data in tracked_objects.items():
        prev_cx, prev_cy, prev_x, prev_y, prev_w, prev_h = data
        best_match = None
        min_distance = float('inf')

        for (cx, cy, x, y, w, h, class_id) in unmatched_current_objects:
            distance = euclidean_distance((cx, cy), (prev_cx, prev_cy))
            if distance < min_distance and distance < MAX_DISTANCE:
                min_distance = distance
                best_match = (cx, cy, x, y, w, h, class_id)

        if best_match:
            updated_tracked_objects[obj_id] = best_match[:6]
            unmatched_current_objects.remove(best_match)
            object_paths[obj_id].append(best_match[:2])

            # Calculate velocity
            prev_pos = np.array((prev_cx, prev_cy))
            curr_pos = np.array(best_match[:2])
            velocity_vector = (curr_pos - prev_pos) / frame_time
            velocity_magnitude = np.linalg.norm(velocity_vector)
            object_velocities[obj_id].append(velocity_magnitude)

            # Calculate acceleration
            if len(object_velocities[obj_id]) > 1:
                acceleration = (object_velocities[obj_id][-1] - object_velocities[obj_id][-2]) / frame_time
            else:
                acceleration = 0
            object_accelerations[obj_id].append(acceleration)

            # Calculate y-frequency
            y_changes = [abs(object_paths[obj_id][i][1] - object_paths[obj_id][i - 1][1]) 
                        for i in range(1, len(object_paths[obj_id]))]
            frequency = len([change for change in y_changes if change > 0]) / frame_time
            object_frequencies[obj_id].append(frequency)

            # Store current features for prediction
            features = np.array([
                best_match[0], best_match[1], 
                velocity_magnitude, acceleration, frequency
            ])
            previous_frames[obj_id].append(features)

        else:
            updated_tracked_objects[obj_id] = data

    return updated_tracked_objects, unmatched_current_objects

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detection with YOLO model
    results = yolo_model(frame)
    current_frame_objects = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            w, h = x2 - x1, y2 - y1
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            if confidence > CONFIDENCE_THRESHOLD:
                current_frame_objects.append((cx, cy, int(x1), int(y1), int(w), int(h), class_id))

    tracked_objects, unmatched_current_objects = associate_detections(tracked_objects, current_frame_objects)

    # Draw tracking lines, bounding boxes, and display object data
    for obj_id, (cx, cy, x, y, w, h) in tracked_objects.items():
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Prepare features for Random Forest model
        if len(previous_frames[obj_id]) == SEQ_LENGTH:  # Only predict if we have enough frames
            features_sequence = np.array(list(previous_frames[obj_id]))
            # Flatten the sequence for the model input
            feature_vector = features_sequence.flatten().reshape(1, -1)
            feature_vector = scaler.transform(feature_vector)  # Normalize features

            # Perform prediction using Random Forest model
            predicted_class = rf_model.predict(feature_vector)[0]

            # Mapping predicted class to string labels
            class_label = ''
            if predicted_class == 0:  # Assuming class labels are integers starting from 0
                class_label = 'drone'
            elif predicted_class == 1:
                class_label = 'eagle'
            elif predicted_class == 2:
                class_label = 'bird'

            # Display classification label
            object_info = f'ID: {obj_id} Class: {class_label}'
            cv2.putText(frame, object_info, (x, y - 15), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

            # Draw tracking line
            if obj_id in object_paths and len(object_paths[obj_id]) > 1:
                for i in range(1, len(object_paths[obj_id])):
                    cv2.line(frame, object_paths[obj_id][i - 1], object_paths[obj_id][i], LINE_COLOR, 2)

    for (cx, cy, x, y, w, h, class_id) in unmatched_current_objects:
        tracked_objects[next_id] = (cx, cy, x, y, w, h)
        object_paths[next_id] = [(cx, cy)]
        object_velocities[next_id] = [0]
        object_accelerations[next_id] = [0]
        object_frequencies[next_id] = [0]
        previous_frames[next_id] = deque(maxlen=SEQ_LENGTH)  # Initialize deque for the new object
        next_id += 1

    # Display the frame
    cv2.namedWindow("Real-Time Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-Time Detection", width, height)
    cv2.imshow("Real-Time Detection", frame)
    

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
