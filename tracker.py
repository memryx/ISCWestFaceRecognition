import cv2
import numpy as np
from pathlib import Path
from modules.MXFace import MXFace
from modules.bytetracker import BYTETracker

# Initialize models
mx_face = MXFace(models_dir=Path('assets/models'))
tracker = BYTETracker()

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detection
    mx_face.put(rgb_frame)
    annotated_frame = mx_face.get()

    dets = []
    for face in annotated_frame.detected_faces:
        x, y, w, h = face.bbox
        conf = 0.99  # Dummy confidence
        cls_id = 0   # Dummy class ID for faces
        dets.append(np.array([x, y, x+w, y+h, conf, cls_id]))

    dets = np.array(dets, dtype=np.float32)

    # Tracking
    if len(dets) > 0:
        tracked_objects = tracker.update(dets, None)

        # Draw tracked boxes
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, cls, score = obj.astype(int)
            label = f"ID: {track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    cv2.imshow('Face Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
mx_face.stop()



