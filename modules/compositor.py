import queue
import numpy as np
import cv2

from PySide6.QtCore import QTimer, QObject, Signal

class Compositor(QObject):
    frame_ready = Signal(np.ndarray)

    def __init__(self, face_tracker, interval=33, parent=None):
        super().__init__(parent)
        self.face_tracker = face_tracker
        self.timer = QTimer(self)
        self.timer.setInterval(interval)
        self.timer.timeout.connect(self.poll_queue)

        self.show_conf = False

    def start(self):
        self.timer.start()

    def stop(self):
        print("Shutting down Compositor")
        self.timer.stop()

    def draw_boxes(self, composite_frame):

        faces = []
        frame = composite_frame.image
        for tracked_object in self.face_tracker.tracker_dict.values():
            faces.append((tracked_object.bbox, tracked_object.name))

        for obj in composite_frame.tracked_objects:
            (left, top, right, bottom) = obj.bbox
            name = obj.track_id

            #profile_name, distance_list = face_database.find(face.embedding)
            #face.person_id = profile_name

            label = f'{name}'
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            if False: #self.conf_checkbox.isChecked():
                for i, (name, distance) in enumerate(distance_list):
                    if i == 3:
                        break
                    label = f'{name}: {distance:.1f}'
                    cv2.putText(frame, label, (left + 10, top + 10 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            if True: #self.bbox_checkbox.isChecked():
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw keypoints if checkbox is checked
            if False: #self.keypoints_checkbox.isChecked():
                for (x, y) in face.keypoints:
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            # Draw semi-transparent rectangle if mouse is inside bounding box
            if False: #self.mouse_position:
                mouse_x, mouse_y = self.mouse_position
                if left <= mouse_x <= left + width and top <= mouse_y <= top + height:
                    overlay = frame.copy()
                    alpha = 0.5  # Transparency factor
                    cv2.rectangle(overlay, (left, top), (left + width, top + height), (0, 0, 255), -1)
                    # Apply the overlay
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def poll_queue(self):
        try:
            # Use get_nowait so that we do not block the timer callback
            composite_frame = self.face_tracker.composite_queue.get_nowait() 
            frame = self.draw_boxes(composite_frame)

            self.frame_ready.emit(frame)
        except queue.Empty:
            pass  # No new frame available at the moment
