import queue
import numpy as np
import cv2
from PIL import Image  # Import PIL for handling .ico files

from PySide6.QtCore import QTimer, QObject, Signal
from PySide6.QtWidgets import QCheckBox

from .utils import Framerate

class MxColors: 
    LightBlue = (198, 234, 242)
    Blue = (53, 169, 188)
    DarkBlue =(20, 53, 85) 
    Teal = (60, 187, 187)

class Compositor(QObject):
    frame_ready = Signal(np.ndarray)

    def __init__(self, face_tracker, parent=None):
        super().__init__(parent)
        self.face_tracker = face_tracker
        self.framerate = Framerate()

        # Config
        self.mouse_position = (-1,-1)
        self.bbox_checkbox = QCheckBox("Draw Boxes")
        self.bbox_checkbox.setChecked(True)
        self.keypoints_checkbox = QCheckBox("Draw Keypoints")
        self.keypoints_checkbox.setChecked(False)
        self.distance_checkbox = QCheckBox("Show Similarity")
        self.distance_checkbox.setChecked(False)

        # Load icons and resize
        self.load_icons()

    def load_icons(self):
        #pil_im = Image.open("assets/icon.ico")
        #pil_im = pil_im.convert("RGBA")  # Ensure image has an alpha channel
        #icon = cv2.resize(np.array(pil_im), (100, 100))
        #self.icon_left = icon 

        icon = cv2.imread("assets/logo.png", cv2.IMREAD_UNCHANGED)
        icon = cv2.resize(icon, (100, 100))
        self.logo = cv2.cvtColor(icon, cv2.COLOR_RGBA2BGRA)

    def update_mouse_pos(self, pos):
        self.mouse_position = pos

    def draw_name(self, frame, obj):
        (left, top, right, bottom) = obj.bbox
        label = f'{obj.name}({obj.track_id})'
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, MxColors.Blue, 2)

    def draw_objects(self, frame, tracked_objects):
        for obj in tracked_objects:
            (left, top, right, bottom) = obj.bbox

            label = f'{obj.name}({obj.track_id})'
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, MxColors.Blue, 2)
            if self.distance_checkbox.isChecked():
                for i, (name, distance) in enumerate(obj.distances):
                    if i == 3:
                        break
                    label = f'{name}: {distance:.1f}'
                    cv2.putText(frame, label, (left + 10, top + 10 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            if self.bbox_checkbox.isChecked(): 
                cv2.rectangle(frame, (left, top), (right, bottom), MxColors.Blue, 2)

            # Draw keypoints if checkbox is checked
            if self.keypoints_checkbox.isChecked():
                for (x, y) in obj.keypoints:
                    cv2.circle(frame, (x, y), 5, MxColors.LightBlue, -1)

            # Draw semi-transparent rectangle if mouse is inside bounding box
            if self.mouse_position:
                mouse_x, mouse_y = self.mouse_position
                if left <= mouse_x <= right and top <= mouse_y <= bottom:
                    overlay = frame.copy()
                    alpha = 0.5  # Transparency factor
                    cv2.rectangle(overlay, (left, top), (right, bottom), MxColors.Blue, -1)
                    # Apply the overlay
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def overlay_icon(self, frame):
        x, y = frame.shape[1]- self.logo.shape[1] - 10, 10
        h, w = self.logo.shape[:2]
        if self.logo.shape[2] == 4:  # Handle alpha channel
            alpha_s = self.logo[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(3):
                frame[y:y+h, x:x+w, c] = (
                    alpha_s * self.logo[:, :, c] +
                    alpha_l * frame[y:y+h, x:x+w, c]
                )
        else:
            frame[y:y+h, x:x+w] = self.logo
        return frame

    def draw(self, frame):
        self.framerate.update()

        frame = np.copy(frame)
        frame = self.overlay_icon(frame)

        # Draw face recognitions
        tracked_objects = self.face_tracker.get_activated_tracker_objects()
        frame = self.draw_objects(frame, tracked_objects)

        self.frame_ready.emit(frame)
