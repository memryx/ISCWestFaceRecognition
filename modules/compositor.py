import queue
import numpy as np
import cv2
from PySide6.QtCore import QTimer, QObject, Signal
from PySide6.QtWidgets import QCheckBox

from .utils import Framerate, SliderWithLabel


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

        # Instantiate SliderWithLabel widgets:
        self.label_scale_slider = SliderWithLabel("Font Scale:", 
                                                  minimum=50, 
                                                  maximum=300, 
                                                  initial=85, 
                                                  step=5, 
                                                  multiplier=0.01)
        self.label_thickness_slider = SliderWithLabel("Font Thickness:", 
                                                      minimum=1,
                                                      maximum=10, 
                                                      initial=2,
                                                      step=1, 
                                                      multiplier=1)
        self.line_thickness_slider = SliderWithLabel("Line Thickness:", 
                                                      minimum=1,
                                                      maximum=10, 
                                                      initial=2,
                                                      step=1, 
                                                      multiplier=1)


        # Load icons and resize
        self.load_icons()

    def load_icons(self):
        icon = cv2.imread("assets/logo.png", cv2.IMREAD_UNCHANGED)
        icon = cv2.resize(icon, (100, 100))
        self.logo = cv2.cvtColor(icon, cv2.COLOR_RGBA2BGRA)

    def update_mouse_pos(self, pos):
        self.mouse_position = pos

    def draw_name_plain(self, frame, obj):
        (left, top, right, bottom) = obj.bbox
        label = f'{obj.name}({obj.track_id})'
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, MxColors.Blue, 2)

    def draw_name(self, frame, obj):
        (left, top, right, bottom) = obj.bbox
        label = f'{obj.name}({obj.track_id})'
        h, w = frame.shape[:2]

        # Get dynamic font parameters from the slider widgets.
        font_scale = self.label_scale_slider.value()  # Float value (e.g. 0.85)
        thickness = int(self.label_thickness_slider.value())

        # Compute center x coordinate of detection box.
        cx = left + (right - left) // 2

        # Parameters for the label line.
        diag_length = (right - left) // 2   # Length of diagonal segment.
        margin = 10        # Extra length added to horizontal segment.
        line_thickness = int(self.line_thickness_slider.value())

        # Get text size for label.
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        horiz_length = text_width + margin

        # Default horizontal direction (to the right).
        dx = 1

        # Try upward placement by default.
        upward = True
        start = (cx, top)
        diag_end = (cx + dx * diag_length, top - diag_length)
        # Upward text position: drawn above the horizontal segment.
        upward_text_y = diag_end[1] - 10

        # If the label would be drawn above the image, switch to downward placement.
        if upward_text_y < 0:
            upward = False

        if upward:
            start = (cx, top)
            diag_end = (cx + dx * diag_length, top - diag_length)
            text_y = diag_end[1] - 10  # Draw text above the line.
        else:
            # Downward configuration: start at the bottom-center of the box.
            start = (cx, bottom)
            diag_end = (cx + dx * diag_length, bottom + diag_length)
            text_y = diag_end[1] + text_height + 10  # Draw text below the line.

        # Check if the horizontal segment (from diag_end) would extend outside the image.
        # If so, flip dx (i.e. have the line slant toward the left).
        if dx == 1 and (diag_end[0] + horiz_length > w):
            dx = -1
        elif dx == -1 and (diag_end[0] - horiz_length < 0):
            dx = 1

        # Recompute diag_end if dx changed.
        if upward:
            diag_end = (cx + dx * diag_length, top - diag_length)
        else:
            diag_end = (cx + dx * diag_length, bottom + diag_length)

        # Compute end of horizontal segment.
        horiz_end = (int(diag_end[0] + dx * horiz_length), diag_end[1])

        # Compute center of horizontal segment for centering the text.
        center_horiz = diag_end[0] + dx * (horiz_length / 2)
        text_x = int(center_horiz - text_width / 2)

        # Draw the label line in DarkBlue.
        cv2.line(frame, (cx, top) if upward else (cx, bottom), 
                 (int(diag_end[0]), int(diag_end[1])), MxColors.DarkBlue, thickness=line_thickness)
        cv2.line(frame, (int(diag_end[0]), int(diag_end[1])), 
                 (int(horiz_end[0]), int(horiz_end[1])), MxColors.DarkBlue, thickness=line_thickness)

        # Draw the label text in Blue.
        cv2.putText(frame, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, MxColors.Blue, thickness)

    def draw_objects(self, frame, tracked_objects):
        for obj in tracked_objects:
            (left, top, right, bottom) = obj.bbox

            self.draw_name(frame, obj)
            if self.distance_checkbox.isChecked():
                for i, (name, distance) in enumerate(obj.distances):
                    if i == 3:
                        break
                    label = f'{name}: {distance:.1f}'
                    cv2.putText(frame, label, (left + 10, top + 10 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            if self.bbox_checkbox.isChecked(): 
                line_thickness = int(self.line_thickness_slider.value())
                cv2.rectangle(frame, (left, top), (right, bottom), MxColors.Blue, line_thickness)

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
