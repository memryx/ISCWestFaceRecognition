
import time
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QWidget,
                               QVBoxLayout, QLineEdit, QPushButton,
                               QHBoxLayout, QSplitter, QCheckBox, QFrame)
from PySide6.QtGui import QImage, QPixmap, QMouseEvent

# Viewer for video frames  
class FrameViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Viewer")

        # Create and configure the video display label
        self.video_label = QLabel(self)
        self.video_label.setMouseTracking(True)

        # Set a layout and add the video label to the widget
        layout = QVBoxLayout(self)
        layout.addWidget(self.video_label)

        # For frame rate calculation
        self.timestamps = [0] * 30

    def update_frame(self, frame):
        cur_time = time.time()
        self.timestamps.append(int(cur_time))
        self.timestamps.pop(0)

        self.current_frame = frame

        # Resize the frame to fit the available area for the video viewer while preserving the aspect ratio
        video_label_width = self.video_label.width()
        video_label_height = self.video_label.height()
        frame_height, frame_width, _ = frame.shape

        aspect_ratio = frame_width / frame_height
        if video_label_width / video_label_height > aspect_ratio:
            #new_height = min(video_label_height, frame_height)
            new_height = video_label_height
            new_width = int(aspect_ratio * new_height)
        else:
            #new_width = min(video_label_width, frame_width)
            new_width = video_label_width
            new_height = int(new_width / aspect_ratio)

        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        self.video_label.setMinimumSize(1, 1)

        ## Get image information
        height, width, channels = frame.shape
        bytes_per_line = channels * width

        # Create QImage and display it
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

