
import sys
import queue
import cv2
import numpy as np

from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QWidget,
                               QVBoxLayout, QLineEdit, QPushButton,
                               QHBoxLayout, QSplitter, QCheckBox, QFrame,
                               QTreeWidget, QTreeWidgetItem, QInputDialog,
                               QMessageBox, QFileDialog)
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QKeyEvent
from PySide6.QtCore import QTimer, Qt, QThread, Signal, QMutex


import time
from pathlib import Path

from modules.capture import CaptureThread, VIDEO_CONFIG
from modules.compositor import Compositor
from modules.viewer import FrameViewer
from modules.database import FaceDatabase, DatabaseViewerWidget


from modules.MXFace2 import MXFace, MockMXFace
from modules.tracker import FaceTracker

# Viewer for video frames  
class Demo(QMainWindow):
    def __init__(self, video_path='/dev/video0', video_config=None):
        super().__init__()
        self.setWindowTitle("Video Viewer")

        # Create and configure the video display label
        self.viewer_widget = FrameViewer()

        # Set up video-related attributes
        self.mxface = MockMXFace(Path('assets/models'))
        self.capture_thread = CaptureThread(video_path, 
                                            self.mxface, 
                                            video_config)

        self.face_database = FaceDatabase()
        self.database_viewer_widget = DatabaseViewerWidget(self.face_database)


        self.tracker = FaceTracker(self.mxface, self.face_database)
        self.compositor = Compositor(self.tracker, 16)

        # Connect compositor's signal to the viewer's update slot
        self.compositor.frame_ready.connect(self.viewer_widget.update_frame)

        # Layout the widgets
        self.setup_layout()

        # Start the threads
        self.tracker.start()
        self.capture_thread.start()
        self.compositor.start()

        # For frame rate calculation
        self.timestamps = [0] * 30

        self.fps_timer = QTimer(self)
        self.fps_timer.setInterval(1000)
        self.fps_timer.timeout.connect(self.poll_framerates)
        self.fps_timer.start()

    def setup_layout(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setMinimumSize(300, 200)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Splitter to separate control panel and video viewer
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # Control panel widget
        self.control_panel = QWidget()
        self.control_panel.setFixedWidth(300)
        self.control_layout = QVBoxLayout(self.control_panel)
        self.splitter.addWidget(self.control_panel)

        # Video path input and load button
        self.video_path_input = QLineEdit(self)
        self.video_path_input.setPlaceholderText("Enter video file path...")
        self.load_video_button = QPushButton("Load Video Path", self)
        #self.load_video_button.clicked.connect(self.update_video_path)
        self.control_layout.addWidget(self.video_path_input)
        self.control_layout.addWidget(self.load_video_button)

        # Config panel with checkboxes
        self.config_panel = QFrame()
        self.config_layout = QVBoxLayout(self.config_panel)
        self.keypoints_checkbox = QCheckBox("Draw Keypoints", self)
        self.keypoints_checkbox.setChecked(False)
        self.bbox_checkbox = QCheckBox("Draw Boxes", self)
        self.bbox_checkbox.setChecked(True)
        self.conf_checkbox = QCheckBox("Show Distances", self)
        self.conf_checkbox.setChecked(False)
        self.config_layout.addWidget(self.keypoints_checkbox)
        self.config_layout.addWidget(self.bbox_checkbox)
        self.config_layout.addWidget(self.conf_checkbox)
        self.control_layout.addWidget(self.config_panel)

        # Database loader
        self.control_layout.addWidget(self.database_viewer_widget)

        # Video viewer widget
        #self.video_layout = QVBoxLayout(self.viewer)
        self.splitter.addWidget(self.viewer_widget)
        self.splitter.setStretchFactor(1, 1)


    def poll_framerates(self):
        print(f'capture: {self.capture_thread.framerate.get():.1f}')
        print(f'composite: {self.compositor.framerate.get():.1f}')
        print(f't.dt {self.tracker.detection_thread.framerate.get():.1f}')
        print(f't.rt {self.tracker.recognition_thread.framerate.get():.1f}')

    def closeEvent(self, event):
        # Stop threads and release video capture on close
        self.capture_thread.stop()
        self.capture_thread.wait()
        self.compositor.stop()
        self.tracker.stop()
        self.mxface.stop()
        self.fps_timer.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    #video_path = "/dev/video2"  # Replace with your video file path
    #video_path = "/home/jake/Videos/lunch.mp4"
    video_path = 'assets/photos/one_face.jpg'
    player = Demo(video_path, VIDEO_CONFIG['1080p'])
    player.resize(1200, 800)
    player.show()
    sys.exit(app.exec())
