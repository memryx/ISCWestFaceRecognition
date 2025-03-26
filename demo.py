import os
import sys
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
from modules.tracker import FaceTracker

class ConfigPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        #self.bbox_checkbox = QCheckBox("Draw Boxes")
        #self.bbox_checkbox.setChecked(True)
        layout.addWidget(self.parent().compositor.bbox_checkbox)
        layout.addWidget(self.parent().compositor.keypoints_checkbox)
        layout.addWidget(self.parent().compositor.distance_checkbox)

        self.framerate_checkbox = QCheckBox("Print framerates")
        self.framerate_checkbox.setChecked(False)
        layout.addWidget(self.framerate_checkbox)

# Viewer for video frames  
class Demo(QMainWindow):
    def __init__(self, video_path='/dev/video0', video_config=None):
        super().__init__()
        self.setWindowTitle("Video Viewer")

        # Create and configure the video display label
        self.viewer = FrameViewer()

        # Set up video-related attributes
        self.capture_thread = CaptureThread(video_path, 
                                            video_config)

        self.face_database = FaceDatabase()
        self.database_viewer = DatabaseViewerWidget(self.face_database)


        self.tracker = FaceTracker(self.face_database)
        self.compositor = Compositor(self.tracker)

        self.config_panel = ConfigPanel(self)

        # Connections Connect compositor's signal to the viewer's update slot
        self.capture_thread.frame_ready.connect(self.compositor.draw)
        self.capture_thread.frame_ready.connect(self.tracker.detect)

        self.compositor.frame_ready.connect(self.viewer.update_frame)
        self.viewer.mouse_move.connect(self.compositor.update_mouse_pos)
        self.viewer.mouse_click.connect(self.handle_viewer_mouse_click)

        # Layout the widgets
        self.setup_layout()

        # Start the threads
        self.tracker.start()
        self.capture_thread.start()

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
        self.control_layout.addWidget(self.config_panel)

        # Database loader
        self.control_layout.addWidget(self.database_viewer)

        # Video viewer widget
        self.splitter.addWidget(self.viewer)
        self.splitter.setStretchFactor(1, 1)

    def handle_viewer_mouse_click(self, mouse_pos):
        """Click on the viewer; check if its within a face and save the profile."""
        if mouse_pos is None:
            return
            
        tracker_frame = np.copy(self.tracker.current_frame.image)
        tracker_objects = self.tracker.get_activated_tracker_objects()

        # Iterate over each face to check if the click is inside any bounding box
        found = False 
        mouse_x, mouse_y = mouse_pos
        for obj in tracker_objects:
            (left, top, right, bottom) = obj.bbox
            if left <= mouse_x <= right and top <= mouse_y <= bottom:
                width = right - left
                height = bottom - top

                # Make the bounding box square with a 10px margin on all sides
                margin = 10
                bbox_size = max(width, height) + 2 * margin
                center_x, center_y = left + width // 2, top + height // 2
                x_start = max(0, center_x - bbox_size // 2)
                x_end = min(tracker_frame.shape[1], center_x + bbox_size // 2)
                y_start = max(0, center_y - bbox_size // 2)
                y_end = min(tracker_frame.shape[0], center_y + bbox_size // 2)

                # Crop the frame
                cropped_frame = tracker_frame[y_start:y_end, x_start:x_end]

                # Save the cropped image as a jpg file in the selected directory
                profile_path = self.database_viewer.get_selected_directory()
                if not profile_path:
                    if obj.name == 'Unknown':
                        new_profile = self.database_viewer.add_profile()
                        profile_path = os.path.join(self.database_viewer.db_path, new_profile)
                    else:
                        profile_path = os.path.join(self.database_viewer.db_path, obj.name)

                if os.path.exists(profile_path):
                    i = 0
                    while os.path.exists(os.path.join(profile_path, f"{i}.jpg")):
                        i += 1
                    filename = os.path.join(profile_path, f"{i}.jpg")

                    print(f'Saving image to {filename}')
                    cv2.imwrite(filename, cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR))
                    self.database_viewer.load_profiles()
                    #face_database.add_to_database(cropped_frame, filename)
                    self.face_database.add_to_database(obj.embedding, filename)
                found = True
                break

        pass #TODO: Impelment
        # 1. get face locations
        # 2. check if mouse over face
        # 3. save image and embedding to databse

    def poll_framerates(self):
        if self.config_panel.framerate_checkbox.isChecked():
            print(f'capture: {self.capture_thread.framerate.get():.1f}')
            print(f'composite: {self.compositor.framerate.get():.1f}')
            print(f't.dt {self.tracker.detection_thread.framerate.get():.1f}')
            print(f't.rt {self.tracker.recognition_thread.framerate.get():.1f}')

    def closeEvent(self, event):
        # Stop threads and release video capture on close
        self.capture_thread.stop()
        self.capture_thread.wait()
        self.tracker.stop()
        self.fps_timer.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_path = "/dev/video0"  # Replace with your video file path
    #video_path = "/home/jake/Videos/lunch.mp4"
    #video_path = 'assets/photos/joey.jpg'
    player = Demo(video_path, VIDEO_CONFIG['1080p'])
    player.resize(1200, 800)
    player.show()
    sys.exit(app.exec())
