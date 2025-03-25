
import sys
import queue
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QHBoxLayout, QWidget
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, QThread, Signal
import time
from pathlib import Path

from modules.capture import CaptureThread, VIDEO_CONFIG
from modules.compositor import Compositor
from modules.viewer import FrameViewer
from modules.database import FaceDatabase, DatabaseViewerWidget


from modules.MXFace2 import MXFace
from modules.tracker import FaceTracker

# Viewer for video frames  
class Demo(QWidget):
    def __init__(self, video_path='/dev/video0', video_config=None):
        super().__init__()
        self.setWindowTitle("Video Viewer")

        # Create and configure the video display label
        self.viewer = FrameViewer()

        # Set up video-related attributes
        self.mxface = MXFace(Path('assets/models'))
        self.capture_thread = CaptureThread(video_path, 
                                            self.mxface, 
                                            video_config)

        self.face_database = FaceDatabase()
        self.database_viewer_widget = DatabaseViewerWidget(self.face_database)


        self.tracker = FaceTracker(self.mxface, self.face_database)
        self.compositor = Compositor(self.tracker, 16)

        # Connect compositor's signal to the viewer's update slot
        self.compositor.frame_ready.connect(self.viewer.update_frame)

        # Layout the widgets
        layout = QHBoxLayout(self)
        layout.addWidget(self.database_viewer_widget)
        layout.addWidget(self.viewer)

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
