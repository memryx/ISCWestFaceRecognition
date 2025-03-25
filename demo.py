
import sys
import queue
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, QThread, Signal
import time
from pathlib import Path

from modules.capture import CaptureThread, VIDEO_CONFIG
from modules.compositor import Compositor
from modules.viewer import FrameViewer

from modules.MXFace2 import MXFace
from modules.tracker2 import FaceTracker

# Viewer for video frames  
class Demo(QWidget):
    def __init__(self, video_path='/dev/video0', video_config=None):
        super().__init__()
        self.setWindowTitle("Video Viewer")

        # Create and configure the video display label
        self.viewer = FrameViewer()

        # Set a layout and add the video label to the widget
        layout = QVBoxLayout(self)
        layout.addWidget(self.viewer)

        # Set up video-related attributes
        #self.frame_queue = queue.Queue(maxsize=6)
        self.mxface = MXFace(Path('assets/models'))
        self.capture_thread = CaptureThread(video_path, 
                                            self.mxface, 
                                            video_config)

        #self.compositor = Compositor(self.frame_queue, 33)
        self.tracker = FaceTracker(self.mxface)
        self.compositor = Compositor(self.tracker, 40)

        # Connect compositor's signal to the viewer's update slot
        self.compositor.frame_ready.connect(self.viewer.update_frame)

        # Start the threads
        self.tracker.start()
        self.capture_thread.start()
        self.compositor.start()

        # For frame rate calculation
        self.timestamps = [0] * 30

    def closeEvent(self, event):
        # Stop threads and release video capture on close
        self.capture_thread.stop()
        self.capture_thread.wait()

        self.compositor.stop()
        self.mxface.stop()
        self.tracker.stop()

        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_path = "/dev/video2"  # Replace with your video file path
    #video_path = "/home/jake/Videos/lunch.mp4"
    player = Demo(video_path, VIDEO_CONFIG['2k'])
    player.resize(1200, 800)
    player.show()
    sys.exit(app.exec())
