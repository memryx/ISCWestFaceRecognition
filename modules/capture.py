import sys
import queue
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, QThread, Signal
import time

from .utils import Framerate

def is_image(source):
    return source.endswith(('.jpg', '.jpeg', '.png', '.bmp'))

def is_video(source):
    return source.endswith(('.mp4', '.webm'))

class VideoConfig:
    def __init__(self, width=3840, height=2160, fourcc='MJPG', fps=30):
        self.config = {
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*fourcc),
            cv2.CAP_PROP_FPS: fps,
        }

    def set(self, cap: cv2.VideoCapture):
        for k, v in self.config.items():
            cap.set(k, v)

VIDEO_CONFIG = {
    '4k': VideoConfig(3840, 2160), 
    '2k': VideoConfig(2560, 1440),
    '1080p': VideoConfig(1920,1080, fps=60),
    '720p': VideoConfig(1280, 720)
}

# Thread for reading video frames
class CaptureThread(QThread):

    def __init__(self, video_source, mxface, video_config=None):
        super().__init__()
        self.video_config = video_config
        self.video_source = video_source
        self.mxface = mxface
        self.stop_threads = False
        self.pause = False
        self.cur_frame = None

        self.framerate = Framerate()

    def _read_image(self):
        """Read image file and simulate video stream"""
        frame = cv2.imread(self.video_source)
        if frame is None:
            print("Failed to load image.")
            return

        while not self.stop_threads:
            self.framerate.update()

            #if not self.mxface.full():
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                self.mxface.detect_put(np.array(rgb_frame), block=False)
            except queue.Full:
                pass

            #time.sleep(1 / 30)  # Simulate 30fps for static image

    def _read_video(self):
        """Read video file or stream"""

        # Handle video case
        cap = cv2.VideoCapture(self.video_source)
        if self.video_config is not None: 
            self.video_config.set(cap)

        while not self.stop_threads:
            if self.pause:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                if is_video(self.video_source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("Stream ended or failed to grab frame.")
                    break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.cur_frame = np.array(rgb_frame)

            # Simulating real-time video stream (30fps)
            if is_video(self.video_source):
                start = time.time()
                self.mxface.detect_put(np.array(rgb_frame), block=False)
                dt = time.time() - start
                time.sleep(max(0.033-dt, 0))  
            else:
                try:
                    self.mxface.detect_put(self.cur_frame, timeout=0.033)
                except queue.Full:
                    print('Dropped Frame')

        cap.release()

    def run(self):
        """Read video frames and emit signal"""
        if is_image(self.video_source):
            self._read_image()
        else:
            self._read_video()

    def toggle_play(self):
        self.pause = not self.pause

    def stop(self):
        print("Shutting down CaptureThread")
        self.stop_threads = True

