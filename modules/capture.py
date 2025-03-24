import sys
import queue
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, QThread, Signal
import time

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

# Thread for reading and rendering the final video frames
class CompositeThread(QThread):
    frame_ready = Signal(np.ndarray)

    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.stop_threads = False

    def run(self):
        while not self.stop_threads or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=1)  # Timeout to allow shutdown
                self.frame_ready.emit(frame)
            except queue.Empty:
                #print('Unable to get.. skipping')
                continue
        print("Video display stopped.")

    def stop(self):
        self.stop_threads = True

# Thread for reading video frames
class CaptureThread(QThread):

    def __init__(self, video_source, frame_queue, video_config=None):
        super().__init__()
        self.video_config = video_config
        self.video_source = video_source
        self.frame_queue = frame_queue
        self.stop_threads = False
        self.pause = False
        self.cur_frame = None

    def _read_image(self):
        """Read image file and simulate video stream"""
        frame = cv2.imread(self.video_source)
        if frame is None:
            print("Failed to load image.")
            return
        while not self.stop_threads:
            if not self.frame_queue.full():
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_queue.put(np.array(rgb_frame))
            time.sleep(1 / 30)  # Simulate 30fps for static image

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
                self.frame_queue.put(np.array(rgb_frame))
                dt = time.time() - start
                time.sleep(max(0.033-dt, 0))  
            else:
                try:
                    self.frame_queue.put(self.cur_frame, timeout=0.033)
                except queue.Full:
                    print('Dropped Frame')

        cap.release()
        print("Video reader stopped.")

    def run(self):
        """Read video frames and emit signal"""
        if is_image(self.video_source):
            self._read_image()
        else:
            self._read_video()

    def toggle_play(self):
        self.pause = not self.pause

    def stop(self):
        self.stop_threads = True

# Viewer for video frames  
class Viewer(QWidget):
    def __init__(self, video_path='/dev/video0'):
        super().__init__()
        self.setWindowTitle("Video Viewer")

        # Create and configure the video display label
        self.video_label = QLabel(self)
        self.video_label.setMouseTracking(True)

        # Set a layout and add the video label to the widget
        layout = QVBoxLayout(self)
        layout.addWidget(self.video_label)

        # Set up video-related attributes
        self.frame_queue = queue.Queue(maxsize=6)
        self.video_reader_thread = CaptureThread(video_path, self.frame_queue)
        self.video_display_thread = CompositeThread(self.frame_queue)

        # Connect only the display thread signal to the update_frame slot.
        self.video_display_thread.frame_ready.connect(self.update_frame)

        # Start the threads
        self.video_reader_thread.start()
        self.video_display_thread.start()

        # For frame rate calculation
        self.timestamps = [0] * 30

    def update_frame(self, frame):
        cur_time = time.time()
        self.timestamps.append(int(cur_time))
        self.timestamps.pop(0)
        dt = np.average([self.timestamps[i + 1] - self.timestamps[i] for i in range(len(self.timestamps) - 1)])

        self.current_frame = frame

        # Draw bounding boxes and labels for each face in the frame
        #frame = frame.copy()

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

        # Get image information
        height, width, channels = frame.shape
        bytes_per_line = channels * width

        # Create QImage and display it
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))


    def closeEvent(self, event):
        # Stop threads and release video capture on close
        self.video_reader_thread.stop()
        self.video_reader_thread.wait()
        self.video_display_thread.stop()
        self.video_display_thread.wait()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_path = "/dev/video0"  # Replace with your video file path
    player = Viewer(video_path)
    player.resize(640, 480)
    player.show()
    sys.exit(app.exec())
