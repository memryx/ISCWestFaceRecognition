from PySide6.QtCore import QThread, Signal
from modules.bytetracker import BYTETracker
import queue
from dataclasses import dataclass
import numpy as np
from modules.MXFace2 import MXFace, AnnotatedFrame
import cv2
import time
from .database import FaceDatabase


@dataclass
class TrackedObject:
    bbox: tuple[int, int, int, int]
    track_id: int
    name: str
    activated: bool = True


@dataclass
class CompositeFrame:
    image: np.ndarray
    tracked_objects: list


class FaceTracker:
    """
    FaceTracker now manages two threads:
      - DetectionThread: continuously pulls detections from mxface.detect_get(),
        updates the tracker and pushes unknown faces (with track_id) to be recognized.
      - RecognitionThread: continuously pulls recognition results from mxface.recognize_get()
        and updates the tracker_dict with the recognized name.
    """
    def __init__(self, mxface: MXFace):
        self.tracker = BYTETracker()
        self.mxface = mxface
        self.tracker_dict = {}  # Mapping from track_id to TrackedObject
        self.current_frame = AnnotatedFrame(np.zeros([10, 10, 3]))
        self.composite_queue = queue.Queue(maxsize=1)
        self.database = FaceDatabase() 
        self.database.load_database_embeddings('assets/db')
        
        # Create worker threads for detection and recognition
        self.detection_thread = DetectionThread(self)
        self.recognition_thread = RecognitionThread(self)

    def start(self):
        self.detection_thread.start()
        self.recognition_thread.start()

    def stop(self):
        self.detection_thread.stop()
        self.recognition_thread.stop()
        self.detection_thread.wait()
        self.recognition_thread.wait()

    def _extract_face(self, image: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = xyxy
        orig_h, orig_w, _ = image.shape
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), orig_w)
        y2 = min(int(y2), orig_h)
        face = image[y1:y2, x1:x2]
        return face

    def _align_eyes(self, image: np.ndarray, detected_face):
        right_eye = detected_face.keypoints[0]
        left_eye = detected_face.keypoints[1]
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        rotation_angle = angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h))
        x, y, bw, bh = detected_face.bbox
        corners = np.array([
            [x, y],
            [x + bw, y],
            [x, y + bh],
            [x + bw, y + bh]
        ], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.transform(corners, M).reshape(-1, 2)
        x_min = int(np.min(transformed[:, 0]))
        y_min = int(np.min(transformed[:, 1]))
        x_max = int(np.max(transformed[:, 0]))
        y_max = int(np.max(transformed[:, 1]))
        new_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        return rotated_image, new_bbox


class DetectionThread(QThread):
    """
    This thread continuously calls mxface.detect_get() to get new annotated frames.
    It then updates the tracker using the detected bounding boxes, marks existing objects as inactive,
    and for each new unknown track, extracts the face and pushes a tuple (track_id, face)
    to mxface.recognize_put() for further processing.
    It also pushes a CompositeFrame (current image and activated objects) into a composite_queue.
    """
    def __init__(self, face_tracker):
        super().__init__()
        self.face_tracker = face_tracker
        self.stop_threads = False

    def _update_detections(self):
        try:
            annotated_frame = self.face_tracker.mxface.detect_get(timeout=0.033)
            self.face_tracker.current_frame = annotated_frame
        except queue.Empty:
            return

        # Mark all current tracked objects as not active
        for tracked_object in self.face_tracker.tracker_dict.values():
            tracked_object.activated = False

        if annotated_frame.num_detected_faces == 0:
            return

        # Build detections array expected by BYTETracker
        dets = []
        for bbox, score in zip(annotated_frame.boxes, annotated_frame.scores):
            x, y, w, h = bbox
            dets.append(np.array([x, y, x+w, y+h, score, 0]))
        dets = np.array(dets, dtype=np.float32)

        # Update tracker with the new detections
        for tracklet in self.face_tracker.tracker.update(dets, None):
            x1, y1, x2, y2, track_id, _, _ = tracklet.astype(int)
            if track_id in self.face_tracker.tracker_dict:
                self.face_tracker.tracker_dict[track_id].bbox = (x1, y1, x2, y2)
                self.face_tracker.tracker_dict[track_id].activated = True
            else:
                # New track detected; create a new tracked object
                self.face_tracker.tracker_dict[track_id] = TrackedObject((x1, y1, x2, y2), track_id, "Unknown")
                # Extract the face from the current frame and push it for recognition
                face = self.face_tracker._extract_face(annotated_frame.image, (x1, y1, x2, y2))
                try:
                    self.face_tracker.mxface.recognize_put((track_id, face), block=False)
                except queue.Full:
                    pass


    def run(self):
        while not self.stop_threads:
            self._update_detections()

            # Push composite frame (image and currently activated objects) to the queue
            activated_objects = [obj for obj in self.face_tracker.tracker_dict.values() if obj.activated]

            try:
                self.face_tracker.composite_queue.put_nowait(
                    CompositeFrame(self.face_tracker.current_frame.image, 
                                   activated_objects)
                )
            except queue.Full:
                pass

    def stop(self):
        self.stop_threads = True


class RecognitionThread(QThread):
    """
    This thread continuously calls mxface.recognize_get() to retrieve recognition results.
    Each result is expected to be a tuple (track_id, recognized_name). The thread then updates
    the corresponding tracked object with the new recognized name.
    """
    def __init__(self, face_tracker):
        super().__init__()
        self.face_tracker = face_tracker
        self.stop_threads = False

    def run(self):
        while not self.stop_threads:
            try:
                # Expect recognition results as (track_id, recognized_name)
                track_id, embedding = self.face_tracker.mxface.recognize_get(timeout=0.1)
                if track_id in self.face_tracker.tracker_dict:
                    name, distances = self.face_tracker.database.find(embedding)
                    self.face_tracker.tracker_dict[track_id].name = name
                    self.face_tracker.tracker_dict[track_id].distance = distances[0]
            except queue.Empty:
                continue

    def stop(self):
        self.stop_threads = True
