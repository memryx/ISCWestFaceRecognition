from PySide6.QtCore import QThread, Signal
from modules.bytetracker import BYTETracker
import queue
from dataclasses import dataclass, field
import numpy as np
from modules.MXFace2 import MXFace, AnnotatedFrame
import cv2
import time
from .database import FaceDatabase
from .utils import Framerate
import threading  # Import the threading module for locks

@dataclass
class TrackedObject:
    bbox: tuple[int, int, int, int]
    keypoints: list[tuple[int, int]]
    track_id: int
    name: str = "Unknown"
    activated: bool = True
    last_recognition: float = 0.0
    distances: list[float] = field(default_factory=list)
    embedding: np.ndarray = field(default_factory=lambda: np.zeros([128]))


@dataclass
class CompositeFrame:
    image: np.ndarray
    tracked_objects: list


def compute_iou(boxA, boxB):
    # boxA and boxB are (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


class FaceTracker:
    """
    DetectionThread: continuously pulls detections from mxface.detect_get(),
    updates the tracker and pushes unknown faces (with track_id) to be recognized.
    RecognitionThread: continuously pulls recognition results from mxface.recognize_get()
    and updates the tracker_dict with the recognized name.
    """
    def __init__(self, mxface: MXFace, face_database: FaceDatabase):
        self.tracker = BYTETracker()
        self.mxface = mxface
        self.tracker_dict = {}  # Mapping from track_id to TrackedObject
        self.tracker_dict_lock = threading.Lock()  # Lock for tracker_dict
        self.current_frame = AnnotatedFrame(np.zeros([10, 10, 3]))
        self.composite_queue = queue.Queue(maxsize=1)
        self.database = face_database
        
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

    def detect(self, frame):
        try:
            self.mxface.detect_put(frame, block=False)
        except queue.Full:
            pass

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

    def get_tracker_dict_copy(self) -> dict:
        """Return a thread-safe shallow copy of tracker_dict."""
        with self.tracker_dict_lock:
            return dict(self.tracker_dict)


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
        self.refresh_interval = 1
        self.framerate = Framerate()

    def _update_detections(self):
        self.framerate.update()

        try:
            annotated_frame = self.face_tracker.mxface.detect_get(timeout=0.033)
            self.face_tracker.current_frame = annotated_frame
        except queue.Empty:
            return

        # Mark all current tracked objects as not active (protected by lock)
        with self.face_tracker.tracker_dict_lock:
            for tracked_object in self.face_tracker.tracker_dict.values():
                tracked_object.activated = False

        current_time = time.time()
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
            keypoints = self._get_keypoints((x1, y1, x2, y2), annotated_frame)

            with self.face_tracker.tracker_dict_lock:
                # For an existing track, update bbox and activate it.
                if track_id in self.face_tracker.tracker_dict:
                    tracked_obj = self.face_tracker.tracker_dict[track_id]
                    tracked_obj.bbox = (x1, y1, x2, y2)
                    tracked_obj.activated = True

                    # Refresh active track if refresh_interval elapsed.
                    if current_time - tracked_obj.last_recognition > self.refresh_interval:
                        #face = self.face_tracker._extract_face(annotated_frame.image, (x1, y1, x2, y2))
                        try:
                            #self.face_tracker.mxface.recognize_put((track_id, face), block=False)
                            self.face_tracker.mxface.recognize_put((track_id, annotated_frame.image, (x1, y1, x2, y2)), block=False)
                        except queue.Full:
                            pass
                else:
                    # New track: create a new tracked object and request recognition immediately.
                    new_obj = TrackedObject(
                        bbox=(x1, y1, x2, y2), 
                        keypoints=keypoints, 
                        track_id=track_id, 
                        last_recognition=current_time
                    )
                    self.face_tracker.tracker_dict[track_id] = new_obj
                    #face = self.face_tracker._extract_face(annotated_frame.image, (x1, y1, x2, y2))
                    try:
                        self.face_tracker.mxface.recognize_put((track_id, annotated_frame.image, (x1, y1, x2, y2)), block=False)
                    except queue.Full:
                        pass

        # Push composite frame (image and currently activated objects) to the queue
        with self.face_tracker.tracker_dict_lock:
            activated_objects = [obj for obj in self.face_tracker.tracker_dict.values() if obj.activated]

    def run(self):
        while not self.stop_threads:
            self._update_detections()

    def stop(self):
        self.stop_threads = True

    def _get_keypoints(self, track_box, annotated_frame):
        """Re-associate the tracked box with the detected box to extract keypoints."""
        best_iou = 0
        best_idx = None

        # Loop over detections from annotated_frame
        for idx, det_box in enumerate(annotated_frame.boxes):
            # Convert detection box from (x, y, w, h) to (x1, y1, x2, y2)
            det_box_converted = (det_box[0], det_box[1], det_box[0] + det_box[2], det_box[1] + det_box[3])
            iou = compute_iou(track_box, det_box_converted)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        return annotated_frame.keypoints[best_idx]


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
        self.framerate = Framerate()

    def run(self):
        while not self.stop_threads:
            self.framerate.update()
            self._run()

    def _run(self):
        try:
            # Expect recognition results as (track_id, embedding)
            track_id, embedding = self.face_tracker.mxface.recognize_get(timeout=0.1)
        except queue.Empty:
            return

        with self.face_tracker.tracker_dict_lock:
            if track_id not in self.face_tracker.tracker_dict:
                return

            name, distances = self.face_tracker.database.find(embedding)
            tracked_obj = self.face_tracker.tracker_dict[track_id]
            tracked_obj.embedding = embedding
            tracked_obj.name = name
            tracked_obj.distances = distances
            tracked_obj.last_recognition = time.time()

    def stop(self):
        self.stop_threads = True
