from PySide6.QtCore import QTimer, QThread, Signal
from modules.bytetracker import BYTETracker
import queue
from dataclasses import dataclass
import numpy as np
from modules.MXFace2 import MXFace, AnnotatedFrame
import cv2

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

class FaceTracker(QThread):
    frame_ready = Signal()

    def __init__(self, mxface: MXFace):
        super().__init__()
        self.tracker = BYTETracker()
        self.mxface = mxface
        self.stop_threads = False
        self.tracked_objects = []
        self.tracker_dict = {}
        self.current_frame = AnnotatedFrame(np.zeros([10,10,3]))

        self.composite_queue = queue.Queue(maxsize=1)

    def _update_detections(self):
        try:
            annotated_frame = self.mxface.detect_get(timeout=0.033)  # Timeout to allow shutdown
            self.current_frame = annotated_frame
        except queue.Empty:
            return 

        for track_id, tracked_object in self.tracker_dict.items():
            tracked_object.activated = False

        if self.current_frame.num_detected_faces == 0:
            return

        dets = []
        for bbox, score in zip(annotated_frame.boxes, annotated_frame.scores):
            x, y, w, h = bbox
            dets.append(np.array([x, y, x+w, y+h, score, 0]))

        dets = np.array(dets, dtype=np.float32)

        for tracklet in self.tracker.update(dets, None):
            x1, y1, x2, y2, track_id, _, _ = tracklet.astype(int)
            if track_id in self.tracker_dict:
                self.tracker_dict[track_id].bbox = (x1, y1, x2, y2)
                self.tracker_dict[track_id].activated = True
            else:
                self.tracker_dict[track_id] = TrackedObject((x1, y1, x2, y2), track_id, "Unknown")

    def _recognize_faces(self):
        for tracked_object in self.tracker_dict.values():
            if tracked_object.name == "Unknown":
                face = self._extract_face(self.current_frame.image, tracked_object.bbox)
                self.mxface.recognize_put(face)

    def _extract_face(self, image: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = xyxy 

        # Get the original image dimensions
        orig_h, orig_w, _ = image.shape

        # Compute the new top-left and bottom-right corners in the original image
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), orig_w)
        y2 = min(int(y2), orig_h)

        # Extract the face from the original image using the adjusted bounding box
        face = image[y1:y2, x1:x2]

        #if self.do_eye_alignment:
        #    face, bbox = self._align_eyes(face, detected_face)

        return face

    def _align_eyes(self, image: np.ndarray, detected_face):
        # write the code here

        right_eye = detected_face.keypoints[0]
        left_eye = detected_face.keypoints[1]
        
        # Compute the angle between the eyes.
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # To align the eyes horizontally, rotate by the negative of this angle.
        rotation_angle = angle

        # Compute the rotation matrix.
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        # Rotate the image.
        rotated_image = cv2.warpAffine(image, M, (w, h))
        
        # Assume the first bounding box is used; expected format: (x, y, w, h)
        x, y, bw, bh = detected_face.bbox
        # Define the four corners of the bounding box.
        corners = np.array([
            [x, y],
            [x + bw, y],
            [x, y + bh],
            [x + bw, y + bh]
        ], dtype=np.float32).reshape(-1, 1, 2)
        # Transform the corners using the same affine matrix.
        transformed = cv2.transform(corners, M).reshape(-1, 2)
        # Compute the minimal axis-aligned bounding box that encloses the transformed corners.
        x_min = int(np.min(transformed[:, 0]))
        y_min = int(np.min(transformed[:, 1]))
        x_max = int(np.max(transformed[:, 0]))
        y_max = int(np.max(transformed[:, 1]))
        new_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        return rotated_image, new_bbox

    def run(self):
        """Main logic for the face tracker. 
        1. Pull frames from the MXFace recognizer. Update tracker with detections.
        2. For new tracked objects (or objects that need a refresh), send them to MXFace recognizer.
        """
        
        while not self.stop_threads:
            self._update_detections()

            try:
                activated_objects = [obj for obj in self.tracker_dict.values() if obj.activated]
                self.composite_queue.put_nowait(CompositeFrame(self.current_frame.image, activated_objects))
            except queue.Full:
                pass

            #self._recognize_faces()

    def stop(self):
        self.stop_threads = True
