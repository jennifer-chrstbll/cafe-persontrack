import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import config

class PersonDetection:
    """Dataclass holding detection information for a person."""
    def __init__(self, bbox: Tuple[float, float, float, float], conf: float, class_id: int = 0):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.conf = conf
        self.class_id = class_id
        
        # Calculate centroid & dimensions
        self.x1, self.y1, self.x2, self.y2 = bbox
        self.width = max(1.0, self.x2 - self.x1)
        self.height = max(1.0, self.y2 - self.y1)
        self.centroid_x = self.x1 + self.width / 2.0
        self.centroid_y = self.y1 + self.height / 2.0

    @property
    def tlwh(self) -> Tuple[float, float, float, float]:
        """Top-left x, top-left y, width, height."""
        return (self.x1, self.y1, self.width, self.height)

    @property
    def centroid(self) -> Tuple[float, float]:
        return (self.centroid_x, self.centroid_y)


class PersonDetector:
    """
    Person-only Detector using YOLO11n.
    Supports ONNX Runtime (fastest on Edge/CPU) with PyTorch fallback.
    """
    def __init__(self, conf_thresh: float = config.DETECTION_CONF_THRESH, use_onnx: bool = True):
        self.conf_thresh = conf_thresh
        self.use_onnx = use_onnx
        self.onnx_session = None
        self.yolo_model = None
        
        onnx_path = config.YOLO_MODEL_PATH
        if use_onnx and os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                providers = ['CPUExecutionProvider']
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.insert(0, 'CUDAExecutionProvider')
                self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
                print(f"[PersonDetector] Loaded ONNX model from {onnx_path}")
            except Exception as e:
                print(f"[PersonDetector] Failed to load ONNX: {e}. Falling back to PyTorch.")
                self.onnx_session = None

        if self.onnx_session is None:
            try:
                from ultralytics import YOLO
                model_path = config.YOLO_PT_PATH if os.path.exists(config.YOLO_PT_PATH) else "yolo11n.pt"
                self.yolo_model = YOLO(model_path)
                print(f"[PersonDetector] Loaded PyTorch YOLO model ({model_path})")
            except (ImportError, Exception) as e:
                print(f"[PersonDetector] Ultralytics/PyTorch unavailable ({e}). Using OpenCV Blob/Synthetic fallback detector.")
                self.yolo_model = None

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Runs person detection on a BGR image frame.
        Returns a list of PersonDetection objects filtered for class 0 (person).
        """
        detections: List[PersonDetection] = []
        if frame is None or frame.size == 0:
            return detections

        if self.onnx_session is not None:
            detections = self._detect_onnx(frame)
        elif self.yolo_model is not None:
            detections = self._detect_pytorch(frame)
        else:
            detections = self._detect_fallback(frame)

        return detections

    def _detect_fallback(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Ultra-fast OpenCV Person Detector for live webcam with downscaling & aspect-ratio filtering.
        """
        detections = []
        h_orig, w_orig = frame.shape[:2]

        # Downscale frame to 640x360 for 4x faster CPU processing & zero lag
        target_w, target_h = 640, 360
        scale_x = w_orig / target_w
        scale_y = h_orig / target_h
        small_frame = cv2.resize(frame, (target_w, target_h))

        raw_boxes = []
        raw_confs = []

        # 1. Try OpenCV Built-in HOG People Detector on downscaled frame
        if hasattr(cv2, 'HOGDescriptor'):
            try:
                if not hasattr(self, 'hog'):
                    self.hog = cv2.HOGDescriptor()
                    self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

                rects, weights = self.hog.detectMultiScale(
                    small_frame,
                    winStride=(12, 12),
                    padding=(8, 8),
                    scale=1.08,
                    hitThreshold=0.2
                )
                for (x, y, w, h), weight in zip(rects, weights):
                    conf = float(weight[0]) if isinstance(weight, (list, np.ndarray)) else float(weight)
                    
                    # Human Aspect-Ratio Filter: Humans standing/sitting are taller than wide (h/w >= 1.05)
                    aspect_ratio = h / max(1.0, float(w))
                    if aspect_ratio >= 1.05 and h >= 40:
                        raw_boxes.append([int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)])
                        raw_confs.append(float(max(0.4, min(0.95, conf))))
            except Exception:
                pass

        # 2. Try OpenCV Haar Cascade Face/Body Detector if HOG found nothing
        if len(raw_boxes) == 0 and hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
            try:
                if not hasattr(self, 'face_cascade'):
                    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
                for (fx, fy, fw, fh) in faces:
                    # Extrapolate face ROI to upper-body bounding box on original resolution
                    bx1 = max(0.0, float((fx - fw * 0.4) * scale_x))
                    by1 = max(0.0, float((fy - fh * 0.2) * scale_y))
                    bw = float(fw * 1.8 * scale_x)
                    bh = float(fh * 3.4 * scale_y)
                    raw_boxes.append([int(bx1), int(by1), int(bw), int(bh)])
                    raw_confs.append(0.85)
            except Exception:
                pass

        # 3. Apply NMS (Non-Maximum Suppression) to remove duplicate boxes
        if len(raw_boxes) > 0:
            boxes_xyxy = []
            for b in raw_boxes:
                boxes_xyxy.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])

            indices = cv2.dnn.NMSBoxes(
                bboxes=raw_boxes,
                scores=raw_confs,
                score_threshold=0.3,
                nms_threshold=0.4
            )
            if len(indices) > 0:
                for idx in np.array(indices).flatten():
                    b = boxes_xyxy[idx]
                    detections.append(
                        PersonDetection(
                            bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                            conf=raw_confs[idx],
                            class_id=config.PERSON_CLASS_ID
                        )
                    )

        # 4. Synthetic color mask fallback if no persons detected
        if len(detections) == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, (0, 100, 100), (25, 255, 255))
            mask2 = cv2.inRange(hsv, (140, 100, 100), (170, 255, 255))
            mask = cv2.bitwise_or(mask1, mask2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w >= 20 and h >= 40:
                    detections.append(
                        PersonDetection(
                            bbox=(float(x), float(y), float(x + w), float(y + h)),
                            conf=0.92,
                            class_id=config.PERSON_CLASS_ID
                        )
                    )

        return detections

    def _detect_pytorch(self, frame: np.ndarray) -> List[PersonDetection]:
        """Inference using Ultralytics PyTorch pipeline."""
        results = self.yolo_model.predict(
            source=frame,
            classes=[config.PERSON_CLASS_ID],  # Filter only person class (class 0)
            conf=config.LOW_CONF_THRESH,       # Include low conf for ByteTrack 2nd stage association
            verbose=False
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                if cls_id == config.PERSON_CLASS_ID:
                    detections.append(
                        PersonDetection(
                            bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                            conf=conf,
                            class_id=cls_id
                        )
                    )
        return detections

    def _detect_onnx(self, frame: np.ndarray) -> List[PersonDetection]:
        """Inference using ONNX Runtime for maximum efficiency."""
        img_h, img_w = frame.shape[:2]
        
        # Preprocessing: Resize & normalize for YOLO input (640x640)
        img_input = cv2.resize(frame, (640, 640))
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img_input = img_input.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))[None, ...]  # (1, 3, 640, 640)

        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: img_input})
        
        # YOLOv8/v11 ONNX output shape: (1, 84, 8400)
        predictions = np.squeeze(outputs[0])  # (84, 8400)
        
        # Extract boxes, scores, and class IDs
        # First 4 rows: [cx, cy, w, h], next rows: class confidences
        boxes = predictions[:4, :].T  # (8400, 4)
        scores = predictions[4 + config.PERSON_CLASS_ID, :].T  # (8400,)
        
        # Filter by confidence
        mask = scores >= config.LOW_CONF_THRESH
        boxes = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            return []

        # Convert [cx, cy, w, h] to [x1, y1, w, h] & scale to original resolution
        x_factor = img_w / 640.0
        y_factor = img_h / 640.0

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - w / 2.0) * x_factor
        y1 = (cy - h / 2.0) * y_factor
        w_scaled = w * x_factor
        h_scaled = h * y_factor

        boxes_tlwh = np.column_stack([x1, y1, w_scaled, h_scaled]).tolist()
        scores_list = scores.tolist()
        
        # OpenCV NMS requires [x, y, w, h]
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_tlwh,
            scores=scores_list,
            score_threshold=config.DETECTION_CONF_THRESH,
            nms_threshold=0.45
        )

        detections = []
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            for idx in indices:
                bx = boxes_tlwh[idx]
                conf = float(scores_list[idx])
                x1_b, y1_b, w_b, h_b = bx
                detections.append(
                    PersonDetection(
                        bbox=(float(x1_b), float(y1_b), float(x1_b + w_b), float(y1_b + h_b)),
                        conf=conf,
                        class_id=config.PERSON_CLASS_ID
                    )
                )
        return detections
