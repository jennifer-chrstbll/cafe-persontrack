import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
import config


class PersonDetection:
    """Single person detection result."""
    def __init__(self, bbox: Tuple[float, float, float, float],
                 conf: float, class_id: int = 0):
        self.bbox = bbox
        self.conf = conf
        self.class_id = class_id
        self.x1, self.y1, self.x2, self.y2 = bbox
        self.width  = max(1.0, self.x2 - self.x1)
        self.height = max(1.0, self.y2 - self.y1)
        self.centroid_x = self.x1 + self.width  / 2.0
        self.centroid_y = self.y1 + self.height / 2.0

    @property
    def tlwh(self):
        return (self.x1, self.y1, self.width, self.height)

    @property
    def centroid(self):
        return (self.centroid_x, self.centroid_y)


# ── Image enhancement ─────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """CLAHE + unsharp mask. Compensates blur & low contrast of CCTV footage."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = _clahe.apply(l)
    blur = cv2.GaussianBlur(l_eq, (0, 0), sigmaX=3)
    l_sh = cv2.addWeighted(l_eq, 1.5, blur, -0.5, 0)  # uint8 input -> saturates automatically, no np.clip needed
    return cv2.cvtColor(cv2.merge([l_sh, a, b]), cv2.COLOR_LAB2BGR)


# ── ONNX inference primitives ─────────────────────────────────────────────────

def _onnx_infer_yolo11(session, frame: np.ndarray,
                       input_size: int,
                       low_thresh: float,
                       conf_thresh: float) -> List[PersonDetection]:  # NOTE: conf_thresh kept for API compatibility but filtering uses low_thresh; ByteTrack handles high/low split internally
    """
    YOLO11 ONNX inference — standard NMS output format.
    Output shape: (1, 84, 8400)
    """
    ih, iw = frame.shape[:2]
    img = cv2.resize(frame, (input_size, input_size))
    img = img[:, :, ::-1]  # BGR->RGB
    blob = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)[None] / 255.0

    out = session.run(None, {session.get_inputs()[0].name: blob})
    pred = np.squeeze(out[0])  # (84, 8400)

    scores = pred[4 + config.PERSON_CLASS_ID, :]
    mask = scores >= low_thresh
    if not mask.any():
        return []

    boxes = pred[:4, mask].T
    scores = scores[mask]
    sx, sy = iw / input_size, ih / input_size
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * sx
    y1 = (cy - h / 2) * sy
    x2 = (cx + w / 2) * sx
    y2 = (cy + h / 2) * sy

    # Aspect ratio filter (relaxed for partial body / head-only crops)
    ratio = (y2 - y1) / np.maximum(x2 - x1, 1.0)
    valid = ratio >= 0.5
    if not valid.any():
        return []

    boxes_tlwh = np.stack([x1[valid], y1[valid],
                           (x2 - x1)[valid], (y2 - y1)[valid]], axis=1).tolist()
    scores_l = scores[valid].tolist()
    idxs = cv2.dnn.NMSBoxes(boxes_tlwh, scores_l,
                            score_threshold=low_thresh, nms_threshold=0.45)
    dets = []
    for i in (np.array(idxs).flatten() if len(idxs) else []):
        bx = boxes_tlwh[i]
        dets.append(PersonDetection(
            bbox=(bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3]),
            conf=float(scores_l[i])))
    return dets


def _onnx_infer_yolo26(session, frame: np.ndarray,
                       input_size: int,
                       low_thresh: float,
                       conf_thresh: float) -> List[PersonDetection]:  # NOTE: conf_thresh kept for API compatibility; ByteTrack high/low split is done by TRACK_THRESH, not here
    """
    YOLO26 ONNX inference — NMS-free end-to-end output format.
    Output shape: (1, 300, 6) where 6 = [x1, y1, x2, y2, conf, class_id]
    Allows candidates down to low_thresh so ByteTrack can use them in Step 2.
    """
    ih, iw = frame.shape[:2]
    img = cv2.resize(frame, (input_size, input_size))
    img = img[:, :, ::-1]  # BGR->RGB
    blob = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)[None] / 255.0

    out = session.run(None, {session.get_inputs()[0].name: blob})
    preds = out[0][0]  # (300, 6)

    sx, sy = iw / input_size, ih / input_size
    dets = []
    for row in preds:
        x1, y1, x2, y2, conf, cls_id = row
        if int(cls_id) != config.PERSON_CLASS_ID:
            continue
        # Use low_thresh here identically to YOLO11 so ByteTrack gets low-confidence tier
        if conf < low_thresh:
            continue
        # Scale back to original frame coords
        x1 *= sx; y1 *= sy; x2 *= sx; y2 *= sy
        # Aspect ratio filter (relaxed for partial body)
        h_box = y2 - y1
        w_box = x2 - x1
        if h_box / max(w_box, 1.0) < 0.5:
            continue
        dets.append(PersonDetection(
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            conf=float(conf)))
    return dets


def _merge_nms(dets: List[PersonDetection], conf_thresh: float) -> List[PersonDetection]:
    """Global NMS across multi-pass detections to remove cross-pass duplicates.

    nms_threshold=0.55 (raised from 0.45) — cross-pass boxes for the same person
    may overlap less than same-pass boxes because Pass 2 crops a sub-region and
    rescales differently. A higher threshold ensures we still merge them as
    duplicates even when IoU drops to ~0.50.
    """
    if not dets:
        return []
    boxes = [[d.x1, d.y1, d.width, d.height] for d in dets]
    scores = [d.conf for d in dets]
    idxs = cv2.dnn.NMSBoxes(boxes, scores,
                            score_threshold=conf_thresh, nms_threshold=0.55)
    return [dets[i] for i in (np.array(idxs).flatten() if len(idxs) else [])]


# ── PersonDetector ────────────────────────────────────────────────────────────

class PersonDetector:
    """
    Person detector supporting both YOLO11n and YOLO26n with 100% parameter parity.

    Model selection via config.ACTIVE_MODEL or model_name:
      "yolo11" (default) — Anchor-free + NMS
      "yolo26"           — End-to-end NMS-free with STAL
    """

    FAR_REGION_Y_FRAC = 0.55   # Top 55% = where far/small people appear

    def __init__(self, conf_thresh: float = config.DETECTION_CONF_THRESH,
                 use_onnx: bool = True,
                 input_size: int = None,
                 model_name: str = None):
        self.conf_thresh = conf_thresh
        self.input_size  = input_size or getattr(config, 'YOLO_INPUT_SIZE', 960)
        self.model_name  = (model_name or getattr(config, 'ACTIVE_MODEL', 'yolo11')).lower()
        self.session     = None
        self.yolo_model  = None
        self.is_yolo26   = (self.model_name == 'yolo26')

        if self.is_yolo26:
            onnx_path = getattr(config, 'YOLO26_MODEL_PATH', '')
            pt_path   = getattr(config, 'YOLO26_PT_PATH',    'yolo26n.pt')
        else:
            onnx_path = config.YOLO_MODEL_PATH
            pt_path   = config.YOLO_PT_PATH

        label = 'YOLO26n (NMS-free, STAL)' if self.is_yolo26 else 'YOLO11n'

        if use_onnx and onnx_path and os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                             if 'CUDAExecutionProvider' in ort.get_available_providers()
                             else ['CPUExecutionProvider'])
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(onnx_path, sess_options=opts,
                                                    providers=providers)
                _ishape = self.session.get_inputs()[0].shape
                self._model_native_size = int(_ishape[2]) if isinstance(_ishape[2], int) and int(_ishape[2]) > 0 else None
                _sz_str = str(self._model_native_size) if self._model_native_size else 'dynamic'
                print(f"[PersonDetector] {label} ONNX: {onnx_path}  model_size={_sz_str}  requested={self.input_size}")
            except Exception as e:
                print(f'[PersonDetector] ONNX load failed: {e}')

        if self.session is None and self.yolo_model is None and self.is_yolo26:
            print("[PersonDetector] YOLO26 unavailable, attempting fallback to YOLO11...")
            self.is_yolo26 = False
            self.model_name = 'yolo11'
            onnx_path = config.YOLO_MODEL_PATH
            pt_path   = config.YOLO_PT_PATH
            if use_onnx and onnx_path and os.path.exists(onnx_path):
                try:
                    import onnxruntime as ort
                    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                                 if 'CUDAExecutionProvider' in ort.get_available_providers()
                                 else ['CPUExecutionProvider'])
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 4
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
                    _ishape = self.session.get_inputs()[0].shape
                    self._model_native_size = int(_ishape[2]) if isinstance(_ishape[2], int) and int(_ishape[2]) > 0 else None
                    print(f"[PersonDetector] Fallback to YOLO11n ONNX: {onnx_path}")
                except Exception as e:
                    print(f'[PersonDetector] Fallback ONNX load failed: {e}')

        if self.session is None and self.yolo_model is None:
            try:
                from ultralytics import YOLO
                pt = pt_path if os.path.exists(pt_path) else (
                    'yolo26n.pt' if self.is_yolo26 else 'yolo11n.pt')
                self.yolo_model = YOLO(pt)
                print(f'[PersonDetector] {label} PyTorch: {pt}  imgsz={self.input_size}')
            except Exception as e:
                print(f'[PersonDetector] No model available: {e}')

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        if frame is None or frame.size == 0:
            return []
        enhanced = enhance_frame(frame)
        if self.session is not None:
            return self._detect_onnx(enhanced)
        elif self.yolo_model is not None:
            return self._detect_pytorch(enhanced)
        else:
            return self._detect_hog(enhanced)

    def _detect_onnx(self, frame: np.ndarray) -> List[PersonDetection]:
        H = frame.shape[0]
        conf_t = self.conf_thresh
        low_t  = config.LOW_CONF_THRESH
        native = getattr(self, '_model_native_size', None)
        main_sz = native if native else self.input_size

        if self.is_yolo26:
            dets1 = _onnx_infer_yolo26(self.session, frame, main_sz, low_t, conf_t)
        else:
            dets1 = _onnx_infer_yolo11(self.session, frame, main_sz, low_t, conf_t)

        enable_far_pass = getattr(config, 'ENABLE_FAR_REGION_PASS', False)
        if enable_far_pass:
            far_sz  = main_sz
            far_y2  = int(H * self.FAR_REGION_Y_FRAC)
            far_crop = frame[:far_y2, :]
            if self.is_yolo26:
                dets2_r = _onnx_infer_yolo26(self.session, far_crop, far_sz, low_t * 0.85, conf_t * 0.85)
            else:
                dets2_r = _onnx_infer_yolo11(self.session, far_crop, far_sz, low_t * 0.85, conf_t * 0.85)
            dets2 = [PersonDetection(bbox=(d.x1, d.y1, d.x2, d.y2),
                                     conf=d.conf * 0.90) for d in dets2_r]
            return _merge_nms(dets1 + dets2, low_t)

        return dets1

    def _detect_pytorch(self, frame: np.ndarray) -> List[PersonDetection]:
        results = self.yolo_model.predict(
            source=frame, classes=[config.PERSON_CLASS_ID],
            conf=config.LOW_CONF_THRESH, imgsz=self.input_size, verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                xy, c = box.xyxy[0].cpu().numpy(), float(box.conf[0])
                if int(box.cls[0]) == config.PERSON_CLASS_ID and c >= config.LOW_CONF_THRESH:
                    dets.append(PersonDetection(
                        bbox=(float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])),
                        conf=c))
        return dets

    def _detect_hog(self, frame: np.ndarray) -> List[PersonDetection]:
        if not hasattr(self, '_hog'):
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        small = cv2.resize(frame, (640, 360))
        sx, sy = frame.shape[1] / 640, frame.shape[0] / 360
        rects, ws = self._hog.detectMultiScale(
            small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        dets = []
        for (x, y, w, h), wt in zip(rects, ws):
            c = float(wt[0] if hasattr(wt, '__len__') else wt)
            if h / max(w, 1) >= 0.8 and h >= 30:
                dets.append(PersonDetection(
                    bbox=(x * sx, y * sy, (x + w) * sx, (y + h) * sy),
                    conf=min(0.90, max(0.40, c))))
        return dets
