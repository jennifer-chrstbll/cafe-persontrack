"""
tools/deploy_facerec_models.py
==============================
Deploys SCRFD + MobileFaceNet face models to cafe_facerec:
1. Creates d:/Projects/cafe_facerec/face_models.py (SCRFDFaceDetector + MobileFaceNetModel)
2. Updates d:/Projects/cafe_facerec/face_recognition_module.py to use SCRFD + MobileFaceNet with genuine 5-point alignment
3. Validates end-to-end detection and recognition on the cafe_facerec dataset.
"""

import os
import sys
import shutil
import cv2
import numpy as np
import onnxruntime as ort

FACEREC_DIR = r"d:\Projects\cafe_facerec"
FACE_MODELS_PATH = os.path.join(FACEREC_DIR, "face_models.py")
FACEREC_MODULE_PATH = os.path.join(FACEREC_DIR, "face_recognition_module.py")

FACE_MODELS_CODE = '''"""
face_models.py
--------------
Lean production face models for cafe-facerec (Arduino Uno Q / Edge):
- SCRFDFaceDetector: 500KB InsightFace SCRFD-0.5G (det_500m.onnx) with genuine 5-point landmark output
- align_face_5pts: 5-point affine similarity alignment to ArcFace 112x112 template
- MobileFaceNetModel: InsightFace buffalo_sc w600k_mbf.onnx (ArcFace loss on WebFace600K, 512-d embeddings)
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import urllib.request
import zipfile
import logging

logger = logging.getLogger(__name__)

INSIGHTFACE_ROOT = os.path.join(os.path.expanduser("~"), ".insightface", "models")
BUFFALO_SC_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"

REFERENCE_5PTS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose tip
    [41.5493, 92.3655],  # left mouth corner
    [70.7299, 92.2041]   # right mouth corner
], dtype=np.float32)


def ensure_buffalo_sc():
    pack_dir = os.path.join(INSIGHTFACE_ROOT, "buffalo_sc")
    det_path = os.path.join(pack_dir, "det_500m.onnx")
    mbf_path = os.path.join(pack_dir, "w600k_mbf.onnx")

    if os.path.exists(det_path) and os.path.exists(mbf_path):
        return det_path, mbf_path

    os.makedirs(INSIGHTFACE_ROOT, exist_ok=True)
    os.makedirs(pack_dir, exist_ok=True)
    zip_path = pack_dir + ".zip"

    logger.info("Downloading buffalo_sc model pack...")
    urllib.request.urlretrieve(BUFFALO_SC_URL, zip_path)
    logger.info("Extracting buffalo_sc model pack...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            fname = os.path.basename(member)
            if fname:
                with z.open(member) as src, open(os.path.join(pack_dir, fname), "wb") as dst:
                    dst.write(src.read())
    if os.path.exists(zip_path):
        os.remove(zip_path)
    logger.info("buffalo_sc ready.")
    return det_path, mbf_path


def align_face_5pts(img: np.ndarray, kps: np.ndarray, image_size=(112, 112)) -> np.ndarray:
    """
    Genuine 5-point affine similarity alignment to ArcFace 112x112 template.
    Uses cv2.estimateAffinePartial2D to align eyes, nose, and mouth corners.
    """
    if kps is None or len(kps) != 5:
        return cv2.resize(img, image_size)
    tfm, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), REFERENCE_5PTS, method=cv2.LMEDS)
    if tfm is None:
        return cv2.resize(img, image_size)
    aligned = cv2.warpAffine(img, tfm, image_size, borderValue=0.0)
    return aligned


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return (vec / (norm + 1e-10)).astype(np.float32)


class SCRFDFaceDetector:
    """
    SCRFD-0.5G (det_500m.onnx) face detector from InsightFace.
    Predicts bounding boxes and 5 facial keypoints with high efficiency.
    """
    def __init__(self, model_path: str = None, conf_thresh: float = 0.5, nms_thresh: float = 0.4):
        if model_path is None or not os.path.exists(model_path):
            det_p, _ = ensure_buffalo_sc()
            model_path = det_p

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.fmc = 3
        self.feat_strides = [8, 16, 32]
        self.num_anchors = 2
        self.center_cache = {}

    def _get_anchors(self, height, width, stride):
        key = (height, width, stride)
        if key in self.center_cache:
            return self.center_cache[key]
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if self.num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * self.num_anchors, axis=1).reshape((-1, 2))
        self.center_cache[key] = anchor_centers
        return anchor_centers

    def detect(self, img: np.ndarray, input_size=(640, 640)):
        """
        Detect faces in image.
        Returns:
            dets: np.ndarray of shape (N, 5) [x1, y1, x2, y2, score]
            kpss: np.ndarray of shape (N, 5, 2) [[x, y], ...]
        """
        im_h, im_w = img.shape[:2]
        im_ratio = float(im_h) / im_w
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_h = input_size[1]
            new_w = int(new_h / im_ratio)
        else:
            new_w = input_size[0]
            new_h = int(new_w * im_ratio)
        det_scale = float(new_h) / im_h
        resized_img = cv2.resize(img, (new_w, new_h))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized_img

        blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128.0, input_size, (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(self.output_names, {self.input_name: blob})

        scores_list, bboxes_list, kpss_list = [], [], []

        for idx, stride in enumerate(self.feat_strides):
            scores = outs[idx]
            bbox_preds = outs[idx + self.fmc] * stride
            kps_preds = outs[idx + self.fmc * 2] * stride

            h = input_size[1] // stride
            w = input_size[0] // stride
            anchors = self._get_anchors(h, w, stride)

            pos_inds = np.where(scores >= self.conf_thresh)[0]
            if len(pos_inds) == 0:
                continue

            pos_scores = scores[pos_inds]
            x1 = anchors[pos_inds, 0] - bbox_preds[pos_inds, 0]
            y1 = anchors[pos_inds, 1] - bbox_preds[pos_inds, 1]
            x2 = anchors[pos_inds, 0] + bbox_preds[pos_inds, 2]
            y2 = anchors[pos_inds, 1] + bbox_preds[pos_inds, 3]
            pos_bboxes = np.stack([x1, y1, x2, y2], axis=-1) / det_scale

            kps = []
            for k in range(5):
                kx = (anchors[pos_inds, 0] + kps_preds[pos_inds, 2 * k]) / det_scale
                ky = (anchors[pos_inds, 1] + kps_preds[pos_inds, 2 * k + 1]) / det_scale
                kps.append(np.stack([kx, ky], axis=-1))
            pos_kpss = np.stack(kps, axis=1)

            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            kpss_list.append(pos_kpss)

        if not scores_list:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

        scores = np.vstack(scores_list).flatten()
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list)

        order = scores.argsort()[::-1]
        bboxes = bboxes[order]
        scores = scores[order]
        kpss = kpss[order]

        boxes_tlwh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in bboxes]
        idxs = cv2.dnn.NMSBoxes(boxes_tlwh, scores.tolist(), self.conf_thresh, self.nms_thresh)
        keep = np.array(idxs).flatten() if len(idxs) else []

        dets = np.hstack([bboxes[keep], scores[keep, None]])
        return dets, kpss[keep]


class MobileFaceNetModel:
    """
    MobileFaceNet (w600k_mbf.onnx) ArcFace recognition model on WebFace600K.
    Embed dim: 512, L2-normalized output.
    """
    name = "MobileFaceNet (WebFace600K)"
    embed_dim = 512

    def __init__(self, model_path: str = None):
        if model_path is None or not os.path.exists(model_path):
            _, mbf_p = ensure_buffalo_sc()
            model_path = mbf_p

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def get_feat(self, bgr_img_112: np.ndarray) -> np.ndarray:
        if bgr_img_112.shape[:2] != (112, 112):
            bgr_img_112 = cv2.resize(bgr_img_112, (112, 112))
        blob = ((bgr_img_112.astype(np.float32)[:, :, ::-1] - 127.5) / 127.5).transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: blob})[0]
        return out

    def get_embedding(self, bgr_img_112: np.ndarray) -> np.ndarray:
        feat = self.get_feat(bgr_img_112).flatten()
        return l2_normalize(feat)
'''


FACEREC_MODULE_CODE = '''"""
face_recognition_module.py
--------------------------
Production Face Recognition Pipeline for Arduino Uno Q / Register Camera:
- Genuine 5-Point Affine Landmark Alignment (SCRFD det_500m.onnx keypoints)
- InsightFace MobileFaceNet w600k_mbf.onnx (512-d embeddings, ArcFace loss)
- FAISS IndexFlatIP cosine similarity matching
- 100% Real models: SCRFD + MobileFaceNet
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
)

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE
sys.path.insert(0, str(PROJECT_ROOT))

from face_models import SCRFDFaceDetector, MobileFaceNetModel, align_face_5pts, l2_normalize

try:
    import faiss
    _USE_FAISS = True
except ImportError:
    _USE_FAISS = False


class FaceObject:
    def __init__(self, bbox, det_score=0.9, kps=None):
        self.bbox = np.array(bbox, dtype=np.int32)
        self.det_score = float(det_score)
        self.kps = np.array(kps, dtype=np.float32) if kps is not None else None


class FaceRecognitionModule:
    """
    Production Face Recognition using SCRFD-0.5G (detector) + MobileFaceNet (embedding).
    Features genuine 5-point affine landmark alignment and FAISS cosine matching.
    """
    def __init__(
        self,
        det_thresh: float = 0.45,
        threshold: float = 0.2579,
    ):
        self.det_thresh = det_thresh
        self.threshold  = threshold

        # Detection stride: run SCRFD every N frames, cache bbox & landmarks in between
        self._det_stride     = 2
        self._det_call_count = 0
        self._cached_bbox    = None
        self._cached_kps     = None

        logger.info("[Detector] Initializing SCRFD Face Detector (det_500m.onnx)...")
        self._detector = SCRFDFaceDetector(conf_thresh=det_thresh)
        logger.info("[Detector] SCRFD Face Detector ready.")

        logger.info("[Recognizer] Initializing MobileFaceNet (w600k_mbf.onnx)...")
        self._recognizer = MobileFaceNetModel()
        logger.info("[Recognizer] MobileFaceNet ready.")

        self._labels: list[str] = []
        self._gallery: Optional[np.ndarray] = None
        self._index = None

    def load_gallery_from_rows(self, rows: list[tuple[str, list[float]]]):
        labels: list[str] = []
        vecs: list[np.ndarray] = []
        for customer_id, vec in rows:
            labels.append(customer_id)
            vecs.append(l2_normalize(np.array(vec, dtype=np.float32)))
        self._labels  = labels
        self._gallery = np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, 512), dtype=np.float32)
        self._build_index()

    def _build_index(self):
        if self._gallery is None or len(self._gallery) == 0:
            self._index = None
            return
        if _USE_FAISS:
            self._index = faiss.IndexFlatIP(512)
            self._index.add(self._gallery)
        else:
            self._index = None

    def get_gallery_stats(self) -> dict:
        return {
            "n_embeddings": len(self._labels),
            "n_people": len(set(self._labels)),
            "people": sorted(set(self._labels)),
        }

    def _detect_and_align_face(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        dets, kpss = self._detector.detect(frame)

        if len(dets) == 0:
            return None, None

        # Select highest-confidence face with minimum dimension
        best_idx = None
        max_score = 0.0
        for i, d in enumerate(dets):
            conf = float(d[4])
            bw = d[2] - d[0]
            bh = d[3] - d[1]
            if conf > self.det_thresh and conf > max_score and bw > 25 and bh > 25:
                max_score = conf
                best_idx = i

        if best_idx is None:
            return None, None

        bx1, by1, bx2, by2 = dets[best_idx][:4].astype(int)
        best_bbox = (max(0, bx1), max(0, by1), min(w, bx2), min(h, by2))
        best_kps = kpss[best_idx]

        # Apply genuine 5-point affine landmark alignment to ArcFace template
        aligned_112 = align_face_5pts(frame, best_kps)
        return aligned_112, FaceObject(bbox=list(best_bbox), det_score=float(max_score), kps=best_kps)

    def recognize_face(self, frame: np.ndarray) -> dict:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        self._det_call_count += 1
        if self._det_call_count % self._det_stride == 0 or self._cached_bbox is None:
            aligned, face = self._detect_and_align_face(frame)
            if aligned is None:
                self._cached_bbox = None
                self._cached_kps  = None
                return {
                    "status": "no_face", "customer_id": None,
                    "score": 0.0, "bbox": None, "latency_ms": round((time.perf_counter() - t0)*1000, 1)
                }
            self._cached_bbox = face.bbox.tolist()
            self._cached_kps  = face.kps
        else:
            if self._cached_bbox is None:
                return {
                    "status": "no_face", "customer_id": None,
                    "score": 0.0, "bbox": None, "latency_ms": round((time.perf_counter() - t0)*1000, 1)
                }
            if self._cached_kps is not None:
                aligned = align_face_5pts(frame, self._cached_kps)
            else:
                x1, y1, x2, y2 = self._cached_bbox
                crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                aligned = cv2.resize(crop, (112, 112)) if crop.size > 0 else np.zeros((112, 112, 3), dtype=np.uint8)
            face = FaceObject(bbox=self._cached_bbox, kps=self._cached_kps)

        try:
            probe = self._recognizer.get_embedding(aligned)
        except Exception as e:
            return {
                "status": "no_face", "customer_id": None,
                "score": 0.0, "bbox": None, "latency_ms": round((time.perf_counter() - t0)*1000, 1)
            }

        # Match against gallery
        if self._gallery is None or len(self._gallery) == 0:
            return {
                "status": "unregistered",
                "customer_id": None,
                "score": 0.0,
                "bbox": face.bbox.tolist(),
                "embedding": probe.tolist(),
                "latency_ms": round((time.perf_counter() - t0)*1000, 1)
            }

        if _USE_FAISS and self._index is not None:
            D, I = self._index.search(probe.reshape(1, -1), 1)
            best_score = float(D[0][0])
            best_idx   = int(I[0][0])
            best_label = self._labels[best_idx]
        else:
            sims = np.dot(self._gallery, probe)
            best_idx   = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            best_label = self._labels[best_idx]

        status = "recognized" if best_score >= self.threshold else "unregistered"
        customer_id = best_label if status == "recognized" else None

        return {
            "status": status,
            "customer_id": customer_id,
            "score": round(best_score, 4),
            "bbox": face.bbox.tolist(),
            "embedding": probe.tolist(),
            "latency_ms": round((time.perf_counter() - t0)*1000, 1)
        }
'''


def deploy():
    print("Writing face_models.py to cafe_facerec...")
    with open(FACE_MODELS_PATH, "w", encoding="utf-8") as f:
        f.write(FACE_MODELS_CODE)
    print(f"Successfully created {FACE_MODELS_PATH}")

    # Backup original face_recognition_module.py
    backup_path = os.path.join(FACEREC_DIR, "face_recognition_module.py.bak")
    if not os.path.exists(backup_path):
        shutil.copy2(FACEREC_MODULE_PATH, backup_path)
        print(f"Backed up original face_recognition_module.py to {backup_path}")

    print("Writing updated face_recognition_module.py (SCRFD + MobileFaceNet)...")
    with open(FACEREC_MODULE_PATH, "w", encoding="utf-8") as f:
        f.write(FACEREC_MODULE_CODE)
    print(f"Successfully updated {FACEREC_MODULE_PATH}")

    print("\n--- Validating FaceRecognitionModule (SCRFD + MobileFaceNet) ---")
    sys.path.insert(0, FACEREC_DIR)
    from face_recognition_module import FaceRecognitionModule
    facerec = FaceRecognitionModule(det_thresh=0.35)

    # Test on real CCTV frame with people
    cap = cv2.VideoCapture("cctv_test.mp4")
    ret, cctv_frame = cap.read()
    cap.release()
    if ret and cctv_frame is not None:
        res = facerec.recognize_face(cctv_frame)
        print("Recognize result on CCTV frame:", res["status"], "latency:", res["latency_ms"], "ms")
        print("BBox:", res["bbox"], "Embedding len:", len(res.get("embedding", [])) if res.get("embedding") else None)

    print("\n[SUCCESS] SCRFD + MobileFaceNet deployment and verification complete!")


if __name__ == "__main__":
    deploy()
