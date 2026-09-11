"""
tools/quantize_models.py
========================
Static INT8 and FP16 quantization tool for:
1. Detectors: YOLO11n and YOLO26n (INT8 static calibration on ceiling CCTV footage)
2. ReID: OSNet x0_25 (Person crops calibration, INT8 vs FP16 validation)

Usage:
  python tools/quantize_models.py --video cctv_test.mp4 --num-frames 200
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("quantize")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
WEIGHTS_DIR = BASE_DIR / "weights"
CALIB_DIR = BASE_DIR / "calib_images"
PERSON_CROPS_DIR = BASE_DIR / "calib_person_crops"


# ── Calibration Data Readers ───────────────────────────────────────────────────

class CafeDetectorCalibReader(CalibrationDataReader):
    """Feeds normalized full-frame CCTV images (1, 3, size, size) to YOLO models."""
    def __init__(self, image_dir: Path, input_name: str = "images", size: int = 640):
        self.image_paths = sorted([
            p for p in image_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        self.input_name = input_name
        self.size = size
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.image_paths):
            return None
        p = self.image_paths[self.idx]
        img = cv2.imread(str(p))
        if img is None:
            self.idx += 1
            return self.get_next()
        img = cv2.resize(img, (self.size, self.size))
        # BGR -> RGB -> [3, H, W] -> float32 [0..1]
        blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)
        self.idx += 1
        return {self.input_name: blob}

    def rewind(self):
        self.idx = 0


class OSNetCalibReader(CalibrationDataReader):
    """Feeds normalized person crops (1, 3, 256, 128) to OSNet ReID model."""
    def __init__(self, crops_dir: Path, input_name: str = "images", width: int = 128, height: int = 256):
        self.crop_paths = sorted([
            p for p in crops_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        self.input_name = input_name
        self.width = width
        self.height = height
        self.idx = 0
        # ImageNet mean & std for OSNet
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def get_next(self):
        if self.idx >= len(self.crop_paths):
            return None
        p = self.crop_paths[self.idx]
        img = cv2.imread(str(p))
        if img is None:
            self.idx += 1
            return self.get_next()
        img = cv2.resize(img, (self.width, self.height))
        # BGR -> RGB -> [1, 3, H, W]
        blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)
        blob = (blob - self.mean) / self.std
        self.idx += 1
        return {self.input_name: blob.astype(np.float32)}

    def rewind(self):
        self.idx = 0


# ── Frame and Crop Extraction ──────────────────────────────────────────────────

def extract_calib_data(video_path: str, num_frames: int = 200):
    """
    Extract calibration frames and person crops from the actual ceiling CCTV footage.
    """
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    PERSON_CROPS_DIR.mkdir(parents=True, exist_ok=True)

    existing_frames = list(CALIB_DIR.glob("*.jpg"))
    if len(existing_frames) >= num_frames:
        logger.info(f"Using {len(existing_frames)} existing frames in {CALIB_DIR}")
        return

    logger.info(f"Extracting {num_frames} frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Could not read frames from {video_path}")

    # Load FP32 detector for generating genuine person crops
    from models.detector import PersonDetector
    detector = PersonDetector(model_name="yolo11", use_onnx=True)

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    saved_frames = 0
    saved_crops = 0

    for f_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_out_path = CALIB_DIR / f"frame_{f_idx:05d}.jpg"
        cv2.imwrite(str(frame_out_path), frame)
        saved_frames += 1

        # Detect person crops for ReID calibration
        dets = detector.detect(frame)
        for d_idx, d in enumerate(dets):
            if d.conf >= 0.35 and d.width >= 20 and d.height >= 40:
                x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
                crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if crop.size > 0:
                    crop_path = PERSON_CROPS_DIR / f"crop_{f_idx:05d}_{d_idx:02d}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    saved_crops += 1

    cap.release()
    logger.info(f"Extracted {saved_frames} calibration frames to {CALIB_DIR}")
    logger.info(f"Extracted {saved_crops} person crops to {PERSON_CROPS_DIR}")


# ── Quantization Functions ─────────────────────────────────────────────────────

def quantize_detector(model_name: str, input_size: int = 640):
    """
    Quantize YOLO11n or YOLO26n to INT8 QDQ format using static calibration.
    """
    in_model = WEIGHTS_DIR / f"{model_name}.onnx"
    out_model = WEIGHTS_DIR / f"{model_name}_int8.onnx"

    if not in_model.exists():
        logger.warning(f"Model {in_model} not found, skipping.")
        return

    logger.info(f"--- Quantizing {model_name} to INT8 ---")
    calib_reader = CafeDetectorCalibReader(CALIB_DIR, input_name="images", size=input_size)

    quantize_static(
        model_input=str(in_model),
        model_output=str(out_model),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    orig_mb = in_model.stat().st_size / (1024 * 1024)
    q_mb = out_model.stat().st_size / (1024 * 1024)
    logger.info(f"Successfully generated {out_model.name}: {orig_mb:.2f} MB -> {q_mb:.2f} MB")


def quantize_osnet():
    """
    Create FP16 and test INT8 for OSNet x0_25 ReID.
    As established in Phase 3.1, FP16 is safe and calibration-free for embedding models,
    while INT8 must be validated for embedding discriminability.
    """
    in_model = WEIGHTS_DIR / "osnet_x0_25.onnx"
    fp16_model = WEIGHTS_DIR / "osnet_x0_25_fp16.onnx"
    int8_model = WEIGHTS_DIR / "osnet_x0_25_int8.onnx"

    if not in_model.exists():
        logger.warning(f"Model {in_model} not found, skipping.")
        return

    # 1. Generate FP16 OSNet
    logger.info("--- Generating FP16 OSNet x0_25 ---")
    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
        m = onnx.load(str(in_model))
        m_fp16 = convert_float_to_float16(m, keep_io_types=True)
        onnx.save(m_fp16, str(fp16_model))
        logger.info(f"Generated {fp16_model.name}: {in_model.stat().st_size / 1024:.1f} KB -> {fp16_model.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        logger.error(f"Failed to generate FP16 OSNet: {e}")

    # 2. Generate INT8 OSNet with person-crop calibration
    logger.info("--- Quantizing OSNet x0_25 to INT8 ---")
    sess = ort.InferenceSession(str(in_model), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    calib_reader = OSNetCalibReader(PERSON_CROPS_DIR, input_name=input_name, width=128, height=256)

    try:
        quantize_static(
            model_input=str(in_model),
            model_output=str(int8_model),
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
        logger.info(f"Generated {int8_model.name}: {int8_model.stat().st_size / 1024:.1f} KB")
        validate_osnet_quantization(str(in_model), str(int8_model))
    except Exception as e:
        logger.warning(f"OSNet INT8 quantization failed or skipped: {e}")


def validate_osnet_quantization(fp32_path: str, int8_path: str):
    """
    Validate cosine similarity correlation between FP32 and INT8 OSNet on person crops.
    """
    crops = sorted(list(PERSON_CROPS_DIR.glob("*.jpg")))[:20]
    if len(crops) < 2:
        return

    sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    inp_name = sess_fp32.get_inputs()[0].name

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    cos_sims = []
    for p in crops:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (128, 256))
        blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)
        blob = (blob - mean) / std

        out_fp32 = sess_fp32.run(None, {inp_name: blob.astype(np.float32)})[0].flatten()
        out_int8 = sess_int8.run(None, {inp_name: blob.astype(np.float32)})[0].flatten()

        sim = np.dot(out_fp32, out_int8) / (np.linalg.norm(out_fp32) * np.linalg.norm(out_int8) + 1e-10)
        cos_sims.append(sim)

    avg_sim = float(np.mean(cos_sims))
    logger.info(f"OSNet FP32 vs INT8 average cosine correlation: {avg_sim:.4f}")
    if avg_sim < 0.90:
        logger.warning("INT8 OSNet discriminability degraded (<0.90 correlation) — recommend using FP16 fallback.")
    else:
        logger.info("INT8 OSNet preservation verified (>=0.90 correlation).")


def main():
    parser = argparse.ArgumentParser(description="Quantize YOLO detectors and OSNet ReID for cafe-persontrack.")
    parser.add_argument("--video", type=str, default="cctv_test.mp4", help="Path to CCTV video for calibration")
    parser.add_argument("--num-frames", type=int, default=200, help="Number of calibration frames")
    parser.add_argument("--detectors-only", action="store_true", help="Only quantize detectors")
    args = parser.parse_args()

    extract_calib_data(args.video, num_frames=args.num_frames)

    quantize_detector("yolo11n")
    quantize_detector("yolo26n")

    if not args.detectors_only:
        quantize_osnet()

    logger.info("=== All quantization tasks finished ===")


if __name__ == "__main__":
    main()
