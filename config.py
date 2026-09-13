import os
from typing import Dict, List, Tuple, Any

# Root Project Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ────────────────────────────────────────────────────
# Detector backend switch — flip with env var, no code edit needed
# Usage: DETECTOR_BACKEND=yolo26n python demo_single_cam.py
# ────────────────────────────────────────────────────
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "yolo11n")  # "yolo11n" | "yolo26n"

DETECTOR_PATHS: Dict[str, Dict[str, str]] = {
    "yolo11n": {
        "pt":   os.path.join(BASE_DIR, "weights", "yolo11n.pt"),
        "onnx": os.path.join(BASE_DIR, "weights", "yolo11n.onnx"),
        "int8": os.path.join(BASE_DIR, "weights", "yolo11n_int8.onnx"),
    },
    "yolo26n": {
        "pt":   os.path.join(BASE_DIR, "weights", "yolo26n.pt"),
        "onnx": os.path.join(BASE_DIR, "weights", "yolo26n.onnx"),
        "int8": os.path.join(BASE_DIR, "weights", "yolo26n_int8.onnx"),
    },
}

# detector.py expects "yolo11" / "yolo26" (no trailing 'n'), not "yolo11n" / "yolo26n"
ACTIVE_MODEL = "yolo26" if DETECTOR_BACKEND == "yolo26n" else "yolo11"
 
YOLO11_MODEL_PATH = DETECTOR_PATHS["yolo11n"][DETECTOR_PRECISION if DETECTOR_PRECISION in DETECTOR_PATHS["yolo11n"] else "onnx"]
YOLO11_PT_PATH = DETECTOR_PATHS["yolo11n"]["pt"]
YOLO26_MODEL_PATH = DETECTOR_PATHS["yolo26n"][DETECTOR_PRECISION if DETECTOR_PRECISION in DETECTOR_PATHS["yolo26n"] else "onnx"]
YOLO26_PT_PATH = DETECTOR_PATHS["yolo26n"]["pt"]

if DETECTOR_BACKEND not in DETECTOR_PATHS:
    raise ValueError(
        f"Unknown DETECTOR_BACKEND='{DETECTOR_BACKEND}'. "
        f"Choose from: {list(DETECTOR_PATHS.keys())}"
    )

# Precision switch — "int8" once you've quantized + validated, else "onnx" (fp32) or export fp16 yourself
DETECTOR_PRECISION = os.getenv("DETECTOR_PRECISION", "onnx")  # "onnx" | "int8"

# Backward-compatible names some of your other modules may still import directly.
# TODO: once everything reads DETECTOR_PATHS[DETECTOR_BACKEND] instead, these two lines can be deleted.
YOLO_MODEL_PATH = DETECTOR_PATHS[DETECTOR_BACKEND][DETECTOR_PRECISION if DETECTOR_PRECISION in DETECTOR_PATHS[DETECTOR_BACKEND] else "onnx"]
YOLO_PT_PATH = DETECTOR_PATHS[DETECTOR_BACKEND]["pt"]

OSNET_MODEL_PATH = os.path.join(BASE_DIR, "weights", "osnet_x0_25.onnx")
OSNET_MODEL_PATH_FP16 = os.path.join(BASE_DIR, "weights", "osnet_x0_25_fp16.onnx")
OSNET_MODEL_PATH_INT8 = os.path.join(BASE_DIR, "weights", "osnet_x0_25_int8.onnx")
OSNET_PT_PATH = os.path.join(BASE_DIR, "weights", "osnet_x0_25_msmt17.pth")

# Which OSNet precision to actually load — default FP16 until you've validated INT8
# per the quantization-warning checklist (same-person / different-person similarity test).
OSNET_PRECISION = os.getenv("OSNET_PRECISION", "fp16")  # "fp32" | "fp16" | "int8"
_OSNET_PATHS = {
    "fp32": OSNET_MODEL_PATH,
    "fp16": OSNET_MODEL_PATH_FP16,
    "int8": OSNET_MODEL_PATH_INT8,
}
OSNET_ACTIVE_PATH = _OSNET_PATHS[OSNET_PRECISION]

# Detection Parameters
DETECTION_CONF_THRESH = 0.35
PERSON_CLASS_ID = 0  # YOLO class 0 is 'person'

# ────────────────────────────────────────────────────
# Legacy ByteTrack Parameters — kept only for the A/B comparison path
# (tracker/byte_track.py). Not used by BoT-SORT.
# ────────────────────────────────────────────────────
TRACK_THRESH = 0.4       # Threshold for high-confidence detections
TRACK_BUFFER = 90        # Frames to keep lost tracks before removing (3 seconds @ 30fps)
MATCH_THRESH = 0.85      # Max IoU distance for first association (ByteTrack-specific, NOT reused by BoT-SORT)
LOW_CONF_THRESH = 0.1    # Threshold for low-confidence detections

# ────────────────────────────────────────────────────
# BoT-SORT (BoxMOT) Parameters — active tracker
# NOTE: match_thresh / appearance_thresh below are BoT-SORT's own cost-fusion
# thresholds, NOT the same scale/meaning as MATCH_THRESH above. Retune on your
# own footage — see Phase 1.4 in the migration plan.
# ────────────────────────────────────────────────────
TRACKER_BACKEND = os.getenv("TRACKER_BACKEND", "botsort")  # "botsort" | "bytetrack" (for A/B testing)

BOTSORT_TRACK_BUFFER = int(os.getenv("BOTSORT_TRACK_BUFFER", "90"))      # start = old TRACK_BUFFER value
BOTSORT_MATCH_THRESH = float(os.getenv("BOTSORT_MATCH_THRESH", "0.8"))   # sweep 0.7 / 0.8 / 0.9 on your footage
BOTSORT_APPEARANCE_THRESH = float(os.getenv("BOTSORT_APPEARANCE_THRESH", "0.65"))  # start = old REID_SIMILARITY_THRESH
BOTSORT_CMC_METHOD = "none"  # static ceiling camera — Camera Motion Compensation OFF, saves CPU
BOTSORT_DEVICE = os.getenv("BOTSORT_DEVICE", "cpu")  # Pi 5 has no CUDA
BOTSORT_HALF = False  # keep False until OSNET_PRECISION=="fp16" path is confirmed working end-to-end

# ReID Parameters
REID_SIMILARITY_THRESH = 0.65  # Cosine similarity threshold tau_reid (used by multicam_manager's lazy-ReID step)
REID_FEATURE_DIM = 512
REID_IMAGE_SIZE = (128, 256)  # (width, height) for OSNet input

# Transition Zone & Multi-Camera Parameters
TRANSITION_TIME_WINDOW_SEC = 5.0
OCCLUSION_TIMEOUT_SEC = 3.0

# Polygons for Transition Zones (Normalized coordinates [0.0..1.0])
# Area tangga / border between Camera 1 (Floor 1) and Camera 2 (Floor 2)
DEFAULT_CAMERAS_CONFIG: Dict[str, Dict[str, Any]] = {
    "CAM_1": {
        "name": "CCTV Ceiling - Lantai 1 (Kasir & Tangga)",
        "floor": 1,
        "resolution": (1920, 1080),
        # RTSP URL for CCTV Lantai 1 — webcam for simulation, real CCTV for production
        # Webcam: "rtsp_url": "0" (or integer index)
        # RTSP CCTV: "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream1"
        "rtsp_url": os.getenv("CAM_1_URL", "0"),
        "transition_zones": [
            {
                "zone_id": "TZ_STAIRS_FL1",
                "polygon": [[0.70, 0.60], [0.95, 0.60], [0.95, 0.95], [0.70, 0.95]],
                "target_camera": "CAM_2",
                "target_zone": "TZ_STAIRS_FL2"
            }
        ]
    },
    "CAM_2": {
        "name": "CCTV Ceiling - Lantai 2 (Seating & Tangga)",
        "floor": 2,
        "resolution": (1920, 1080),
        # Webcam: "rtsp_url": "1" (second USB webcam)
        # RTSP CCTV: "rtsp_url": "rtsp://admin:password@192.168.1.101:554/stream1"
        "rtsp_url": os.getenv("CAM_2_URL", "1"),
        "transition_zones": [
            {
                "zone_id": "TZ_STAIRS_FL2",
                "polygon": [[0.05, 0.05], [0.30, 0.05], [0.30, 0.40], [0.05, 0.40]],
                "target_camera": "CAM_1",
                "target_zone": "TZ_STAIRS_FL1"
            }
        ]
    }
}

# ────────────────────────────────────────────────────
# Supabase Configuration (Direct DB Write — No Backend Server Needed)
# Set these in a .env file or as environment variables on Raspberry Pi 5
# ────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

# Sync interval — how often (seconds) to push occupancy count to Supabase
SYNC_INTERVAL_SEC = 1.0