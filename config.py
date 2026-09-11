import os
from typing import Dict, List, Tuple, Any

# Root Project Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ────────────────────────────────────────────────────
# Detector Backend Selection
# ────────────────────────────────────────────────────
# Switch detectors with one env var — no code changes needed:
#   DETECTOR_BACKEND=yolo11n python process_video.py --video cctv_test.mp4
#   DETECTOR_BACKEND=yolo26n python process_video.py --video cctv_test.mp4
#
# Defaults to yolo26n (NMS-free, better for edge deployment).
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "yolo26n")

DETECTOR_PATHS: Dict[str, Dict[str, str]] = {
    "yolo11n": {
        "pt":   os.path.join(BASE_DIR, "weights", "yolo11n.pt"),
        "onnx": os.path.join(BASE_DIR, "weights", "yolo11n.onnx"),
        "int8": os.path.join(BASE_DIR, "weights", "yolo11n_int8.onnx"),
        "fp16": os.path.join(BASE_DIR, "weights", "yolo11n_fp16.onnx"),
    },
    "yolo26n": {
        "pt":   os.path.join(BASE_DIR, "weights", "yolo26n.pt"),
        "onnx": os.path.join(BASE_DIR, "weights", "yolo26n.onnx"),
        "int8": os.path.join(BASE_DIR, "weights", "yolo26n_int8.onnx"),
        "fp16": os.path.join(BASE_DIR, "weights", "yolo26n_fp16.onnx"),
    },
}

def resolve_detector_path(backend_name: str) -> str:
    """Resolve the best ONNX path (INT8/FP16/FP32) for a given detector backend."""
    b = "yolo26n" if "26" in backend_name.lower() else "yolo11n"
    paths = DETECTOR_PATHS[b]
    if b == "yolo26n":
        # YOLO26n STAL attention head: prefer validated FP16
        if os.path.exists(paths["fp16"]):
            return paths["fp16"]
    else:
        # YOLO11n: prefer validated Conv-INT8, fallback to FP16
        if os.path.exists(paths["int8"]):
            return paths["int8"]
        if os.path.exists(paths["fp16"]):
            return paths["fp16"]
    return paths["onnx"]

# Resolved paths for the active detector backend
YOLO_MODEL_PATH = resolve_detector_path(DETECTOR_BACKEND)
YOLO_PT_PATH    = DETECTOR_PATHS.get(DETECTOR_BACKEND, DETECTOR_PATHS["yolo26n"])["pt"]

# Legacy aliases (detector.py reads ACTIVE_MODEL, YOLO26_MODEL_PATH, YOLO26_PT_PATH)
ACTIVE_MODEL     = "yolo26" if DETECTOR_BACKEND == "yolo26n" else "yolo11"
YOLO26_MODEL_PATH = resolve_detector_path("yolo26n")
YOLO26_PT_PATH    = DETECTOR_PATHS["yolo26n"]["pt"]
YOLO11_MODEL_PATH = resolve_detector_path("yolo11n")
YOLO11_PT_PATH    = DETECTOR_PATHS["yolo11n"]["pt"]

# OSNet ReID weights (used by BotSortTracker and MultiCamManager)
OSNET_MODEL_PATH = os.path.join(BASE_DIR, "weights", "osnet_x0_25.onnx")
OSNET_PT_PATH    = os.path.join(BASE_DIR, "weights", "osnet_x0_25_msmt17.pth")
OSNET_PTH_PATH   = OSNET_PT_PATH  # alias used by reid.py

# Detection Parameters
DETECTION_CONF_THRESH = 0.18
PERSON_CLASS_ID = 0  # YOLO class 0 is 'person'

# ────────────────────────────────────────────────────
# BoT-SORT Tracker Parameters
# ────────────────────────────────────────────────────
# These map directly to BotSort() constructor kwargs in tracker/botsort_tracker.py.
# Tune by setting env vars or editing here; retuning guide in migration plan Phase 1.4.
#
# track_high_thresh: min detection confidence to start / continue a track as "high"
BOTSORT_HIGH_THRESH     = float(os.getenv("BOTSORT_HIGH_THRESH",     "0.25"))
# track_low_thresh: min detection confidence for the second low-conf association pass
BOTSORT_LOW_THRESH      = float(os.getenv("BOTSORT_LOW_THRESH",      "0.05"))
# new_track_thresh: min score for a completely new track to be initialised
BOTSORT_NEW_THRESH      = float(os.getenv("BOTSORT_NEW_THRESH",      "0.25"))
# track_buffer: frames to keep a lost track before deleting it.
# At ~13fps on Pi 5: 150 frames ≈ 11.5s — long enough for seated customers.
BOTSORT_TRACK_BUFFER    = int(os.getenv("BOTSORT_TRACK_BUFFER",      "150"))
# match_thresh: combined motion+appearance cost gate for first association.
# BoT-SORT's match_thresh is NOT the same scale as the old ByteTrack MATCH_THRESH.
# Start at 0.80 (BoxMOT default) and sweep {0.70, 0.80, 0.90} per Phase 1.4.
BOTSORT_MATCH_THRESH    = float(os.getenv("BOTSORT_MATCH_THRESH",    "0.80"))
# proximity_thresh: IoU gate that limits appearance matching to nearby tracks only
BOTSORT_PROXIMITY_THRESH = float(os.getenv("BOTSORT_PROXIMITY_THRESH", "0.5"))
# appearance_thresh: appearance distance gate (1 - cosine_sim).
# 0.25 → requires cosine similarity ≥ 0.75. Sweep {0.20, 0.25, 0.30} per Phase 1.4.
BOTSORT_APPEARANCE_THRESH = float(os.getenv("BOTSORT_APPEARANCE_THRESH", "0.25"))
# second_match_thresh: cost gate for the second association (lost tracks)
BOTSORT_SECOND_MATCH_THRESH = float(os.getenv("BOTSORT_SECOND_MATCH_THRESH", "0.5"))

# ────────────────────────────────────────────────────
# Legacy ByteTrack Parameters (kept for backward compat with existing unit tests)
# ────────────────────────────────────────────────────
TRACK_THRESH     = BOTSORT_HIGH_THRESH
TRACK_BUFFER     = BOTSORT_TRACK_BUFFER
MATCH_THRESH     = 0.65   # original ByteTrack value — kept for test assertions
LOW_CONF_THRESH  = BOTSORT_LOW_THRESH

# ReID Parameters (used by OSNetExtractor in multicam_manager.py)
REID_SIMILARITY_THRESH = 0.62  # Cosine similarity threshold for cross-camera ReID
REID_FEATURE_DIM = 512
REID_IMAGE_SIZE  = (128, 256)  # (width, height) for OSNet input
REID_COST_WEIGHT = 0.55        # kept for any legacy references
REID_EASY_THRESH = 0.30        # kept for any legacy references

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
        # Webcam:   "rtsp_url": "0"  (or integer index)
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
        # Webcam:   "rtsp_url": "1"  (second USB webcam)
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

BACKEND_API_URL = 'http://localhost:8001/api/v1'

# YOLO inference input size.
# 640 is the native resolution for YOLO11n / YOLO26n and optimal for Edge CPU (RPi5 / Laptop).
YOLO_INPUT_SIZE = 640

# Enable 2nd-pass detection on the upper/far region of the frame.
# False (default): 1-pass detection for high throughput on Edge CPU (RPi5 / laptop).
# True: 2-pass detection to maximize recall on distant people (at ~2x compute cost).
ENABLE_FAR_REGION_PASS = False
