import os
from typing import Dict, List, Tuple, Any

# Root Project Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ────────────────────────────────────────────────────
# Detector backend switch — flip with env var, no code edit needed
# Usage (PowerShell):  $env:DETECTOR_BACKEND="yolo26n"; python demo_single_cam.py
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

if DETECTOR_BACKEND not in DETECTOR_PATHS:
    raise ValueError(
        f"Unknown DETECTOR_BACKEND='{DETECTOR_BACKEND}'. "
        f"Choose from: {list(DETECTOR_PATHS.keys())}"
    )

# Precision switch — "int8" once you've quantized + validated, else "onnx" (fp32)
DETECTOR_PRECISION = os.getenv("DETECTOR_PRECISION", "onnx")  # "onnx" | "int8"

# Backward-compatible names — some modules may still import these directly.
YOLO_MODEL_PATH = DETECTOR_PATHS[DETECTOR_BACKEND][DETECTOR_PRECISION if DETECTOR_PRECISION in DETECTOR_PATHS[DETECTOR_BACKEND] else "onnx"]
YOLO_PT_PATH = DETECTOR_PATHS[DETECTOR_BACKEND]["pt"]

# ────────────────────────────────────────────────────
# models/detector.py reads THESE specific names (ACTIVE_MODEL, YOLO11_*, YOLO26_*),
# not DETECTOR_BACKEND/DETECTOR_PATHS above. This block wires them together so
# there's one real switch instead of two disconnected ones.
# MUST come after DETECTOR_PRECISION is defined above — that's what broke last time.
# ────────────────────────────────────────────────────
ACTIVE_MODEL = "yolo26" if DETECTOR_BACKEND == "yolo26n" else "yolo11"  # detector.py expects no trailing 'n'

YOLO11_MODEL_PATH = DETECTOR_PATHS["yolo11n"][DETECTOR_PRECISION if DETECTOR_PRECISION in DETECTOR_PATHS["yolo11n"] else "onnx"]
YOLO11_PT_PATH = DETECTOR_PATHS["yolo11n"]["pt"]
YOLO26_MODEL_PATH = DETECTOR_PATHS["yolo26n"][DETECTOR_PRECISION if DETECTOR_PRECISION in DETECTOR_PATHS["yolo26n"] else "onnx"]
YOLO26_PT_PATH = DETECTOR_PATHS["yolo26n"]["pt"]
# NOTE: yolo26n.onnx doesn't exist on disk yet. Setting DETECTOR_BACKEND=yolo26n
# now will trigger detector.py's own built-in fallback-to-YOLO11 (it prints a
# warning and keeps working) until you export yolo26n.onnx per the migration plan.

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
# ────────────────────────────────────────────────────
TRACKER_BACKEND = os.getenv("TRACKER_BACKEND", "botsort")  # "botsort" | "bytetrack" (for A/B testing)

# Used by tracker/botsort_tracker.py (the actual BotSort(**kwargs) call).
# Defaults below match boxmot's own BotSort.__init__ defaults — safe starting
# points, not yet tuned to your footage.
BOTSORT_HIGH_THRESH = float(os.getenv("BOTSORT_HIGH_THRESH", "0.5"))    # track_high_thresh & det_thresh
BOTSORT_LOW_THRESH = float(os.getenv("BOTSORT_LOW_THRESH", "0.1"))      # track_low_thresh
BOTSORT_NEW_THRESH = float(os.getenv("BOTSORT_NEW_THRESH", "0.6"))      # new_track_thresh
BOTSORT_PROXIMITY_THRESH = float(os.getenv("BOTSORT_PROXIMITY_THRESH", "0.5"))
BOTSORT_SECOND_MATCH_THRESH = float(os.getenv("BOTSORT_SECOND_MATCH_THRESH", "0.5"))

BOTSORT_TRACK_BUFFER = int(os.getenv("BOTSORT_TRACK_BUFFER", "90"))
BOTSORT_MATCH_THRESH = float(os.getenv("BOTSORT_MATCH_THRESH", "0.8"))
BOTSORT_APPEARANCE_THRESH = float(os.getenv("BOTSORT_APPEARANCE_THRESH", "0.65"))
BOTSORT_CMC_METHOD = "none"  # NOTE: tracker/botsort_tracker.py hardcodes use_cmc=False directly and
                             # does NOT read this value — kept only for tracker/botsort_wrapper.py's
                             # own (currently unused-by-demo) code path. See PATCH_NOTES for details.
BOTSORT_DEVICE = os.getenv("BOTSORT_DEVICE", "cpu")  # Pi 5 has no CUDA
BOTSORT_HALF = False  # keep False until OSNET_PRECISION=="fp16" path is confirmed working end-to-end

# ReID Parameters
REID_SIMILARITY_THRESH = 0.65  # Cosine similarity threshold tau_reid
REID_FEATURE_DIM = 512
REID_IMAGE_SIZE = (128, 256)  # (width, height) for OSNet input

# Transition Zone & Multi-Camera Parameters
TRANSITION_TIME_WINDOW_SEC = 5.0
OCCLUSION_TIMEOUT_SEC = 3.0

# Polygons for Transition Zones (Normalized coordinates [0.0..1.0])
DEFAULT_CAMERAS_CONFIG: Dict[str, Dict[str, Any]] = {
    "CAM_1": {
        "name": "CCTV Ceiling - Lantai 1 (Kasir & Tangga)",
        "floor": 1,
        "resolution": (1920, 1080),
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
# CRM Backend API (cafe-crm) — required by api_client.py's CRMBackendClient.
# Not defined anywhere before; api_client.py's network calls fail silently
# (return False) if this backend isn't actually running, so a placeholder
# default is safe for local testing without cafe-crm up yet.
# ────────────────────────────────────────────────────
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# ────────────────────────────────────────────────────
# Supabase Configuration
# ────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SYNC_INTERVAL_SEC = 1.0