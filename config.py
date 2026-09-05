import os
from typing import Dict, List, Tuple, Any

# Root Project Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Paths
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "weights", "yolo11n.onnx")
YOLO_PT_PATH = os.path.join(BASE_DIR, "weights", "yolo11n.pt")
OSNET_MODEL_PATH = os.path.join(BASE_DIR, "weights", "osnet_x0_25.onnx")
OSNET_PT_PATH = os.path.join(BASE_DIR, "weights", "osnet_x0_25_msmt17.pth")

# Detection Parameters
DETECTION_CONF_THRESH = 0.18
PERSON_CLASS_ID = 0  # YOLO class 0 is 'person'

# ByteTrack Parameters
TRACK_THRESH = 0.25         # Threshold for high-confidence detections
TRACK_BUFFER = 150          # Frames to keep lost tracks before removing.
                             # At 13fps = ~11.5s. Was 250 (~19s) which kept
                             # ghost tracks alive too long -> wrong reconnections.
MATCH_THRESH = 0.65         # Maximum fused cost for primary association. Was 0.60.
                             # Slightly tighter gate reduces wrong matches when
                             # people are at similar positions.
LOW_CONF_THRESH = 0.05      # Threshold for low-confidence detections

# ReID Parameters
REID_SIMILARITY_THRESH = 0.62  # Cosine similarity threshold for lost-track reconnection.
                                # Was 0.50 — too permissive for OSNet x0.25 which can
                                # return sim>0.50 for visually different people.
                                # 0.62 requires stronger appearance match to reconnect.
REID_FEATURE_DIM = 512
REID_IMAGE_SIZE = (128, 256)   # (width, height) for OSNet input

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

# Fused IoU + ReID cost weight (0=pure IoU, 1=pure ReID)
# Was 0.40. Raised to 0.55 so appearance signal (clothes color, texture)
# dominates over position in ambiguous crossing/occlusion situations.
# This directly fixes the "baju jelas beda tapi ID swap" problem.
REID_COST_WEIGHT = 0.55

# 2-Stage primary association gate (Stage 2A IoU-only threshold).
# Detections with IoU cost < this value are matched unambiguously without ReID.
# IoU cost 0.30 = IoU overlap 0.70 — well-separated people almost always hit this.
# Only detections above 0.30 (ambiguous positions) proceed to Stage 2B with ReID.
# Performance impact: ~0 OSNet calls in clear frames; 1-3 only during crossings.
REID_EASY_THRESH = 0.30

# YOLO inference input size.
# 640 is the native resolution for YOLO11n / YOLO26n and optimal for Edge CPU (RPi5 / Laptop).
# 960 or 1280 can be used when dedicated GPU acceleration is available.
YOLO_INPUT_SIZE = 640

# Enable 2nd-pass detection on the upper/far region of the frame.
# False (default): 1-pass detection for high throughput on Edge CPU (RPi5 / laptop).
# True: 2-pass detection to maximize recall on distant people (at ~2x compute cost).
ENABLE_FAR_REGION_PASS = False

# YOLO26 Model Paths (edge-optimized, NMS-free)
YOLO26_MODEL_PATH = os.path.join(BASE_DIR, "weights", "yolo26n.onnx")
YOLO26_PT_PATH    = os.path.join(BASE_DIR, "weights", "yolo26n.pt")

# Active model selector: "yolo26" (NMS-free, default) or "yolo11"
ACTIVE_MODEL = "yolo26"

# PyTorch fallback weight for OSNet
OSNET_PTH_PATH = os.path.join(BASE_DIR, "weights", "osnet_x0_25_msmt17.pth")
