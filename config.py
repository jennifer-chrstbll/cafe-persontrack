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
TRACK_BUFFER = 250          # Frames to keep lost tracks before removing (3 seconds)
MATCH_THRESH = 0.60        # Maximum IoU distance for first association
LOW_CONF_THRESH = 0.05      # Threshold for low-confidence detections

# ReID Parameters
REID_SIMILARITY_THRESH = 0.50  # Cosine similarity threshold tau_reid
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
REID_COST_WEIGHT = 0.40

# YOLO inference input size. Larger = better for small/far people.
# 1280 recommended for laptop. 640 for Raspberry Pi 5 (use --skip 3).
YOLO_INPUT_SIZE = 1280

# YOLO26 Model Paths (new, edge-optimized, NMS-free)
YOLO26_MODEL_PATH = os.path.join(BASE_DIR, "weights", "yolo26n.onnx")
YOLO26_PT_PATH    = os.path.join(BASE_DIR, "weights", "yolo26n.pt")

# Active model selector: "yolo11" or "yolo26"
ACTIVE_MODEL = "yolo11"  # Switch to "yolo11" to revert
