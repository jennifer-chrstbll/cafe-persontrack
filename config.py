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
DETECTION_CONF_THRESH = 0.35
PERSON_CLASS_ID = 0  # YOLO class 0 is 'person'

# ByteTrack Parameters
TRACK_THRESH = 0.4         # Threshold for high-confidence detections
TRACK_BUFFER = 90          # Frames to keep lost tracks before removing (3 seconds)
MATCH_THRESH = 0.85        # Maximum IoU distance for first association
LOW_CONF_THRESH = 0.1      # Threshold for low-confidence detections

# ReID Parameters
REID_SIMILARITY_THRESH = 0.65  # Cosine similarity threshold tau_reid
REID_FEATURE_DIM = 512
REID_IMAGE_SIZE = (128, 256)   # (width, height) for OSNet input

# Transition Zone & Multi-Camera Parameters
# Window time (in seconds) within which a track transition across cameras is matched
TRANSITION_TIME_WINDOW_SEC = 5.0
OCCLUSION_TIMEOUT_SEC = 3.0

# Polygons for Transition Zones (Normalized coordinates [0.0..1.0] or Pixel coordinates)
# Example: Area Tangga / Border between Camera 1 (Floor 1) and Camera 2 (Floor 2)
DEFAULT_CAMERAS_CONFIG: Dict[str, Dict[str, Any]] = {
    "CAM_1": {
        "name": "CCTV Ceiling - Lantai 1 (Kasir & Tangga)",
        "floor": 1,
        "resolution": (1920, 1080),
        "transition_zones": [
            {
                "zone_id": "TZ_STAIRS_FL1",
                "polygon": [[0.70, 0.60], [0.95, 0.60], [0.95, 0.95], [0.70, 0.95]], # Normalized [x, y]
                "target_camera": "CAM_2",
                "target_zone": "TZ_STAIRS_FL2"
            }
        ]
    },
    "CAM_2": {
        "name": "CCTV Ceiling - Lantai 2 (Seating & Tangga)",
        "floor": 2,
        "resolution": (1920, 1080),
        "transition_zones": [
            {
                "zone_id": "TZ_STAIRS_FL2",
                "polygon": [[0.05, 0.05], [0.30, 0.05], [0.30, 0.40], [0.05, 0.40]], # Normalized [x, y]
                "target_camera": "CAM_1",
                "target_zone": "TZ_STAIRS_FL1"
            }
        ]
    }
}

# Backend API Configuration
BACKEND_API_URL = os.getenv("CRM_BACKEND_URL", "http://localhost:8001/api/v1")
SYNC_INTERVAL_SEC = 1.0  # Send track & occupancy updates every 1 second
