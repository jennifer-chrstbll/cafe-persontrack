import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional

class TransitionZone:
    """
    Represents a designated spatial Transition Zone (ROI polygon).
    Checks if a track's centroid is inside the zone and handles entry/exit events.
    """
    def __init__(self, zone_id: str, polygon: List[List[float]], target_camera: str, target_zone: str, resolution: Tuple[int, int] = (1920, 1080)):
        self.zone_id = zone_id
        self.normalized_polygon = polygon  # List of [x, y] in [0..1]
        self.target_camera = target_camera
        self.target_zone = target_zone
        self.resolution = resolution  # (w, h)

        # Convert normalized polygon to pixel coordinates array
        w, h = resolution
        self.pixel_polygon = np.array([
            [int(pt[0] * w), int(pt[1] * h)] for pt in polygon
        ], dtype=np.int32)

    def is_inside(self, centroid: Tuple[float, float], normalized: bool = False) -> bool:
        """
        Pipes point polygon test to determine if centroid (pos_x, pos_y) is inside polygon.
        """
        cx, cy = centroid
        if normalized:
            w, h = self.resolution
            cx_pix, cy_pix = float(cx * w), float(cy * h)
        else:
            cx_pix, cy_pix = float(cx), float(cy)

        res = cv2.pointPolygonTest(self.pixel_polygon, (cx_pix, cy_pix), measureDist=False)
        return res >= 0

    def draw(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255), alpha: float = 0.3) -> np.ndarray:
        """
        Draws filled translucent polygon ROI on image frame for debug visualization.
        """
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.pixel_polygon], color)
        cv2.polylines(frame, [self.pixel_polygon], isClosed=True, color=color, thickness=2)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Label zone_id
        moments = cv2.moments(self.pixel_polygon)
        if moments["m00"] != 0:
            mcx = int(moments["m10"] / moments["m00"])
            mcy = int(moments["m01"] / moments["m00"])
            cv2.putText(frame, self.zone_id, (mcx - 40, mcy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        return frame
