import time
import cv2
import numpy as np
from typing import Dict, List, Any, Optional

from models.detector import PersonDetector
from tracker.byte_track import ByteTracker, STrack
from multicam.multicam_manager import MultiCamManager
from api_client import CRMBackendClient, sync_telemetry_background
import config

class MultiCamPipeline:
    """
    Multi-Camera Person Detection, ByteTrack, & Lazy ReID Pipeline Engine.
    Processes video frames from multiple CCTV feeds simultaneously.
    """
    def __init__(self, camera_ids: Optional[List[str]] = None, use_onnx: bool = True):
        if camera_ids is None:
            camera_ids = ["CAM_1", "CAM_2"]
            
        self.camera_ids = camera_ids
        self.detector = PersonDetector(conf_thresh=config.DETECTION_CONF_THRESH, use_onnx=use_onnx)
        self.trackers: Dict[str, ByteTracker] = {
            cam_id: ByteTracker(camera_id=cam_id) for cam_id in camera_ids
        }
        self.multicam_manager = MultiCamManager()
        self.backend_client = CRMBackendClient()

        self.last_sync_time: Dict[str, float] = {cam_id: 0.0 for cam_id in camera_ids}

    def process_frame(self, camera_id: str, frame: np.ndarray) -> List[STrack]:
        """
        Processes a single camera frame:
        1. Runs YOLO11n person detection (class 0 only).
        2. Updates single-camera ByteTrack tracker.
        3. Applies Multi-Camera Global ID mapping & Lazy ReID association.
        4. Telemetry push to backend every SYNC_INTERVAL_SEC.
        """
        if camera_id not in self.trackers:
            self.trackers[camera_id] = ByteTracker(camera_id=camera_id)

        # Step 1: Detect Persons
        detections = self.detector.detect(frame)

        # Step 2: ByteTrack Single Camera Association
        tracker = self.trackers[camera_id]
        local_tracks = tracker.update(detections)

        # Step 3: Multi-Camera Lazy ReID & Global ID Assignment
        global_tracks = self.multicam_manager.process_camera_tracks(
            camera_id=camera_id,
            tracks=local_tracks,
            frame=frame
        )

        # Step 4: Background Sync to CRM Backend
        now = time.time()
        if (now - self.last_sync_time.get(camera_id, 0.0)) >= config.SYNC_INTERVAL_SEC:
            self.last_sync_time[camera_id] = now
            cam_cfg = config.DEFAULT_CAMERAS_CONFIG.get(camera_id, {})
            floor = cam_cfg.get("floor", 1)

            tracks_payload = [
                {
                    "camera_id": camera_id,
                    "raw_track_id": track.global_track_id or f"LOCAL-{track.track_id}",
                    "pos_x": round(track.centroid[0], 2),
                    "pos_y": round(track.centroid[1], 2),
                    "velocity_x": round(track.velocity_x, 2),
                    "velocity_y": round(track.velocity_y, 2),
                    "status": "ACTIVE"
                }
                for track in global_tracks
            ]
            sync_telemetry_background(self.backend_client, camera_id, floor, tracks_payload)

        return global_tracks

    def draw_tracks(
        self,
        frame: np.ndarray,
        camera_id: str,
        tracks: List[STrack],
        fps: float = 0.0
    ) -> np.ndarray:
        """
        Visualization overlay for debugging & demo:
        - Draws transition zones.
        - Bounding boxes, Global Track IDs, and velocity vectors.
        - Occupancy counter & FPS badge.
        """
        annotated_frame = frame.copy()
        
        # 1. Draw Transition Zones
        cam_zones = self.multicam_manager.transition_zones.get(camera_id, [])
        for zone in cam_zones:
            annotated_frame = zone.draw(annotated_frame)

        # 2. Draw Track Bounding Boxes
        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.tlbr]
            cx, cy = [int(v) for v in track.centroid]
            gid = track.global_track_id or f"ID-{track.track_id}"

            # Distinct color per track ID
            color_hash = hash(gid) & 0xFFFFFF
            color = (color_hash & 0xFF, (color_hash >> 8) & 0xFF, (color_hash >> 16) & 0xFF)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)

            # Draw velocity vector line
            vx_end = int(cx + track.velocity_x * 5)
            vy_end = int(cy + track.velocity_y * 5)
            cv2.line(annotated_frame, (cx, cy), (vx_end, vy_end), (255, 0, 0), 2)

            # Text label
            label = f"{gid} ({track.score:.2f})"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + t_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 3. Draw Header Stats Badge
        h, w = annotated_frame.shape[:2]
        info_text = f"Cam: {camera_id} | Occupancy: {len(tracks)} | FPS: {fps:.1f}"
        cv2.rectangle(annotated_frame, (0, 0), (w, 35), (20, 20, 20), -1)
        cv2.putText(annotated_frame, info_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        return annotated_frame
