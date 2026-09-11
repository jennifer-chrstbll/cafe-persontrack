import time
import cv2
import numpy as np
from typing import Dict, List, Any, Optional

from models.detector import PersonDetector
from tracker.botsort_tracker import BotSortTracker, BotSortTrack
from multicam.multicam_manager import MultiCamManager
from api_client import CRMBackendClient, sync_telemetry_background
import config

class MultiCamPipeline:
    """
    Multi-Camera Person Detection, BoT-SORT, & Lazy ReID Pipeline Engine.
    Processes video frames from multiple CCTV feeds simultaneously.

    Uses YOLO26n (NMS-free) or YOLO11n for detection (via DETECTOR_BACKEND env var)
    and BoxMOT BoT-SORT with OSNet x0_25 for tracking + ReID.
    """
    def __init__(self, camera_ids: Optional[List[str]] = None, use_onnx: bool = True):
        if camera_ids is None:
            camera_ids = ["CAM_1", "CAM_2"]

        self.camera_ids = camera_ids
        self.detector = PersonDetector(conf_thresh=config.DETECTION_CONF_THRESH, use_onnx=use_onnx)
        self.trackers: Dict[str, BotSortTracker] = {
            cam_id: BotSortTracker(camera_id=cam_id) for cam_id in camera_ids
        }
        self.multicam_manager = MultiCamManager()
        self.backend_client = CRMBackendClient()

        self.last_sync_time: Dict[str, float] = {cam_id: 0.0 for cam_id in camera_ids}

        # Streak debounce — mirrors MIN_HITS logic from process_video.py.
        # A track is only counted as "confirmed" after being seen for MIN_HITS
        # consecutive processed frames. This prevents occupancy from fluctuating
        # on every single missed detection (motion blur, brief occlusion).
        self._MIN_HITS  = 3   # frames seen consecutively before counting as confirmed
        self._MISS_GRACE = 5  # consecutive misses before removing from confirmed
        self._streaks: Dict[str, Dict[int, int]] = {cam_id: {} for cam_id in camera_ids}
        self._misses:  Dict[str, Dict[int, int]] = {cam_id: {} for cam_id in camera_ids}
        self._confirmed: Dict[str, set] = {cam_id: set() for cam_id in camera_ids}

    def process_frame(self, camera_id: str, frame: np.ndarray) -> List[BotSortTrack]:
        """
        Processes a single camera frame:
        1. Runs YOLO26n/YOLO11n person detection (class 0 only).
        2. Updates single-camera BoT-SORT tracker.
        3. Applies Multi-Camera Global ID mapping & Lazy ReID association.
        4. Updates streak debounce (MIN_HITS=3) to stabilize occupancy count.
        5. Telemetry push to backend every SYNC_INTERVAL_SEC.
        """
        if camera_id not in self.trackers:
            self.trackers[camera_id] = BotSortTracker(camera_id=camera_id)
            self._streaks[camera_id]  = {}
            self._misses[camera_id]   = {}
            self._confirmed[camera_id] = set()

        # Step 1: Detect Persons
        detections = self.detector.detect(frame)

        # Step 2: BoT-SORT Single Camera Association
        # Passes frame so BotSort can run OSNet ReID appearance matching internally.
        tracker = self.trackers[camera_id]
        local_tracks = tracker.update(detections, frame=frame)

        # Step 3: Multi-Camera Lazy ReID & Global ID Assignment
        global_tracks = self.multicam_manager.process_camera_tracks(
            camera_id=camera_id,
            tracks=local_tracks,
            frame=frame
        )

        # Step 4: Streak debounce with grace period on both sides.
        # ADD side: need MIN_HITS consecutive frames before counting as confirmed.
        # REMOVE side: need MISS_GRACE consecutive misses before removing from
        # confirmed. A single missed detection (motion blur, brief occlusion) no
        # longer instantly drops the occupancy count.
        streaks   = self._streaks[camera_id]
        misses    = self._misses[camera_id]
        confirmed = self._confirmed[camera_id]
        seen = {t.track_id for t in global_tracks}

        for tid in seen:
            streaks[tid] = streaks.get(tid, 0) + 1
            misses[tid]  = 0  # seen this frame — reset miss counter
            if streaks[tid] >= self._MIN_HITS:
                confirmed.add(tid)

        for tid in list(streaks):
            if tid not in seen:
                misses[tid] = misses.get(tid, 0) + 1
                if misses[tid] >= self._MISS_GRACE:
                    # Missing for MISS_GRACE frames in a row — now safe to remove
                    confirmed.discard(tid)
                    streaks.pop(tid, None)
                    misses.pop(tid, None)

        confirmed_tracks = [t for t in global_tracks if t.track_id in confirmed]

        # Step 5: Background Sync to CRM Backend
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
                for track in confirmed_tracks
            ]
            sync_telemetry_background(self.backend_client, camera_id, floor, tracks_payload)

        return confirmed_tracks

    def draw_tracks(
        self,
        frame: np.ndarray,
        camera_id: str,
        tracks: List[BotSortTrack],
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
        # len(tracks) here is already the debounced confirmed count (from process_frame)
        info_text = f"Cam: {camera_id} | Occupancy: {len(tracks)} | FPS: {fps:.1f}"
        cv2.rectangle(annotated_frame, (0, 0), (w, 35), (20, 20, 20), -1)
        cv2.putText(annotated_frame, info_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        return annotated_frame
