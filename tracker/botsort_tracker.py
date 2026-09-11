"""
tracker/botsort_tracker.py
==========================
BoxMOT BoT-SORT wrapper for cafe-persontrack.

This module wraps boxmot.BotSort behind the same duck-typed interface that
ByteTracker exposed, so pipeline.py, demo_single_cam.py, process_video.py,
rpi_main.py, and multicam_manager.py need only a one-line import swap.

BoxMOT BotSort update() returns a numpy array of shape (N, 8):
    [x1, y1, x2, y2, track_id, conf, cls, det_idx]

We convert each row into a BotSortTrack object that exposes the same
fields that the rest of the codebase reads from STrack:
    .track_id       int    — unique track ID (1-based after conversion)
    .tlbr           array  — [x1, y1, x2, y2]
    .centroid       tuple  — (cx, cy)
    .score          float  — detection confidence
    .velocity_x     float  — pixels/frame (from Kalman state)
    .velocity_y     float  — pixels/frame (from Kalman state)
    .is_activated   bool   — always True for returned tracks (BoxMOT only
                             returns confirmed/active tracks from update())
    .global_track_id Optional[str]  — set by multicam_manager.py
    .visit_id        Optional[str]  — set by multicam_manager.py
    .reid_feature    Optional[np.ndarray]  — set by multicam_manager.py
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any

import config
from models.detector import PersonDetection

logger = logging.getLogger(__name__)

# ─── Suppress the noisy INFO banner that BotSort prints on every instantiation ─
import logging as _logging
_logging.getLogger("boxmot").setLevel(_logging.WARNING)


class BotSortTrack:
    """
    Duck-typed replacement for STrack — exposes the same attributes that
    pipeline.py, multicam_manager.py, and the rendering code access.

    Constructed from one row of the (N, 8) numpy array returned by
    BotSort.update(): [x1, y1, x2, y2, track_id, conf, cls, det_idx]
    plus the matching boxmot STrack object (for velocity extraction).
    """

    __slots__ = (
        "track_id", "_tlbr", "score",
        "velocity_x", "velocity_y",
        "is_activated",
        "global_track_id", "visit_id", "reid_feature",
        # kept for process_video.py's Kalman-predict-on-skip path
        "_boxmot_track",
    )

    def __init__(
        self,
        track_id: int,
        tlbr: np.ndarray,
        score: float,
        velocity_x: float = 0.0,
        velocity_y: float = 0.0,
        boxmot_track: Any = None,
    ):
        self.track_id       = track_id
        self._tlbr          = tlbr.astype(np.float32)
        self.score          = score
        self.velocity_x     = velocity_x
        self.velocity_y     = velocity_y
        self.is_activated   = True
        self.global_track_id: Optional[str] = None
        self.visit_id:        Optional[str] = None
        self.reid_feature:    Optional[np.ndarray] = None
        self._boxmot_track  = boxmot_track

    @property
    def tlbr(self) -> np.ndarray:
        return self._tlbr

    @property
    def tlwh(self) -> np.ndarray:
        """[top, left, width, height] — kept for any code that reads tlwh."""
        x1, y1, x2, y2 = self._tlbr
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    @property
    def centroid(self):
        x1, y1, x2, y2 = self._tlbr
        return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

    def predict(self):
        """
        Forward-project this track's position using the underlying BoxMOT
        Kalman filter for frame-skip frames in process_video.py.
        """
        if self._boxmot_track is not None and hasattr(self._boxmot_track, "predict"):
            self._boxmot_track.predict()
            # Update tlbr from the predicted Kalman mean
            if hasattr(self._boxmot_track, "xyxy"):
                self._tlbr = np.asarray(self._boxmot_track.xyxy, dtype=np.float32)


def _extract_velocity(boxmot_track) -> tuple:
    """
    Extract (velocity_x, velocity_y) from the BoxMOT Kalman state.

    BoxMOT STrack stores the Kalman mean in [x, y, w, h, vx, vy, ...] format
    (indices 4 and 5 are x/y velocity in the xywh state space).
    We convert pixel/frame velocity back to screen coordinates.
    """
    try:
        mean = boxmot_track.mean
        if mean is not None and len(mean) >= 6:
            return float(mean[4]), float(mean[5])
    except Exception:
        pass
    return 0.0, 0.0


class BotSortTracker:
    """
    BoT-SORT tracker wrapping boxmot.BotSort.

    Drop-in replacement for ByteTracker.  All callers that previously did:
        tracker = ByteTracker(camera_id="CAM_1")
        tracks  = tracker.update(detections, frame=frame)

    can now do:
        tracker = BotSortTracker(camera_id="CAM_1")
        tracks  = tracker.update(detections, frame=frame)

    Input:  List[PersonDetection]  (same as ByteTracker)
    Output: List[BotSortTrack]    (duck-types STrack)

    The .tracked_stracks property and .predict_all() method are exposed so
    process_video.py's frame-skip path works without modification.
    """

    def __init__(self, camera_id: str = "CAM_1", fps: float = 13.0):
        from boxmot import BotSort

        self.camera_id = camera_id
        self.fps       = fps

        # Resolve OSNet weights path — BotSort will load them itself
        osnet_path = config.OSNET_MODEL_PATH
        reid_weights = Path(osnet_path) if osnet_path and Path(osnet_path).exists() else None

        if reid_weights is None:
            logger.warning(
                "[BotSortTracker] OSNet ONNX not found at %s — "
                "running BoT-SORT without ReID (motion-only matching).",
                osnet_path,
            )

        self._tracker = BotSort(
            # Detection confidence gates
            track_high_thresh   = config.BOTSORT_HIGH_THRESH,
            track_low_thresh    = config.BOTSORT_LOW_THRESH,
            new_track_thresh    = config.BOTSORT_NEW_THRESH,
            # Track lifecycle
            track_buffer        = config.BOTSORT_TRACK_BUFFER,
            det_thresh          = config.BOTSORT_HIGH_THRESH,
            max_age             = config.BOTSORT_TRACK_BUFFER,
            # Association thresholds
            match_thresh        = config.BOTSORT_MATCH_THRESH,
            proximity_thresh    = config.BOTSORT_PROXIMITY_THRESH,
            appearance_thresh   = config.BOTSORT_APPEARANCE_THRESH,
            second_match_thresh = config.BOTSORT_SECOND_MATCH_THRESH,
            # Camera Motion Compensation — disable for fixed ceiling cameras
            use_cmc             = False,
            # Enable appearance embeddings only when OSNet weights exist
            use_embeddings      = (reid_weights is not None),
            # ReID model
            reid_weights        = reid_weights,
            device              = "cpu",
            half                = False,
            # FPS for Kalman filter timing
            frame_rate          = max(1, int(fps)),
        )

        # Current active BotSortTrack objects — updated each frame
        self._active_tracks: List[BotSortTrack] = []

        logger.info(
            "[BotSortTracker] %s initialised — BoT-SORT / OSNet=%s",
            camera_id,
            reid_weights or "disabled",
        )

    # ─── Public interface ──────────────────────────────────────────────────────

    def update(
        self,
        detections: List[PersonDetection],
        frame: Optional[np.ndarray] = None,
    ) -> List[BotSortTrack]:
        """
        Run one tracking step.

        Parameters
        ----------
        detections : List[PersonDetection]
            Output of PersonDetector.detect(frame).
        frame : np.ndarray | None
            BGR frame — required for ReID appearance extraction inside BotSort.
            Pass None only in unit tests where ReID is not needed.

        Returns
        -------
        List[BotSortTrack]
            Currently active (confirmed) tracks.
        """
        # Build (N, 6) detection array: [x1, y1, x2, y2, conf, cls]
        if detections:
            dets_np = np.array(
                [
                    [d.x1, d.y1, d.x2, d.y2, d.conf, d.class_id]
                    for d in detections
                ],
                dtype=np.float32,
            )
        else:
            dets_np = np.empty((0, 6), dtype=np.float32)

        # BoxMOT requires a real frame (even a black one) — never pass None
        if frame is None:
            # Fall back to a minimal black frame so BotSort doesn't crash
            frame_in = np.zeros((2, 2, 3), dtype=np.uint8)
        else:
            frame_in = frame

        # BotSort.update() → (N, 8): [x1, y1, x2, y2, track_id, conf, cls, det_idx]
        result = self._tracker.update(dets_np, frame_in)

        # Build a fast lookup from track_id → boxmot STrack for velocity extraction
        boxmot_by_id: Dict[int, Any] = {
            int(t.track_id): t for t in self._tracker.active_tracks
        }

        tracks: List[BotSortTrack] = []
        if result is not None and len(result) > 0:
            for row in result:
                x1, y1, x2, y2, tid, conf = (
                    float(row[0]), float(row[1]),
                    float(row[2]), float(row[3]),
                    int(row[4]),  float(row[5]),
                )
                boxmot_track = boxmot_by_id.get(tid)
                vx, vy = _extract_velocity(boxmot_track) if boxmot_track else (0.0, 0.0)
                tracks.append(
                    BotSortTrack(
                        track_id    = tid,
                        tlbr        = np.array([x1, y1, x2, y2], dtype=np.float32),
                        score       = conf,
                        velocity_x  = vx,
                        velocity_y  = vy,
                        boxmot_track= boxmot_track,
                    )
                )

        self._active_tracks = tracks
        return tracks

    @property
    def tracked_stracks(self) -> List[BotSortTrack]:
        """
        Mirrors the ByteTracker.tracked_stracks attribute used by the
        frame-skip path in process_video.py:

            for t in tracker.tracked_stracks:
                t.predict()
            tracks = [t for t in tracker.tracked_stracks if t.is_activated]
        """
        return self._active_tracks

    def predict_all(self):
        """
        Kalman-predict all currently tracked stracks forward one frame.
        Called on skipped frames (frame_skip > 1) in process_video.py.
        """
        for t in self._active_tracks:
            t.predict()
