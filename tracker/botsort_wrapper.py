"""
tracker/botsort_wrapper.py  (v2 — corrected after reading detector.py / multicam_manager.py / byte_track.py)
---------------------------------------------------------------------------
Adapts BoxMOT's BotSort (numpy-array in/out) to the STrack-like interface
that pipeline.py and multicam_manager.py require:

    track.track_id          -> int, local per-camera track id
    track.tlbr               -> (x1, y1, x2, y2)
    track.centroid           -> (cx, cy)
    track.velocity_x/_y      -> float, pixels/frame
    track.score               -> float, detection confidence
    track.global_track_id    -> str | None, settable by multicam_manager
    track.reid_feature       -> np.ndarray | None   <-- was MISSING in v1, required by multicam_manager.py
    track.visit_id           -> str | None            <-- was MISSING in v1, required by multicam_manager.py

Fixes vs. v1:
  1. detector.py's PersonDetector.detect() returns List[PersonDetection]
     (objects with .bbox/.conf/.tlwh/.centroid), NOT a numpy array.
     This wrapper now converts that list to BoxMOT's expected Nx6 array itself.
  2. STrack now carries .reid_feature and .visit_id so multicam_manager.py's
     transition-zone ReID caching doesn't raise AttributeError.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from boxmot import BotSort

import config


class STrack:
    """Lightweight adapter object — one per active track, rebuilt every frame."""

    __slots__ = (
        "track_id", "tlbr", "score", "cls", "global_track_id",
        "reid_feature", "visit_id", "last_reid_frame",
        "_prev_centroid", "velocity_x", "velocity_y",
    )

    def __init__(self, track_id: int, tlbr: Tuple[float, float, float, float],
                 score: float, cls: int,
                 prev_centroid: Optional[Tuple[float, float]] = None):
        self.track_id = int(track_id)
        self.tlbr = tuple(float(v) for v in tlbr)
        self.score = float(score)
        self.cls = int(cls)

        # Required by multicam_manager.py — mirrors byte_track.py's STrack defaults
        self.global_track_id: Optional[str] = None
        self.reid_feature: Optional[np.ndarray] = None
        self.visit_id: Optional[str] = None
        self.last_reid_frame: int = 0

        cx, cy = self.centroid
        if prev_centroid is not None:
            self.velocity_x = cx - prev_centroid[0]
            self.velocity_y = cy - prev_centroid[1]
        else:
            self.velocity_x = 0.0
            self.velocity_y = 0.0

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.tlbr
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _detections_to_array(detections: List[Any]) -> np.ndarray:
    """
    Converts PersonDetector.detect()'s List[PersonDetection] into the
    Nx6 [x1,y1,x2,y2,conf,cls] float32 array BoxMOT's BotSort.update() expects.

    PersonDetection (models/detector.py) exposes: .x1 .y1 .x2 .y2 .conf .class_id
    """
    if not detections:
        return np.empty((0, 6), dtype=np.float32)

    rows = []
    for d in detections:
        cls_id = getattr(d, "class_id", config.PERSON_CLASS_ID)
        rows.append([d.x1, d.y1, d.x2, d.y2, d.conf, cls_id])
    return np.asarray(rows, dtype=np.float32)


class BotSortTracker:
    """
    Drop-in replacement for tracker.byte_track.ByteTracker, backed by
    BoxMOT's BotSort. One instance per camera.
    """

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._tracker = BotSort(
            reid_weights=Path(config.OSNET_ACTIVE_PATH),
            device=config.BOTSORT_DEVICE,
            half=config.BOTSORT_HALF,
            track_buffer=config.BOTSORT_TRACK_BUFFER,
            match_thresh=config.BOTSORT_MATCH_THRESH,
            appearance_thresh=config.BOTSORT_APPEARANCE_THRESH,
            cmc_method=config.BOTSORT_CMC_METHOD,
        )
        self._prev_centroids: Dict[int, Tuple[float, float]] = {}

    def update(self, detections: List[Any], frame: np.ndarray) -> List[STrack]:
        """
        detections: List[PersonDetection] as returned by
        models.detector.PersonDetector.detect(frame) — converted to BoxMOT's
        expected array format internally.
        """
        dets = _detections_to_array(detections)
        raw = self._tracker.update(dets, frame)  # -> Mx8: x1,y1,x2,y2,id,conf,cls,ind (BoxMOT convention)

        # VERIFY after `pip install boxmot`: run help(BotSort.update) once and
        # confirm this column order matches your installed version.

        tracks: List[STrack] = []
        seen_ids = set()
        if raw is not None and len(raw) > 0:
            for row in raw:
                x1, y1, x2, y2, tid = row[0], row[1], row[2], row[3], int(row[4])
                conf = row[5] if len(row) > 5 else 0.0
                cls = int(row[6]) if len(row) > 6 else config.PERSON_CLASS_ID
                prev_c = self._prev_centroids.get(tid)
                st = STrack(track_id=tid, tlbr=(x1, y1, x2, y2), score=conf, cls=cls, prev_centroid=prev_c)
                self._prev_centroids[tid] = st.centroid
                tracks.append(st)
                seen_ids.add(tid)

        for tid in list(self._prev_centroids.keys()):
            if tid not in seen_ids:
                del self._prev_centroids[tid]

        return tracks