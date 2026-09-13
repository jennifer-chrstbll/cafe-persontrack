"""
tracker/botsort_wrapper.py
---------------------------
Adapts BoxMOT's BotSort (numpy-array in/out) to the STrack-like interface
that pipeline.py and (presumably) multicam_manager.py already expect from
tracker/byte_track.py's STrack class:

    track.track_id        -> int, local per-camera track id
    track.tlbr             -> (x1, y1, x2, y2)
    track.centroid         -> (cx, cy)
    track.velocity_x/_y    -> float, pixels/frame (finite-difference estimate)
    track.score             -> float, detection confidence
    track.global_track_id  -> str | None, settable by multicam_manager

IMPORTANT — VERIFY BEFORE TRUSTING THIS FILE:
This wrapper was written without visibility into your actual STrack class or
multicam_manager.py. If multicam_manager.py reads any other attribute from
the track object (e.g. `.class_id`, `.age`, `.is_activated`, `.mean`/`.covariance`
for a Kalman state, etc.), you MUST add it here too, or multicam_manager.py
will raise an AttributeError at runtime. Grep your multicam_manager.py for
every `track.<something>` access and cross-check against this class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from boxmot import BotSort

import config


class STrack:
    """Lightweight adapter object — one per active track, rebuilt every frame."""

    __slots__ = (
        "track_id", "tlbr", "score", "cls", "global_track_id",
        "_prev_centroid", "velocity_x", "velocity_y",
    )

    def __init__(self, track_id: int, tlbr: Tuple[float, float, float, float],
                 score: float, cls: int,
                 prev_centroid: Optional[Tuple[float, float]] = None):
        self.track_id = int(track_id)
        self.tlbr = tuple(float(v) for v in tlbr)
        self.score = float(score)
        self.cls = int(cls)
        self.global_track_id: Optional[str] = None  # set later by multicam_manager

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


class BotSortTracker:
    """
    Drop-in replacement for tracker.byte_track.ByteTracker, backed by
    BoxMOT's BotSort. One instance per camera (same usage pattern as before:
    `self.trackers: Dict[str, BotSortTracker] = {cam_id: BotSortTracker(camera_id=cam_id) ...}`).
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
        # centroid memory for finite-difference velocity, keyed by track_id
        self._prev_centroids: Dict[int, Tuple[float, float]] = {}

    def update(self, detections: np.ndarray, frame: np.ndarray) -> List[STrack]:
        """
        detections: Nx5 or Nx6 array. If your PersonDetector.detect() only
        returns [x1,y1,x2,y2,conf] (Nx5), we append a class-0 (person) column
        here since BoxMOT expects Nx6 [x1,y1,x2,y2,conf,cls].
        VERIFY: check models/detector.py's PersonDetector.detect() return shape
        and delete this padding step if it already returns Nx6.
        """
        if detections is None or len(detections) == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            dets = np.asarray(detections, dtype=np.float32)
            if dets.shape[1] == 5:
                cls_col = np.full((dets.shape[0], 1), config.PERSON_CLASS_ID, dtype=np.float32)
                dets = np.hstack([dets, cls_col])

        raw = self._tracker.update(dets, frame)  # -> Mx8: x1,y1,x2,y2,id,conf,cls,ind (BoxMOT convention)

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

        # drop stale centroid memory for tracks BoT-SORT has fully removed
        # (it manages its own track_buffer internally; this just prevents
        # this dict from growing unbounded over a long-running process)
        for tid in list(self._prev_centroids.keys()):
            if tid not in seen_ids:
                del self._prev_centroids[tid]

        return tracks