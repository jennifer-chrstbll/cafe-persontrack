import numpy as np
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any

from .kalman_filter import KalmanFilter
from .matching import iou_distance, linear_assignment
from models.detector import PersonDetection
import config

class TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """Single Track representation in ByteTrack."""
    _count = 0

    def __init__(self, tlwh: Tuple[float, float, float, float], score: float):
        # Top-left x, top-left y, width, height
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score

        STrack._count += 1
        self.track_id = STrack._count
        self.state = TrackState.New

        self.kalman_filter = KalmanFilter()
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        self.is_activated = False
        self.tracklet_len = 0
        self.frame_id = 0
        self.start_frame = 0
        
        # Velocity vectors (from Kalman filter state)
        self.velocity_x = 0.0
        self.velocity_y = 0.0

        # Custom fields for ReID & Identity Association
        self.reid_feature: Optional[np.ndarray] = None
        self.global_track_id: Optional[str] = None  # Unified ID across multi-cam
        self.visit_id: Optional[str] = None

    @classmethod
    def reset_count(cls):
        cls._count = 0

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        """Starts a new track."""
        self.kalman_filter = kalman_filter
        self.track_id = STrack._count
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: 'STrack', frame_id: int, new_id: bool = False):
        """Re-activates a lost track with a new detection."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            STrack._count += 1
            self.track_id = STrack._count
        self.score = new_track.score

    def update(self, new_track: 'STrack', frame_id: int):
        """Updates tracked state with a matched detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

        # Extract velocities from Kalman state: mean = [cx, cy, a, h, vx, vy, va, vh]
        if self.mean is not None and len(self.mean) >= 6:
            self.velocity_x = float(self.mean[4])
            self.velocity_y = float(self.mean[5])

    def predict(self):
        """Kalman filter state prediction step."""
        if self.state != TrackState.Tracked:
            self.mean[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    @property
    def tlwh(self) -> np.ndarray:
        """Top-left x, top-left y, width, height."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2.0
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """Top-left x, top-left y, bottom-right x, bottom-right y."""
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    @property
    def centroid(self) -> Tuple[float, float]:
        """Centroid (pos_x, pos_y)."""
        ret = self.tlwh
        return float(ret[0] + ret[2] / 2.0), float(ret[1] + ret[3] / 2.0)

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box from [x, y, w, h] to [cx, cy, aspect_ratio, h]."""
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2.0
        ret[2] /= ret[3]
        return ret


from models.reid import OSNetExtractor

class ByteTracker:
    """
    ByteTrack implementation for single camera tracking.
    Uses two-stage association (high-confidence & low-confidence detections)
    enhanced with OSNet ReID Local Appearance Persistence.
    """

    def __init__(self, camera_id: str = "CAM_1"):
        self.camera_id = camera_id
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        self.reid_extractor = OSNetExtractor()
        self.appearance_memory: Dict[int, np.ndarray] = {}

    def update(self, detections: List[PersonDetection], frame: Optional[np.ndarray] = None) -> List[STrack]:
        """
        Processes new detections and updates track states.
        Returns list of active tracked STracks.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Split detections into high and low confidence
        high_detections = []
        low_detections = []
        for det in detections:
            strack = STrack(det.tlwh, det.conf)
            if det.conf >= config.TRACK_THRESH:
                high_detections.append(strack)
            elif det.conf >= config.LOW_CONF_THRESH:
                low_detections.append(strack)

        # Separate tracked and lost tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Step 1: Predict new locations of existing tracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        for strack in strack_pool:
            strack.predict()

        # Step 2: First association with high confidence detections
        dists = iou_distance(strack_pool, high_detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=config.MATCH_THRESH)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = high_detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 3: Second association with low confidence detections (handling occlusion/low conf)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_stracks, low_detections)
        matches, u_track_second, u_detection_second = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = low_detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Mark unmatched tracked tracks as lost
        for i in u_track_second:
            track = r_tracked_stracks[i]
            if track.state != TrackState.Lost:
                track.state = TrackState.Lost
                lost_stracks.append(track)

        # Step 4: Deal with unconfirmed tracks
        detections_rem = [high_detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections_rem)
        matches, u_unconfirmed, u_detection_rem = linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_rem[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for i in u_unconfirmed:
            track = unconfirmed[i]
            track.state = TrackState.Removed
            removed_stracks.append(track)

        # Step 5: Init new tracks for unmatched high confidence detections
        for i in u_detection_rem:
            track = detections_rem[i]
            if track.score >= config.TRACK_THRESH:
                reconnected = False

                # 5a. Robust OSNet ReID Appearance & Upper-Body Matching (Handles Cloth / Pillar / Queue Occlusions)
                if frame is not None:
                    det_feat = self.reid_extractor.extract_feature(frame, track.tlbr)
                    det_upper_feat = self.reid_extractor.extract_upper_body_feature(frame, track.tlbr)
                    
                    if det_feat is not None:
                        track.reid_feature = det_feat
                        best_sim = 0.0
                        best_lost_track = None

                        # Check against lost tracks AND active tracks memory
                        candidate_pool = list(self.lost_stracks)
                        for lost_track in candidate_pool:
                            if lost_track.track_id in self.appearance_memory:
                                cached_feat = self.appearance_memory[lost_track.track_id]
                                sim_full = OSNetExtractor.compute_similarity(det_feat, cached_feat)
                                sim_upper = OSNetExtractor.compute_similarity(det_upper_feat, cached_feat) if det_upper_feat is not None else 0.0
                                sim = max(sim_full, sim_upper)

                                # Adaptive occlusion recovery threshold (0.55)
                                if sim >= 0.55 and sim > best_sim:
                                    best_sim = sim
                                    best_lost_track = lost_track

                        if best_lost_track is not None:
                            best_lost_track.re_activate(track, self.frame_id, new_id=False)
                            best_lost_track.reid_feature = det_feat
                            
                            # EMA (Exponential Moving Average) Memory Update for Master Appearance Signature
                            old_feat = self.appearance_memory[best_lost_track.track_id]
                            new_ema = 0.70 * old_feat + 0.30 * det_feat
                            new_ema = new_ema / max(1e-5, float(np.linalg.norm(new_ema)))
                            self.appearance_memory[best_lost_track.track_id] = new_ema
                            
                            refind_stracks.append(best_lost_track)
                            reconnected = True
                            print(f"[ByteTrack] Occlusion Recovery! Track #{best_lost_track.track_id} preserved after occlusion (Sim: {best_sim:.3f})")

                # 5b. Spatial Centroid Proximity Fallback
                if not reconnected:
                    det_cx, det_cy = track.centroid
                    for lost_track in self.lost_stracks:
                        lost_cx, lost_cy = lost_track.centroid
                        dist = np.hypot(det_cx - lost_cx, det_cy - lost_cy)
                        if dist <= 240.0:  # Spatial range
                            lost_track.re_activate(track, self.frame_id, new_id=False)
                            refind_stracks.append(lost_track)
                            reconnected = True
                            break

                if not reconnected:
                    track.activate(self.kalman_filter, self.frame_id)
                    activated_stracks.append(track)
                    if frame is not None and track.reid_feature is None:
                        track.reid_feature = self.reid_extractor.extract_feature(frame, track.tlbr)
                    if track.reid_feature is not None:
                        self.appearance_memory[track.track_id] = track.reid_feature

        # Step 6: Update lost / removed state buffer
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > config.TRACK_BUFFER:
                track.state = TrackState.Removed
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks


def joint_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = True
        res.append(t)
    for t in tlistb:
        if not exists.get(t.track_id, False):
            exists[t.track_id] = True
            res.append(t)
    return res


def sub_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        stracks.pop(t.track_id, None)
    return list(stracks.values())
