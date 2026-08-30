
import numpy as np
from enum import Enum
from typing import List, Tuple, Optional, Dict

from .kalman_filter import KalmanFilter
from .matching import iou_distance, fused_distance, linear_assignment
from models.detector import PersonDetection
import config


class TrackState(Enum):
    New     = 0
    Tracked = 1
    Lost    = 2
    Removed = 3


class STrack:
    _count = 0

    def __init__(self, tlwh: Tuple[float,float,float,float], score: float):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score

        STrack._count += 1
        self.track_id = STrack._count
        self.state    = TrackState.New

        self.kalman_filter = KalmanFilter()
        self.mean:       Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        self.is_activated  = False
        self.tracklet_len  = 0
        self.frame_id      = 0
        self.start_frame   = 0
        self.velocity_x    = 0.0
        self.velocity_y    = 0.0
        self.reid_feature:    Optional[np.ndarray] = None
        self.global_track_id: Optional[str]        = None
        self.visit_id:        Optional[str]        = None
        self.last_reid_frame: int = 0  # track last frame ReID was extracted

    @classmethod
    def reset_count(cls): cls._count = 0

    def activate(self, kf: KalmanFilter, frame_id: int):
        self.kalman_filter = kf
        self.track_id = STrack._count
        self.mean, self.covariance = kf.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1: self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: 'STrack', frame_id: int, new_id: bool = False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.tracklet_len = 0
        self.state        = TrackState.Tracked
        self.is_activated = True
        self.frame_id     = frame_id
        if new_id:
            STrack._count += 1; self.track_id = STrack._count
        self.score = new_track.score

    def update(self, new_track: 'STrack', frame_id: int):
        self.frame_id     = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.state        = TrackState.Tracked
        self.is_activated = True
        self.score        = new_track.score
        if self.mean is not None and len(self.mean) >= 6:
            self.velocity_x = float(self.mean[4])
            self.velocity_y = float(self.mean[5])

    def predict(self):
        if self.state != TrackState.Tracked: self.mean[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None: return self._tlwh.copy()
        ret = self.mean[:4].copy(); ret[2] *= ret[3]; ret[:2] -= ret[2:]/2.0; return ret

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh.copy(); ret[2:] += ret[:2]; return ret

    @property
    def centroid(self) -> Tuple[float,float]:
        ret = self.tlwh
        return float(ret[0]+ret[2]/2.0), float(ret[1]+ret[3]/2.0)

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:]/2.0; ret[2] /= ret[3]; return ret


from models.reid import OSNetExtractor


class ByteTracker:
    """
    Real-time ByteTracker with fused IoU+ReID cost + lazy ReID updates.

    Key design decisions for speed:
    - ReID extracted at NEW track creation (initial fingerprint)
    - ReID updated every REID_UPDATE_INTERVAL frames per track (not every frame)
    - ReID reconnect only checks lost tracks within REID_MAX_AGE_SEC seconds
    - Fused cost matrix for primary matching prevents cross-path ID switches
    """

    REID_UPDATE_INTERVAL = 10   # Extract new ReID feature every N frames per track
    REID_MAX_AGE_SEC     = 8.0  # Only try ReID reconnect if track lost < this many seconds

    def __init__(self, camera_id: str = 'CAM_1', fps: float = 25.0):
        self.camera_id     = camera_id
        self.fps           = fps
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks:    List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id      = 0
        self.kalman_filter = KalmanFilter()
        self.reid_extractor = OSNetExtractor()
        self.appearance_memory: Dict[int, np.ndarray] = {}

    # ── ReID helpers ─────────────────────────────────────────────────────────

    def _extract(self, frame: np.ndarray, tlbr: np.ndarray) -> Optional[np.ndarray]:
        """Extract OSNet feature; try upper-half crop for head-only detections."""
        feat = self.reid_extractor.extract_feature(frame, tlbr)
        if feat is None:
            x1,y1,x2,y2 = tlbr
            feat = self.reid_extractor.extract_feature(
                frame, np.array([x1, y1, x2, y1+(y2-y1)*0.6]))
        return feat

    def _update_memory(self, tid: int, feat: np.ndarray, alpha: float = 0.70):
        old = self.appearance_memory.get(tid)
        merged = alpha*old + (1-alpha)*feat if old is not None else feat.copy()
        norm = float(np.linalg.norm(merged))
        self.appearance_memory[tid] = merged / max(norm, 1e-6)

    def _get_track_feats(self, tracks: List[STrack]) -> Optional[np.ndarray]:
        feats = [self.appearance_memory.get(t.track_id) for t in tracks]
        if all(f is None for f in feats): return None
        dim = next(f.shape[0] for f in feats if f is not None)
        return np.stack([f if f is not None else np.zeros(dim) for f in feats])

    def _lazy_reid_update(self, frame: Optional[np.ndarray], tracks: List[STrack]):
        """Update ReID memory for tracked tracks every REID_UPDATE_INTERVAL frames."""
        if frame is None: return
        for t in tracks:
            if (self.frame_id - t.last_reid_frame) >= self.REID_UPDATE_INTERVAL:
                feat = self._extract(frame, t.tlbr)
                if feat is not None:
                    self._update_memory(t.track_id, feat)
                    t.reid_feature    = feat
                    t.last_reid_frame = self.frame_id

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, detections: List[PersonDetection],
               frame: Optional[np.ndarray] = None) -> List[STrack]:
        self.frame_id += 1
        activated_stracks = []; refind_stracks = []
        lost_stracks = [];      removed_stracks = []

        # Split by confidence
        high_dets, low_dets = [], []
        for det in detections:
            st = STrack(det.tlwh, det.conf)
            (high_dets if det.conf >= config.TRACK_THRESH else low_dets).append(st)

        # Pre-extract features for HIGH-conf new detections only (for fused cost matrix)
        high_det_feats = None
        if frame is not None and high_dets:
            feats = []
            for d in high_dets:
                f = self._extract(frame, d.tlbr)
                d.reid_feature = f
                feats.append(f if f is not None else np.zeros(config.REID_FEATURE_DIM))
            high_det_feats = np.stack(feats)

        unconfirmed, tracked_stracks = [], []
        for t in self.tracked_stracks:
            (unconfirmed if not t.is_activated else tracked_stracks).append(t)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        for st in strack_pool: st.predict()

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: Fused IoU + ReID Primary Association
        # Fusing ReID into cost matrix prevents ID switch when people cross.
        # reid_weight=0.35 balances speed (less ReID extraction) vs accuracy.
        # ═══════════════════════════════════════════════════════════════════
        reid_w     = getattr(config, 'REID_COST_WEIGHT', 0.35)
        tr_feats   = self._get_track_feats(strack_pool)
        cost2      = fused_distance(strack_pool, high_dets, tr_feats,
                                    high_det_feats, reid_weight=reid_w)
        matches, u_track, u_detection = linear_assignment(cost2, config.MATCH_THRESH)

        for it, id_ in matches:
            track = strack_pool[it]; det = high_dets[id_]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id); activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            if det.reid_feature is not None:
                self._update_memory(track.track_id, det.reid_feature)

        # STEP 3: Low-conf association (IoU only, fast)
        r_tracked = [strack_pool[i] for i in u_track
                     if strack_pool[i].state == TrackState.Tracked]
        matches3, u_track2, _ = linear_assignment(iou_distance(r_tracked, low_dets), 0.50)
        for it, id_ in matches3:
            track = r_tracked[it]; det = low_dets[id_]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id); activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        for i in u_track2:
            t = r_tracked[i]
            if t.state != TrackState.Lost:
                t.state = TrackState.Lost; lost_stracks.append(t)

        # STEP 4: Unconfirmed tracks
        dets_rem = [high_dets[i] for i in u_detection]
        matches4, u_unconf, u_det_rem = linear_assignment(
            iou_distance(unconfirmed, dets_rem), 0.70)
        for it, id_ in matches4:
            unconfirmed[it].update(dets_rem[id_], self.frame_id)
            activated_stracks.append(unconfirmed[it])
        for i in u_unconf:
            t = unconfirmed[i]; t.state = TrackState.Removed; removed_stracks.append(t)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 5: New tracks + ReID re-connect (only check recently-lost tracks)
        # Limiting to REID_MAX_AGE_SEC prevents false matches with old people
        # who left the scene long ago.
        # ═══════════════════════════════════════════════════════════════════
        reid_thresh     = getattr(config, 'REID_SIMILARITY_THRESH', 0.50)
        max_lost_frames = int(self.REID_MAX_AGE_SEC * self.fps)
        recent_lost     = [lt for lt in self.lost_stracks
                           if (self.frame_id - lt.frame_id) <= max_lost_frames]

        for i in u_det_rem:
            det = dets_rem[i]
            if det.score < config.TRACK_THRESH: continue
            reconnected = False

            if frame is not None and det.reid_feature is not None:
                nf = det.reid_feature / max(float(np.linalg.norm(det.reid_feature)), 1e-6)
                best_sim, best_lost = 0.0, None
                for lt in recent_lost:
                    cached = self.appearance_memory.get(lt.track_id)
                    if cached is None: continue
                    sim = float(np.dot(nf, cached))
                    if sim >= reid_thresh and sim > best_sim:
                        best_sim, best_lost = sim, lt

                if best_lost is not None:
                    best_lost.re_activate(det, self.frame_id, new_id=False)
                    self._update_memory(best_lost.track_id, det.reid_feature)
                    refind_stracks.append(best_lost)
                    reconnected = True
                    # Remove from recent_lost to prevent double-match
                    if best_lost in recent_lost: recent_lost.remove(best_lost)

            # Spatial proximity fallback (only when no ReID match found)
            if not reconnected and frame is not None:
                frame_w = frame.shape[1]
                prox    = 180.0 * (frame_w / 1280.0)
                dcx, dcy = det.centroid
                for lt in recent_lost:
                    lcx, lcy = lt.centroid
                    if np.hypot(dcx-lcx, dcy-lcy) <= prox:
                        lt.re_activate(det, self.frame_id, new_id=False)
                        refind_stracks.append(lt)
                        reconnected = True
                        if lt in recent_lost: recent_lost.remove(lt)
                        break

            if not reconnected:
                det.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(det)
                if frame is not None:
                    f = det.reid_feature if det.reid_feature is not None else self._extract(frame, det.tlbr)
                    if f is not None:
                        det.reid_feature = f
                        self._update_memory(det.track_id, f)
                        det.last_reid_frame = self.frame_id

        # STEP 6: Age out
        for t in self.lost_stracks:
            if self.frame_id - t.frame_id > config.TRACK_BUFFER:
                t.state = TrackState.Removed; removed_stracks.append(t)
                self.appearance_memory.pop(t.track_id, None)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks  = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks  = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks     = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks     = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        # Lazy ReID memory refresh for stable tracks (every N frames)
        self._lazy_reid_update(frame, self.tracked_stracks)

        return [t for t in self.tracked_stracks if t.is_activated]


def joint_stracks(a, b):
    exists, res = {}, []
    for t in a: exists[t.track_id]=True; res.append(t)
    for t in b:
        if not exists.get(t.track_id): exists[t.track_id]=True; res.append(t)
    return res

def sub_stracks(a, b):
    d = {t.track_id: t for t in a}
    for t in b: d.pop(t.track_id, None)
    return list(d.values())
