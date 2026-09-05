import threading
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
    _count_lock = threading.Lock()

    def __init__(self, tlwh: Tuple[float, float, float, float], score: float):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score

        with STrack._count_lock:
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
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.reid_feature: Optional[np.ndarray] = None
        self.global_track_id: Optional[str] = None
        self.visit_id: Optional[str] = None
        self.last_reid_frame: int = 0

    @classmethod
    def reset_count(cls):
        with cls._count_lock:
            cls._count = 0

    def activate(self, kf: KalmanFilter, frame_id: int):
        self.kalman_filter = kf
        # NOTE: self.track_id is ALREADY unique and assigned in __init__.
        # DO NOT reassign self.track_id = STrack._count here, as that causes
        # multiple simultaneous new tracks to overwrite their IDs to the same value!
        self.mean, self.covariance = kf.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: 'STrack', frame_id: int, new_id: bool = False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            with STrack._count_lock:
                STrack._count += 1
                self.track_id = STrack._count
        self.score = new_track.score

    def update(self, new_track: 'STrack', frame_id: int):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        if self.mean is not None and len(self.mean) >= 6:
            self.velocity_x = float(self.mean[4])
            self.velocity_y = float(self.mean[5])

    def predict(self):
        if self.state != TrackState.Tracked and self.mean is not None:
            self.mean[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2.0
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def centroid(self) -> Tuple[float, float]:
        ret = self.tlwh
        return float(ret[0] + ret[2] / 2.0), float(ret[1] + ret[3] / 2.0)

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2.0
        ret[2] /= ret[3]
        return ret


from models.reid import OSNetExtractor


class ByteTracker:
    """
    ByteTracker with fused IoU+ReID cost + lazy ReID updates + thread-safe unique IDs.

    Appearance memory stores up to GALLERY_SIZE embeddings per track (FIFO) so that
    similarity is computed as MAX over all stored samples. This makes re-identification
    robust to 180° body rotation and varying viewpoints.
    """

    REID_UPDATE_INTERVAL = 5   # frames between lazy ReID refreshes (was 10 — halved for faster pose adaptation)
    REID_MAX_AGE_SEC = 8.0
    GALLERY_SIZE = 5           # number of appearance samples stored per track

    def __init__(self, camera_id: str = 'CAM_1', fps: float = 25.0):
        self.camera_id = camera_id
        self.fps = fps
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        self.reid_extractor = OSNetExtractor()
        # appearance_memory: Dict[track_id, List[np.ndarray]]
        # Each track keeps up to GALLERY_SIZE L2-normalized embedding samples (FIFO).
        self.appearance_memory: Dict[int, list] = {}

    def _extract(self, frame: np.ndarray, tlbr: np.ndarray) -> Optional[np.ndarray]:
        feat = self.reid_extractor.extract_feature(frame, tlbr)
        if feat is None:
            x1, y1, x2, y2 = tlbr
            feat = self.reid_extractor.extract_feature(
                frame, np.array([x1, y1, x2, y1 + (y2 - y1) * 0.6]))
        return feat

    def _update_memory(self, tid: int, feat: np.ndarray):
        """Append feat to the per-track gallery (FIFO, max GALLERY_SIZE samples).

        Using a gallery of samples instead of a single EMA vector means:
        - When a person turns 180°, the new back-view embedding is added to the
          gallery alongside the old front-view embedding.
        - Similarity is computed as MAX over all gallery samples (see _gallery_sim),
          so a match to ANY stored view counts as a hit.
        """
        norm = float(np.linalg.norm(feat))
        if norm < 1e-6:
            return
        feat_n = feat / norm
        gallery = self.appearance_memory.get(tid)
        if gallery is None:
            self.appearance_memory[tid] = [feat_n]
        else:
            gallery.append(feat_n)
            if len(gallery) > self.GALLERY_SIZE:
                gallery.pop(0)  # evict oldest

    def _gallery_sim(self, tid: int, feat_n: np.ndarray) -> float:
        """Max cosine similarity between feat_n and any sample in the gallery for tid."""
        gallery = self.appearance_memory.get(tid)
        if not gallery:
            return 0.0
        return float(max(np.dot(feat_n, s) for s in gallery))

    def _get_track_feats(self, tracks: List[STrack]) -> Optional[np.ndarray]:
        """Return representative embedding per track (mean of gallery samples)."""
        feats = []
        for t in tracks:
            gallery = self.appearance_memory.get(t.track_id)
            if gallery:
                # Mean of gallery samples as representative vector
                mean_feat = np.mean(gallery, axis=0).astype(np.float32)
                norm = float(np.linalg.norm(mean_feat))
                feats.append(mean_feat / max(norm, 1e-6))
            else:
                feats.append(None)
        if all(f is None for f in feats):
            return None
        dim = next(f.shape[0] for f in feats if f is not None)
        return np.stack([f if f is not None else np.zeros(dim) for f in feats])

    def _lazy_reid_update(self, frame: Optional[np.ndarray], tracks: List[STrack]):
        if frame is None:
            return
        for t in tracks:
            if (self.frame_id - t.last_reid_frame) >= self.REID_UPDATE_INTERVAL:
                feat = self._extract(frame, t.tlbr)
                if feat is not None:
                    self._update_memory(t.track_id, feat)
                    t.reid_feature = feat
                    t.last_reid_frame = self.frame_id

    def update(self, detections: List[PersonDetection],
               frame: Optional[np.ndarray] = None) -> List[STrack]:
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Split by confidence
        high_dets, low_dets = [], []
        for det in detections:
            st = STrack(det.tlwh, det.conf)
            (high_dets if det.conf >= config.TRACK_THRESH else low_dets).append(st)
        unconfirmed, tracked_stracks = [], []
        for t in self.tracked_stracks:
            (unconfirmed if not t.is_activated else tracked_stracks).append(t)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        for st in strack_pool:
            st.predict()

        # Step 2: 2-Stage Primary Association
        # ────────────────────────────────────────────────────────────────────
        # WHY: The old approach extracted ReID (OSNet) for EVERY high-conf
        # detection EVERY frame — 5 people = 5× OSNet ≈ 1.3s → 1-2 FPS.
        #
        # Stage 2A — IoU-only for UNAMBIGUOUS matches (zero ReID cost)
        # A detection is "unambiguous" if it clearly overlaps one specific
        # track and no other: IoU cost < EASY_THRESH (i.e., IoU > 0.70).
        # This covers ~80-95% of frames where people are well-separated.
        #
        # Stage 2B — Fused IoU+ReID for AMBIGUOUS subset only
        # Only the tracks/detections that 2A couldn't cleanly resolve go here.
        # Typical count: 0 in clear frames, 1-3 during crossings/occlusions.
        # ReID extraction happens ONLY for this tiny subset.
        # ────────────────────────────────────────────────────────────────────
        easy_thresh = getattr(config, 'REID_EASY_THRESH', 0.30)
        reid_w      = getattr(config, 'REID_COST_WEIGHT',  0.55)

        # Stage 2A: pure IoU matching — no OSNet, very fast
        iou_cost_a = iou_distance(strack_pool, high_dets)
        matches_a, u_track_a, u_det_a = linear_assignment(iou_cost_a, easy_thresh)

        for it, id_ in matches_a:
            track = strack_pool[it]
            det   = high_dets[id_]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            # Appearance memory refreshed lazily by _lazy_reid_update, not here

        # Stage 2B: fused IoU+ReID for ambiguous remainder only
        ambig_tracks = [strack_pool[i] for i in u_track_a]
        ambig_dets   = [high_dets[i]   for i in u_det_a]

        # Extract ReID ONLY for ambiguous detections (often zero!)
        if frame is not None and ambig_dets:
            for d in ambig_dets:
                d.reid_feature = self._extract(frame, d.tlbr)
            ambig_det_feats = np.stack([
                d.reid_feature if d.reid_feature is not None
                else np.zeros(config.REID_FEATURE_DIM)
                for d in ambig_dets
            ])
        else:
            ambig_det_feats = None

        if ambig_tracks and ambig_dets:
            tr_feats_b = self._get_track_feats(ambig_tracks)
            cost_b = fused_distance(ambig_tracks, ambig_dets,
                                    tr_feats_b, ambig_det_feats, reid_weight=reid_w)
            matches_b, u_track_b, u_det_b = linear_assignment(cost_b, config.MATCH_THRESH)

            for it, id_ in matches_b:
                track = ambig_tracks[it]
                det   = ambig_dets[id_]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
                if det.reid_feature is not None:
                    self._update_memory(track.track_id, det.reid_feature)

            # Map back to original strack_pool / high_dets index spaces
            u_track     = [u_track_a[i] for i in u_track_b]
            u_detection = [u_det_a[i]   for i in u_det_b]
        else:
            u_track     = list(u_track_a)
            u_detection = list(u_det_a)

        # Step 3: Low-conf association (IoU only)
        r_tracked = [strack_pool[i] for i in u_track
                     if strack_pool[i].state == TrackState.Tracked]
        matches3, u_track2, _ = linear_assignment(iou_distance(r_tracked, low_dets), 0.50)
        for it, id_ in matches3:
            track = r_tracked[it]
            det = low_dets[id_]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        for i in u_track2:
            t = r_tracked[i]
            if t.state != TrackState.Lost:
                t.state = TrackState.Lost
                lost_stracks.append(t)

        # Step 4: Unconfirmed tracks
        dets_rem = [high_dets[i] for i in u_detection]
        matches4, u_unconf, u_det_rem = linear_assignment(
            iou_distance(unconfirmed, dets_rem), 0.70)
        for it, id_ in matches4:
            unconfirmed[it].update(dets_rem[id_], self.frame_id)
            activated_stracks.append(unconfirmed[it])
        for i in u_unconf:
            t = unconfirmed[i]
            t.state = TrackState.Removed
            removed_stracks.append(t)

        # Step 5: New tracks & ReID reconnect
        reid_thresh = getattr(config, 'REID_SIMILARITY_THRESH', 0.50)
        max_lost_frames = int(self.REID_MAX_AGE_SEC * self.fps)
        recent_lost = [lt for lt in self.lost_stracks
                       if (self.frame_id - lt.frame_id) <= max_lost_frames]

        for i in u_det_rem:
            det = dets_rem[i]
            if det.score < config.TRACK_THRESH:
                continue
            reconnected = False

            if frame is not None:
                if det.reid_feature is None and recent_lost:
                    det.reid_feature = self._extract(frame, det.tlbr)

                best_sim, best_lost = 0.0, None
                if det.reid_feature is not None:
                    nf = det.reid_feature / max(float(np.linalg.norm(det.reid_feature)), 1e-6)
                    for lt in recent_lost:
                        sim = self._gallery_sim(lt.track_id, nf)
                        if sim >= reid_thresh and sim > best_sim:
                            best_sim, best_lost = sim, lt

                if best_lost is not None:
                    best_lost.re_activate(det, self.frame_id, new_id=False)
                    self._update_memory(best_lost.track_id, det.reid_feature)
                    refind_stracks.append(best_lost)
                    reconnected = True
                    if best_lost in recent_lost:
                        recent_lost.remove(best_lost)

            # Spatial proximity fallback — only if no ReID match found.
            # FIX: Radius shrunk from 180px to 80px (scaled by frame width).
            # FIX: Require minimum ReID similarity >= 0.30 as guard — prevents
            # identity swaps in crowded areas where two different people happen
            # to be close to each other. Pure-distance fallback caused the
            # "berbalik badan dianggap orang lain" bug in busy scenes.
            if not reconnected and frame is not None:
                frame_w = frame.shape[1]
                prox = 80.0 * (frame_w / 1280.0)  # was 180.0 — too permissive
                fallback_reid_min = 0.30  # loose but non-zero appearance guard
                dcx, dcy = det.centroid
                for lt in recent_lost:
                    lcx, lcy = lt.centroid
                    if np.hypot(dcx - lcx, dcy - lcy) <= prox:
                        # Check minimum appearance similarity before committing
                        if det.reid_feature is not None:
                            nf_fb = det.reid_feature / max(float(np.linalg.norm(det.reid_feature)), 1e-6)
                            fb_sim = self._gallery_sim(lt.track_id, nf_fb)
                            if fb_sim < fallback_reid_min:
                                continue  # appearance too different — skip this lost track
                        lt.re_activate(det, self.frame_id, new_id=False)
                        refind_stracks.append(lt)
                        reconnected = True
                        if lt in recent_lost:
                            recent_lost.remove(lt)
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

        # Step 6: Age out
        for t in self.lost_stracks:
            if self.frame_id - t.frame_id > config.TRACK_BUFFER:
                t.state = TrackState.Removed
                removed_stracks.append(t)
                self.appearance_memory.pop(t.track_id, None)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        self._lazy_reid_update(frame, self.tracked_stracks)

        return [t for t in self.tracked_stracks if t.is_activated]


def joint_stracks(a, b):
    exists, res = {}, []
    for t in a:
        exists[t.track_id] = True
        res.append(t)
    for t in b:
        if not exists.get(t.track_id):
            exists[t.track_id] = True
            res.append(t)
    return res


def sub_stracks(a, b):
    d = {t.track_id: t for t in a}
    for t in b:
        d.pop(t.track_id, None)
    return list(d.values())
