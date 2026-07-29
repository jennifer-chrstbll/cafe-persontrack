import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tracker.byte_track import STrack
from models.reid import OSNetExtractor
from multicam.transition_zone import TransitionZone
import config

class TransitionCacheItem:
    """Stores cached ReID feature vector when a track exits a transition zone."""
    def __init__(self, global_track_id: str, camera_id: str, zone_id: str, reid_feature: np.ndarray, timestamp: float, visit_id: Optional[str] = None):
        self.global_track_id = global_track_id
        self.camera_id = camera_id
        self.zone_id = zone_id
        self.reid_feature = reid_feature
        self.timestamp = timestamp
        self.visit_id = visit_id


class MultiCamManager:
    """
    Multi-Camera Manager handling Global Track ID assignment and Lazy ReID execution.
    Enforces On-Demand ReID extraction to preserve Raspberry Pi 5 CPU cycles.
    """
    def __init__(self):
        self.reid_extractor = OSNetExtractor()
        self.transition_zones: Dict[str, List[TransitionZone]] = {}
        
        # Cache for tracks exiting transition zones: list of TransitionCacheItem
        self.transition_cache: List[TransitionCacheItem] = []

        # Counter for generating unified Global Track IDs
        self.global_id_counter = 0

        # Map local (camera_id, local_track_id) -> global_track_id
        self.local_to_global_map: Dict[Tuple[str, int], str] = {}

        # Initialize default camera transition zones from config
        self._init_camera_zones()

    def _init_camera_zones(self):
        """Loads transition zones from config."""
        for cam_id, cam_cfg in config.DEFAULT_CAMERAS_CONFIG.items():
            zones = []
            res = cam_cfg.get("resolution", (1920, 1080))
            for z_cfg in cam_cfg.get("transition_zones", []):
                tz = TransitionZone(
                    zone_id=z_cfg["zone_id"],
                    polygon=z_cfg["polygon"],
                    target_camera=z_cfg["target_camera"],
                    target_zone=z_cfg["target_zone"],
                    resolution=res
                )
                zones.append(tz)
            self.transition_zones[cam_id] = zones

    def generate_global_id(self) -> str:
        """Generates a unique global track ID (e.g. 'GT-1001')."""
        self.global_id_counter += 1
        return f"GT-{self.global_id_counter:04d}"

    def clean_expired_cache(self):
        """Removes items from transition cache older than TRANSITION_TIME_WINDOW_SEC."""
        now = time.time()
        self.transition_cache = [
            item for item in self.transition_cache
            if (now - item.timestamp) <= config.TRANSITION_TIME_WINDOW_SEC
        ]

    def process_camera_tracks(
        self,
        camera_id: str,
        tracks: List[STrack],
        frame: np.ndarray
    ) -> List[STrack]:
        """
        Main multi-camera processing step for a single camera frame:
        1. Assigns Global Track IDs to new tracks.
        2. Evaluates Lazy ReID ONLY when tracks are inside/entering transition zones.
        3. Matches incoming tracks with transition cache from other cameras.
        """
        self.clean_expired_cache()
        cam_zones = self.transition_zones.get(camera_id, [])
        now = time.time()

        for track in tracks:
            map_key = (camera_id, track.track_id)
            
            # If track already has a global_track_id mapped, ensure it is set
            if map_key in self.local_to_global_map:
                track.global_track_id = self.local_to_global_map[map_key]
            else:
                # Check if this new/unmapped track is in any transition zone
                inside_zone = None
                centroid = track.centroid
                for zone in cam_zones:
                    if zone.is_inside(centroid):
                        inside_zone = zone
                        break

                if inside_zone is not None and len(self.transition_cache) > 0:
                    # --- LAZY REID TRIGGER #1: New track in Transition Zone ---
                    # Perform OSNet ReID extraction ON-DEMAND
                    feat = self.reid_extractor.extract_feature(frame, track.tlbr)
                    if feat is not None:
                        track.reid_feature = feat
                        
                        # Compare with cached tracks from expected target camera/zone
                        best_match_id = None
                        best_sim = 0.0
                        best_cache_item = None

                        for item in self.transition_cache:
                            if item.camera_id != camera_id and (now - item.timestamp) <= config.TRANSITION_TIME_WINDOW_SEC:
                                sim = OSNetExtractor.compute_similarity(feat, item.reid_feature)
                                if sim >= config.REID_SIMILARITY_THRESH and sim > best_sim:
                                    best_sim = sim
                                    best_match_id = item.global_track_id
                                    best_cache_item = item

                        if best_match_id is not None and best_cache_item is not None:
                            # ReID match successful! Merge identity cross-camera
                            track.global_track_id = best_match_id
                            track.visit_id = best_cache_item.visit_id
                            self.local_to_global_map[map_key] = best_match_id
                            print(f"[MultiCamManager] ReID MATCH SUCCESS! Track {camera_id}:{track.track_id} -> {best_match_id} (Similarity: {best_sim:.3f})")
                            
                            # Remove matched item from cache
                            if best_cache_item in self.transition_cache:
                                self.transition_cache.remove(best_cache_item)

                # Fallback: If still no global_track_id assigned, assign a new Global ID
                if track.global_track_id is None:
                    gid = self.generate_global_id()
                    track.global_track_id = gid
                    self.local_to_global_map[map_key] = gid

            # --- LAZY REID TRIGGER #2: Track inside Transition Zone -> Cache Feature ---
            # Cache feature for cross-camera handover when track is leaving/inside zone
            for zone in cam_zones:
                if zone.is_inside(track.centroid):
                    # If feature not yet extracted for this track in zone, extract it now
                    if track.reid_feature is None:
                        track.reid_feature = self.reid_extractor.extract_feature(frame, track.tlbr)
                    
                    if track.reid_feature is not None:
                        # Update or add to transition cache
                        existing = [
                            c for c in self.transition_cache
                            if c.global_track_id == track.global_track_id
                        ]
                        if len(existing) == 0:
                            self.transition_cache.append(
                                TransitionCacheItem(
                                    global_track_id=track.global_track_id,
                                    camera_id=camera_id,
                                    zone_id=zone.zone_id,
                                    reid_feature=track.reid_feature,
                                    timestamp=now,
                                    visit_id=track.visit_id
                                )
                            )
                        else:
                            existing[0].timestamp = now
                            existing[0].reid_feature = track.reid_feature

        return tracks
