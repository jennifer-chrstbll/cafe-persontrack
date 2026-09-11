"""
tests/test_botsort_tracker.py
==============================
Smoke tests for BotSortTracker — the BoxMOT BoT-SORT wrapper.

These tests verify the public interface contract:
- update() returns BotSortTrack objects with the right attributes
- tracked_stracks property reflects current active tracks
- predict_all() runs without error
- Multiple detections in one frame get unique track IDs after confirmation
- Empty detections don't crash the tracker
"""
import unittest
import numpy as np

from models.detector import PersonDetection
from tracker.botsort_tracker import BotSortTracker, BotSortTrack


class TestBotSortTracker(unittest.TestCase):
    """Smoke tests for the BotSortTracker wrapper."""

    def setUp(self):
        """Create a fresh tracker for each test."""
        self.tracker = BotSortTracker(camera_id="CAM_TEST", fps=13.0)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # ── Basic interface ──────────────────────────────────────────────────────

    def test_empty_detections_returns_empty_list(self):
        """Tracker must not crash on zero detections."""
        tracks = self.tracker.update([], frame=self.frame)
        self.assertIsInstance(tracks, list)
        self.assertEqual(len(tracks), 0)

    def test_empty_detections_no_frame(self):
        """Tracker must handle frame=None gracefully."""
        tracks = self.tracker.update([], frame=None)
        self.assertIsInstance(tracks, list)

    def test_single_detection_returns_track(self):
        """One high-confidence detection should produce one active track."""
        det = PersonDetection(bbox=(100.0, 100.0, 200.0, 300.0), conf=0.90)
        # BotSort requires min_hits=3 by default before a track is confirmed
        tracks = self.tracker.update([det], frame=self.frame)
        # After one frame it may or may not be returned depending on min_hits —
        # just ensure no crash and that return type is correct.
        self.assertIsInstance(tracks, list)
        for t in tracks:
            self.assertIsInstance(t, BotSortTrack)

    def test_track_attributes_present(self):
        """Returned BotSortTrack must expose all fields the rest of the app reads."""
        det = PersonDetection(bbox=(100.0, 100.0, 200.0, 300.0), conf=0.90)
        # Run 4 frames to confirm the track (min_hits=3)
        for _ in range(4):
            tracks = self.tracker.update([det], frame=self.frame)
        self.assertGreater(len(tracks), 0, "Track should be confirmed after 4 frames")
        t = tracks[0]
        self.assertIsInstance(t.track_id,     int)
        self.assertIsInstance(t.tlbr,         np.ndarray)
        self.assertEqual(t.tlbr.shape,        (4,))
        self.assertIsInstance(t.centroid,      tuple)
        self.assertEqual(len(t.centroid),      2)
        self.assertIsInstance(t.score,         float)
        self.assertIsInstance(t.velocity_x,    float)
        self.assertIsInstance(t.velocity_y,    float)
        self.assertTrue(t.is_activated)
        self.assertIsNone(t.global_track_id)   # set by multicam_manager
        self.assertIsNone(t.visit_id)          # set by multicam_manager
        # reid_feature: None until multicam_manager sets it
        self.assertIsNone(t.reid_feature)

    def test_tlwh_property(self):
        """tlwh property should be derivable from tlbr."""
        det = PersonDetection(bbox=(50.0, 60.0, 150.0, 260.0), conf=0.85)
        for _ in range(4):
            tracks = self.tracker.update([det], frame=self.frame)
        if tracks:
            t = tracks[0]
            tlwh = t.tlwh
            self.assertEqual(tlwh.shape, (4,))
            # x1 == tlwh[0], y1 == tlwh[1]
            np.testing.assert_allclose(tlwh[0], t.tlbr[0], atol=2.0)
            np.testing.assert_allclose(tlwh[1], t.tlbr[1], atol=2.0)
            # width > 0, height > 0
            self.assertGreater(tlwh[2], 0)
            self.assertGreater(tlwh[3], 0)

    def test_centroid_within_bbox(self):
        """Centroid must lie inside the bounding box."""
        det = PersonDetection(bbox=(100.0, 100.0, 200.0, 300.0), conf=0.90)
        for _ in range(4):
            tracks = self.tracker.update([det], frame=self.frame)
        if tracks:
            t = tracks[0]
            cx, cy = t.centroid
            x1, y1, x2, y2 = t.tlbr
            self.assertGreaterEqual(cx, x1 - 5)
            self.assertLessEqual(cx, x2 + 5)
            self.assertGreaterEqual(cy, y1 - 5)
            self.assertLessEqual(cy, y2 + 5)

    # ── Multiple people ──────────────────────────────────────────────────────

    def test_multiple_detections_unique_ids(self):
        """N well-separated people should get N unique track IDs after confirmation."""
        dets = [
            PersonDetection(bbox=(x * 130.0, 100.0, x * 130.0 + 100.0, 300.0), conf=0.90)
            for x in range(4)
        ]
        for _ in range(5):
            tracks = self.tracker.update(dets, frame=self.frame)

        if len(tracks) > 1:
            ids = [t.track_id for t in tracks]
            self.assertEqual(
                len(ids), len(set(ids)),
                f"Duplicate track IDs: {ids}"
            )

    # ── tracked_stracks & predict_all ────────────────────────────────────────

    def test_tracked_stracks_property(self):
        """tracked_stracks must reflect the last update() result."""
        det = PersonDetection(bbox=(100.0, 100.0, 200.0, 300.0), conf=0.90)
        for _ in range(4):
            tracks = self.tracker.update([det], frame=self.frame)
        self.assertIs(self.tracker.tracked_stracks, self.tracker._active_tracks)
        self.assertEqual(len(self.tracker.tracked_stracks), len(tracks))

    def test_predict_all_no_crash(self):
        """predict_all() must not crash whether or not tracks are active."""
        # No tracks yet
        self.tracker.predict_all()  # must not raise

        # With active tracks
        det = PersonDetection(bbox=(100.0, 100.0, 200.0, 300.0), conf=0.90)
        for _ in range(4):
            self.tracker.update([det], frame=self.frame)
        self.tracker.predict_all()  # must not raise

    def test_predict_all_after_frame_skip(self):
        """Frame-skip pattern from process_video.py must work end-to-end."""
        det = PersonDetection(bbox=(100.0, 100.0, 200.0, 300.0), conf=0.90)
        # Establish track
        for _ in range(4):
            self.tracker.update([det], frame=self.frame)

        # Simulate frame skip: predict without a new detection
        self.tracker.predict_all()
        skip_tracks = [t for t in self.tracker.tracked_stracks if t.is_activated]
        self.assertIsInstance(skip_tracks, list)

    # ── Global track ID (set by multicam_manager) ─────────────────────────────

    def test_global_track_id_settable(self):
        """multicam_manager sets global_track_id on track objects in-place."""
        det = PersonDetection(bbox=(100.0, 100.0, 200.0, 300.0), conf=0.90)
        for _ in range(4):
            tracks = self.tracker.update([det], frame=self.frame)
        if tracks:
            tracks[0].global_track_id = "GT-0001"
            self.assertEqual(tracks[0].global_track_id, "GT-0001")

    def test_visit_id_settable(self):
        """multicam_manager sets visit_id on track objects in-place."""
        det = PersonDetection(bbox=(100.0, 100.0, 200.0, 300.0), conf=0.90)
        for _ in range(4):
            tracks = self.tracker.update([det], frame=self.frame)
        if tracks:
            tracks[0].visit_id = "VISIT-42"
            self.assertEqual(tracks[0].visit_id, "VISIT-42")


if __name__ == "__main__":
    unittest.main()
