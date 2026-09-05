import unittest
import numpy as np
from models.detector import PersonDetection
from tracker.byte_track import ByteTracker, STrack


class TestEdgeExtractionCrash(unittest.TestCase):
    def test_degenerate_bbox_with_recent_lost_does_not_crash(self):
        """Regression test: Step 5 reconnect must not crash with
        UnboundLocalError when OSNet feature extraction returns None
        (e.g. a detection bbox <10px wide/tall after clipping to the
        frame edge — a normal occurrence for an entrance camera) while
        another track is present in recent_lost.

        Previously `nf` was referenced unconditionally in the similarity
        loop but only assigned when extraction succeeded, crashing the
        whole tracker/camera thread whenever extraction failed.
        """
        STrack.reset_count()
        tracker = ByteTracker(camera_id="CAM_TEST", fps=15.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Person A appears and gets tracked, then disappears (-> lost_stracks)
        det_a = PersonDetection(bbox=(100.0, 100.0, 180.0, 300.0), conf=0.9)
        tracker.update([det_a], frame=frame)
        for _ in range(3):
            tracker.update([], frame=frame)
        self.assertGreaterEqual(len(tracker.lost_stracks), 1)

        # A new, degenerate detection at the frame edge (only 4px wide) —
        # extract_crop() will return None for this, so reid_feature stays None.
        det_edge = PersonDetection(bbox=(635.0, 100.0, 639.0, 300.0), conf=0.9)
        try:
            tracks = tracker.update([det_edge], frame=frame)
        except UnboundLocalError as e:
            self.fail(f"Tracker crashed on degenerate edge bbox: {e}")

        self.assertGreaterEqual(len(tracks), 0)


if __name__ == "__main__":
    unittest.main()
