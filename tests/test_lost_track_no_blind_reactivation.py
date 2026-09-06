import unittest
from unittest.mock import patch
import numpy as np
from models.detector import PersonDetection
from tracker.byte_track import ByteTracker, STrack


class TestLostTrackNoBlindReactivation(unittest.TestCase):
    def test_different_person_in_same_spot_does_not_inherit_stale_track_id(self):
        """Regression test for the severe-undercounting root cause.

        Previously `strack_pool` (used by Stage 2A's appearance-free 'easy'
        IoU match) mixed continuously-tracked AND Lost tracks. A track that
        had been Lost for a long time could be silently re-activated the
        moment ANY new detection landed in a similar screen position, with
        NO appearance check — e.g. a different customer standing at the same
        checkout spot. This caused many distinct real visitors to be merged
        into the same handful of track IDs, undercounting unique people.

        This test forces OSNet's similarity score to be LOW (simulating two
        genuinely different-looking people) and asserts that a Lost track is
        still NOT reactivated just because a new detection lands in the same
        screen position — proving Lost tracks no longer get an
        appearance-free pass through Stage 2A/2B.
        """
        STrack.reset_count()
        tracker = ByteTracker(camera_id="CAM_TEST", fps=15.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (200.0, 150.0, 280.0, 380.0)  # same on-screen spot both times

        # Force every appearance-similarity check to report "very different
        # people" (0.0), regardless of what the real OSNet model would say
        # for these synthetic frames — isolates the association LOGIC from
        # the ReID model's own discriminative quality.
        with patch.object(ByteTracker, "_gallery_sim", return_value=0.0), \
             patch.object(ByteTracker, "_extract", return_value=np.ones(512, dtype=np.float32)):
            det_a = PersonDetection(bbox=bbox, conf=0.9)
            tracks = tracker.update([det_a], frame=frame)
            self.assertEqual(len(tracks), 1)
            person_a_id = tracks[0].track_id

            # Person A leaves: enough missed frames to go clearly Lost.
            for _ in range(10):
                tracker.update([], frame=frame)

            # Person B (forced to look completely different) appears in the
            # exact same spot. New tracks need one extra frame to move from
            # "unconfirmed" to activated (see the Step 4 confirmation gate).
            det_b = PersonDetection(bbox=bbox, conf=0.9)
            tracker.update([det_b], frame=frame)
            tracks = tracker.update([det_b], frame=frame)

        self.assertGreaterEqual(len(tracks), 1)
        person_b_id = tracks[0].track_id
        self.assertNotEqual(
            person_a_id, person_b_id,
            "A new, appearance-verified-different person inherited a stale "
            "Lost track's ID purely from screen-position overlap — Stage "
            "2A/2B must not reactivate Lost tracks without appearance "
            "verification."
        )


if __name__ == "__main__":
    unittest.main()
