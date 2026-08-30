import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import numpy as np
from tracker.byte_track import ByteTracker, STrack
from models.detector import PersonDetection

class TestSimultaneousArrival(unittest.TestCase):
    def setUp(self):
        STrack.reset_count()
        self.tracker = ByteTracker(camera_id="TEST_CAM")

    def test_three_people_arrive_same_frame(self):
        """
        Verify that 3 people appearing simultaneously in the exact same frame
        receive 3 DISTINCT track IDs and are not collapsed into 1 ID.
        """
        dets = [
            PersonDetection((50.0, 50.0, 150.0, 250.0), 0.90),
            PersonDetection((250.0, 50.0, 350.0, 250.0), 0.88),
            PersonDetection((450.0, 50.0, 550.0, 250.0), 0.85),
        ]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = self.tracker.update(dets, frame=dummy_frame)

        track_ids = [t.track_id for t in tracks]
        unique_ids = set(track_ids)

        print(f"\n[TestSimultaneousArrival] Detected: {len(tracks)}, Assigned IDs: {track_ids}")
        self.assertEqual(len(tracks), 3, "Expected 3 tracks created for 3 separate people")
        self.assertEqual(len(unique_ids), 3, "All 3 tracks MUST have distinct unique track_ids")

if __name__ == "__main__":
    unittest.main()
