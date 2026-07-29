import unittest
import numpy as np
from multicam.transition_zone import TransitionZone
from multicam.multicam_manager import MultiCamManager
from tracker.byte_track import STrack
from tracker.kalman_filter import KalmanFilter

class TestMultiCam(unittest.TestCase):
    def test_transition_zone_inside(self):
        poly = [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]
        zone = TransitionZone("TZ_TEST", poly, "CAM_2", "TZ_TARGET", resolution=(1920, 1080))

        self.assertTrue(zone.is_inside((960.0, 540.0), normalized=False))
        self.assertFalse(zone.is_inside((100.0, 100.0), normalized=False))

    def test_multicam_lazy_reid_handover(self):
        mgr = MultiCamManager()
        dummy_frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 150
        kf = KalmanFilter()

        # Track in CAM_1 inside transition zone
        track1 = STrack((1500.0, 800.0, 100.0, 200.0), 0.9)
        track1.activate(kf, 1)

        tracks1 = mgr.process_camera_tracks("CAM_1", [track1], dummy_frame)
        gid1 = tracks1[0].global_track_id
        self.assertIsNotNone(gid1)
        self.assertEqual(len(mgr.transition_cache), 1)

        # Track in CAM_2 inside transition zone
        track2 = STrack((200.0, 200.0, 100.0, 200.0), 0.9)
        track2.activate(kf, 1)
        track2.reid_feature = mgr.transition_cache[0].reid_feature.copy()

        tracks2 = mgr.process_camera_tracks("CAM_2", [track2], dummy_frame)
        gid2 = tracks2[0].global_track_id
        self.assertEqual(gid2, gid1)

if __name__ == '__main__':
    unittest.main()
