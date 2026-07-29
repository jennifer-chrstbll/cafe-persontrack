import unittest
import numpy as np
from models.detector import PersonDetection
from tracker.byte_track import ByteTracker, STrack
from tracker.kalman_filter import KalmanFilter

class TestByteTrack(unittest.TestCase):
    def test_kalman_filter_init_predict_update(self):
        kf = KalmanFilter()
        measurement = np.array([100.0, 200.0, 0.5, 100.0], dtype=np.float32)
        mean, cov = kf.initiate(measurement)
        
        self.assertEqual(mean.shape, (8,))
        self.assertEqual(cov.shape, (8, 8))
        
        # Predict next state
        pred_mean, pred_cov = kf.predict(mean, cov)
        self.assertEqual(pred_mean.shape, (8,))
        
        # Update with new measurement
        new_measurement = np.array([102.0, 201.0, 0.5, 100.0], dtype=np.float32)
        upd_mean, upd_cov = kf.update(pred_mean, pred_cov, new_measurement)
        self.assertGreater(upd_mean[0], 100.0)

    def test_byte_tracker_association(self):
        tracker = ByteTracker(camera_id="CAM_1")
        STrack.reset_count()

        # Frame 1: One detection
        det1 = PersonDetection(bbox=(50.0, 50.0, 100.0, 200.0), conf=0.9)
        tracks_f1 = tracker.update([det1])
        self.assertEqual(len(tracks_f1), 1)
        t1_id = tracks_f1[0].track_id

        # Frame 2: Same object slightly shifted
        det2 = PersonDetection(bbox=(52.0, 51.0, 102.0, 201.0), conf=0.88)
        tracks_f2 = tracker.update([det2])
        self.assertEqual(len(tracks_f2), 1)
        self.assertEqual(tracks_f2[0].track_id, t1_id)

        # Frame 3: Low confidence detection (stage 2 association)
        det3 = PersonDetection(bbox=(54.0, 52.0, 104.0, 202.0), conf=0.3)
        tracks_f3 = tracker.update([det3])
        self.assertEqual(len(tracks_f3), 1)
        self.assertEqual(tracks_f3[0].track_id, t1_id)

if __name__ == '__main__':
    unittest.main()
