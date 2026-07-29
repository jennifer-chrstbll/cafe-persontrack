import unittest
import numpy as np
from models.reid import OSNetExtractor

class TestLazyReID(unittest.TestCase):
    def test_osnet_extractor_similarity(self):
        v1 = np.random.randn(512).astype(np.float32)
        v1 = v1 / np.linalg.norm(v1)

        sim_self = OSNetExtractor.compute_similarity(v1, v1)
        self.assertAlmostEqual(sim_self, 1.0, places=4)

        v2 = np.random.randn(512).astype(np.float32)
        v2 = v2 / np.linalg.norm(v2)
        sim_other = OSNetExtractor.compute_similarity(v1, v2)
        self.assertTrue(-1.0 <= sim_other <= 1.0)

    def test_osnet_crop_and_extract(self):
        extractor = OSNetExtractor()
        dummy_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        bbox = (100.0, 100.0, 200.0, 300.0)

        feat = extractor.extract_feature(dummy_frame, bbox)
        self.assertIsNotNone(feat)
        self.assertEqual(feat.shape, (512,))
        norm = np.linalg.norm(feat)
        self.assertAlmostEqual(norm, 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
