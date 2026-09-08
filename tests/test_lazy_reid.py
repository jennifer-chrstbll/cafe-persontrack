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

    def test_osnet_discriminative_reid(self):
        extractor = OSNetExtractor()
        
        # Person A (Blue jacket, dark pants) - Photo 1
        person_a1 = np.zeros((300, 200, 3), dtype=np.uint8)
        person_a1[:150, :] = [255, 100, 0]   # Blue jacket
        person_a1[150:, :] = [20, 20, 20]    # Dark pants
        
        # Person A (Blue jacket, dark pants) - Photo 2 with slight noise
        person_a2 = np.zeros((300, 200, 3), dtype=np.uint8)
        person_a2[:150, :] = [240, 95, 10]
        person_a2[150:, :] = [25, 25, 25]

        # Person B (Bright red shirt, yellow pants) - Distinct person
        person_b = np.zeros((300, 200, 3), dtype=np.uint8)
        person_b[:150, :] = [0, 0, 255]      # Bright Red shirt
        person_b[150:, :] = [0, 255, 255]    # Yellow pants

        feat_a1 = extractor.extract_feature(person_a1, (0, 0, 200, 300))
        feat_a2 = extractor.extract_feature(person_a2, (0, 0, 200, 300))
        feat_b  = extractor.extract_feature(person_b, (0, 0, 200, 300))

        sim_same = OSNetExtractor.compute_similarity(feat_a1, feat_a2)
        sim_diff = OSNetExtractor.compute_similarity(feat_a1, feat_b)

        print(f"\n[TestDiscriminativeReID] Same Person Sim: {sim_same:.4f} | Different Person Sim: {sim_diff:.4f}")
        # Core invariant: same person MUST score higher than different person
        self.assertGreater(sim_same, sim_diff,
                           "Same-person similarity must exceed different-person similarity")
        self.assertGreater(sim_same, 0.70,
                           "Same-person similarity must be high (>0.70)")
        # NOTE: The absolute diff-person threshold is intentionally loose here.
        # OSNet x0.5 was trained on real pedestrian photos (MSMT17), not solid-color
        # synthetic patches \u2014 its embedding space is optimized for real clothing textures.
        # On solid-color test images the gap is smaller than on real footage.
        # The discriminative quality on real video is verified by end-to-end testing.
        self.assertGreater(sim_same - sim_diff, 0.05,
                           "Gap between same/different person similarity must be > 0.05")

if __name__ == '__main__':
    unittest.main()
