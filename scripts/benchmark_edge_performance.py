import os
import sys
import time
import json
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.detector import PersonDetector, PersonDetection
from models.reid import OSNetExtractor
from tracker.byte_track import ByteTracker
import config

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_FPS_MIN = 8.0  # Realistic minimum target FPS on Edge CPU

def benchmark_tracking_pipeline(num_frames: int = 50):
    print("=========================================================")
    print("   EDGE BENCHMARK: CCTV PERSON TRACKING & REID PIPELINE")
    print("=========================================================")
    
    # 1. Initialize components
    t0 = time.time()
    detector = PersonDetector(conf_thresh=config.DETECTION_CONF_THRESH, use_onnx=True)
    t_detector_init = (time.time() - t0) * 1000.0

    t0 = time.time()
    reid_extractor = OSNetExtractor()
    t_reid_init = (time.time() - t0) * 1000.0

    tracker = ByteTracker(camera_id="CAM_1")

    print(f"[Init] Detector Init Time  : {t_detector_init:.2f} ms")
    print(f"[Init] ReID Extractor Init: {t_reid_init:.2f} ms")
    print("---------------------------------------------------------")

    # Load a real sample frame if available, else generate a textured synthetic multi-person frame
    test_video = os.path.join(BASE_DIR, "cctv_test.mp4")
    sample_frame = None
    if os.path.exists(test_video):
        cap = cv2.VideoCapture(test_video)
        ret, frame = cap.read()
        if ret and frame is not None:
            sample_frame = frame
        cap.release()

    if sample_frame is None:
        sample_frame = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)

    # 2. Benchmark Detection
    det_times = []
    dets = []
    for _ in range(num_frames):
        t_start = time.time()
        dets = detector.detect(sample_frame)
        det_times.append((time.time() - t_start) * 1000.0)

    avg_det_ms = float(np.mean(det_times))
    std_det_ms = float(np.std(det_times))

    # 3. Benchmark ByteTrack with realistic detections (simulate 3 people if none detected)
    if len(dets) == 0:
        sim_dets = [
            PersonDetection((100, 100, 250, 450), 0.85),
            PersonDetection((350, 120, 500, 480), 0.75),
            PersonDetection((600, 80, 720, 420), 0.65),
        ]
    else:
        sim_dets = dets

    track_times = []
    for _ in range(num_frames):
        t_start = time.time()
        active = tracker.update(sim_dets, frame=sample_frame)
        track_times.append((time.time() - t_start) * 1000.0)

    avg_track_ms = float(np.mean(track_times))

    # 4. Benchmark OSNet ReID Feature Extraction per Crop
    reid_times = []
    dummy_bbox = (100.0, 100.0, 250.0, 450.0)
    for _ in range(num_frames):
        t_start = time.time()
        feat = reid_extractor.extract_feature(sample_frame, dummy_bbox)
        reid_times.append((time.time() - t_start) * 1000.0)

    avg_reid_ms = float(np.mean(reid_times))

    # 5. Workload Simulations (1 person, 3 persons, 5 persons) with Lazy ReID (every 10th frame)
    workload_results = {}
    for n_people in [1, 3, 5]:
        total_frame_ms = avg_det_ms + avg_track_ms + (n_people * avg_reid_ms * 0.10)
        est_fps = 1000.0 / max(1e-5, total_frame_ms)
        workload_results[f"{n_people}_people"] = {
            "total_frame_ms": round(total_frame_ms, 2),
            "estimated_fps": round(est_fps, 1)
        }

    # Evaluate dynamic PASS/FAIL against target FPS
    fps_3_people = workload_results["3_people"]["estimated_fps"]
    status = "PASS" if fps_3_people >= TARGET_FPS_MIN else "NEEDS_OPTIMIZATION"

    results = {
        "benchmark_environment": sys.platform,
        "detector_model": getattr(config, 'ACTIVE_MODEL', 'yolo11'),
        "detection_avg_ms": round(avg_det_ms, 2),
        "detection_std_ms": round(std_det_ms, 2),
        "bytetrack_avg_ms": round(avg_track_ms, 2),
        "osnet_reid_crop_ms": round(avg_reid_ms, 2),
        "workload_performance": workload_results,
        "target_fps_threshold": TARGET_FPS_MIN,
        "evaluation_status": status
    }

    print("\n--- BENCHMARK RESULTS ---")
    print(f"Detection Avg Time    : {avg_det_ms:.2f} ms +- {std_det_ms:.2f} ms")
    print(f"ByteTrack Kalman Update: {avg_track_ms:.2f} ms (with {len(sim_dets)} candidates)")
    print(f"OSNet ReID Crop Extract: {avg_reid_ms:.2f} ms per crop")
    for k, v in workload_results.items():
        print(f"Workload ({k}) : {v['total_frame_ms']} ms/frame ({v['estimated_fps']} FPS)")
    print(f"Overall Evaluation Status: {status} (Threshold: {TARGET_FPS_MIN} FPS)")

    out_file = os.path.join(RESULTS_DIR, "edge_tracking_benchmark.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to: {out_file}\n")
    return results

if __name__ == "__main__":
    benchmark_tracking_pipeline()
