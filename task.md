# Cafe Tracking + Face Recognition Migration Status

## Phase 1 — BoxMOT / BoT-SORT + Swappable Detector Toggle
- [x] 1.1 Install `boxmot>=25.0.0` in `requirements.txt` and `requirements_rpi.txt`
- [x] 1.2 Refactor `config.py`: `DETECTOR_BACKEND` env-var (`yolo11n` ↔ `yolo26n`), `DETECTOR_PATHS`, `BOTSORT_*` constants
- [x] 1.3 Create `tracker/botsort_tracker.py` wrapping BoxMOT `BotSort` + `BotSortTrack` duck-typing `STrack`
- [x] 1.4 Update entrypoints (`pipeline.py`, `demo_single_cam.py`, `process_video.py`, `rpi_main.py`, `multicam_manager.py`)
- [x] 1.5 Verify all 25 unit tests pass

## Phase 2 — Detector Quantization & Precision Optimization
- [x] 2.1 ONNX exports for YOLO11n and YOLO26n in `weights/`
- [x] 2.2 Built calibration set (100 CCTV frames + 201 person crops in `calib_images/` and `calib_person_crops/`)
- [x] 2.3 Static INT8 quantization for YOLO11n (`weights/yolo11n_int8.onnx`: 10.42 MB → 3.26 MB)
- [x] 2.4 Precision validation for YOLO26n STAL attention head: FP16 fallback (`weights/yolo26n_fp16.onnx`: 9.94 MB → 5.2 MB) preserves 100% detection recall and confidence

## Phase 3 — ReID & Face Recognition (SCRFD + MobileFaceNet)
- [x] 3.1 OSNet x0_25 FP16 conversion (`weights/osnet_x0_25_fp16.onnx`: 885.9 KB → 493.2 KB)
- [x] 3.2 Created lean `d:/Projects/cafe_facerec/face_models.py` with `SCRFDFaceDetector` (`det_500m.onnx`) and `MobileFaceNetModel` (`w600k_mbf.onnx`)
- [x] 3.3 Implemented genuine 5-point affine similarity alignment (`align_face_5pts`) to ArcFace 112x112 template
- [x] 3.4 Updated `d:/Projects/cafe_facerec/face_recognition_module.py` to use SCRFD + MobileFaceNet (eliminating legacy OpenCV SSD and fake proportions)

## Phase 4 — End-to-End Pipeline Verification
- [x] 4.1 End-to-end CCTV tracking run on `cctv_test.mp4` (1452 frames, 113.6s, 42 unique IDs): `results/output_yolo26n_botsort.mp4`
- [x] 4.2 End-to-end face recognition timing on register node: 47.5 ms total latency (well within the 150–200 ms real-time counter budget)
- [x] 4.3 Zero dead or fake code: all models genuine, loaded directly via ONNX Runtime
