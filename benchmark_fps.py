"""
benchmark_fps.py
-----------------
Mengukur kecepatan (FPS) pipeline produksimu di komputer ini sekarang, dipisah
per komponen (detector saja, tracker saja, gabungan) -- supaya kelihatan mana
yang paling mahal, dan sebagai baseline pembanding sebelum dites di Raspberry
Pi 5 yang jauh lebih lemah.

PENTING: angka FPS di sini adalah FPS di PC/laptop-mu, BUKAN prediksi FPS di
Pi 5. Pi 5 jauh lebih lambat (biasanya 3-8x lebih lambat dari PC modern untuk
beban kerja seperti ini). Gunakan angka ini untuk PERBANDINGAN RELATIF antar
setting (mis. "1280 berapa persen lebih lambat dari 960"), bukan patokan
absolut untuk deployment.

Cara pakai:
    python benchmark_fps.py --seq MOT17/train/MOT17-04-FRCNN --max-frames 100

Untuk bandingkan beberapa setting sekaligus, jalankan berkali-kali dengan env
var berbeda:
    $env:YOLO_INPUT_SIZE="960";  $env:ENABLE_FAR_REGION_PASS="false"; python benchmark_fps.py --seq MOT17/train/MOT17-04-FRCNN --max-frames 100
    $env:YOLO_INPUT_SIZE="1280"; $env:ENABLE_FAR_REGION_PASS="false"; python benchmark_fps.py --seq MOT17/train/MOT17-04-FRCNN --max-frames 100
    $env:YOLO_INPUT_SIZE="1280"; $env:ENABLE_FAR_REGION_PASS="true";  python benchmark_fps.py --seq MOT17/train/MOT17-04-FRCNN --max-frames 100
"""
import argparse
import glob
import os
import sys
import time

import cv2

import config
from models.detector import PersonDetector
from tracker.botsort_tracker import BotSortTracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="Path folder sequence, mis. MOT17/train/MOT17-04-FRCNN")
    ap.add_argument("--max-frames", type=int, default=100, help="Jumlah frame untuk diukur (default 100)")
    ap.add_argument("--warmup", type=int, default=5, help="Frame pemanasan sebelum mulai timing (default 5)")
    args = ap.parse_args()

    img_dir = os.path.join(args.seq, "img1")
    frame_files = sorted(
        glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    )
    if not frame_files:
        sys.exit(f"Tidak ada frame di {img_dir}")

    total_needed = args.warmup + args.max_frames
    frame_files = frame_files[:total_needed]
    if len(frame_files) < total_needed:
        sys.exit(f"Sequence cuma punya {len(frame_files)} frame, butuh {total_needed} (warmup+max_frames)")

    print("=" * 60)
    print(f"Detector backend      : {config.ACTIVE_MODEL} ({config.DETECTOR_BACKEND})")
    print(f"YOLO_INPUT_SIZE        : {getattr(config, 'YOLO_INPUT_SIZE', 960)}")
    print(f"ENABLE_FAR_REGION_PASS : {getattr(config, 'ENABLE_FAR_REGION_PASS', False)}")
    print(f"OSNet precision        : {config.OSNET_PRECISION}")
    print(f"Frame diukur           : {args.max_frames}  (+{args.warmup} warmup)")
    print("=" * 60)

    detector = PersonDetector(conf_thresh=config.DETECTION_CONF_THRESH, use_onnx=True)
    tracker = BotSortTracker(camera_id="BENCH", fps=25.0)

    frames = [cv2.imread(f) for f in frame_files]

    # Warmup -- ONNX Runtime & first-frame allocations tend to be slow;
    # exclude from timing so the number reflects steady-state speed.
    for f in frames[: args.warmup]:
        dets = detector.detect(f)
        tracker.update(dets, f)

    measured = frames[args.warmup :]

    det_times, trk_times = [], []
    for f in measured:
        t0 = time.perf_counter()
        dets = detector.detect(f)
        t1 = time.perf_counter()
        tracker.update(dets, f)
        t2 = time.perf_counter()

        det_times.append(t1 - t0)
        trk_times.append(t2 - t1)

    n = len(measured)
    det_avg_ms = sum(det_times) / n * 1000
    trk_avg_ms = sum(trk_times) / n * 1000
    total_avg_ms = det_avg_ms + trk_avg_ms

    print(f"\nDetector  : {det_avg_ms:6.1f} ms/frame  -> {1000/det_avg_ms:5.1f} FPS (kalau berdiri sendiri)")
    print(f"Tracker   : {trk_avg_ms:6.1f} ms/frame  -> {1000/trk_avg_ms:5.1f} FPS (kalau berdiri sendiri)")
    print(f"Gabungan  : {total_avg_ms:6.1f} ms/frame  -> {1000/total_avg_ms:5.1f} FPS (pipeline penuh, realistis)")
    print(
        f"\n>> Di PC ini, pipeline penuh berjalan ~{1000/total_avg_ms:.1f} FPS.\n"
        f"   Ingat: Pi 5 kemungkinan besar 3-8x LEBIH LAMBAT dari ini untuk beban kerja serupa.\n"
        f"   Estimasi kasar di Pi 5: ~{1000/total_avg_ms/5:.1f} FPS (asumsi 5x lebih lambat -- perlu"
        f" divalidasi langsung di Pi 5 begitu ada aksesnya)."
    )


if __name__ == "__main__":
    main()