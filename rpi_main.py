#!/usr/bin/env python3
"""
rpi_main.py — Raspberry Pi 5 Entry Point
=========================================
Headless, multi-threaded 2-camera person tracking agent.

Usage (on Raspberry Pi 5):
    # Simulation with 2 USB webcams:
    python3 rpi_main.py

    # RTSP CCTV (set env vars in .env or export before running):
    CAM_1_URL=rtsp://admin:pass@192.168.1.100:554/stream1 \
    CAM_2_URL=rtsp://admin:pass@192.168.1.101:554/stream1 \
    python3 rpi_main.py

    # Single webcam for development/testing:
    python3 rpi_main.py --single --cam 0

Environment Variables (set in .env or export):
    SUPABASE_URL          — Your Supabase project URL
    SUPABASE_SERVICE_KEY  — Supabase service role key (not anon key)
    CAM_1_URL             — Camera 1 source (int index OR rtsp:// URL). Default: 0
    CAM_2_URL             — Camera 2 source (int index OR rtsp:// URL). Default: 1
"""

import os
import time
import signal
import logging
import threading
import argparse
import cv2
import numpy as np
from typing import Dict

# Load .env file if present (for development on laptop / first Pi setup)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional — env vars can be exported directly on Pi

from models.detector import PersonDetector
from tracker.byte_track import ByteTracker
from multicam.multicam_manager import MultiCamManager
from supabase_client import push_occupancy_background
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s"
)
logger = logging.getLogger("rpi_main")

# ─── Graceful Shutdown ───────────────────────────────────────────────────────
_running = True

def _signal_handler(sig, frame):
    global _running
    logger.info("Shutdown signal received. Stopping camera threads...")
    _running = False

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ─── Camera Source Helper ────────────────────────────────────────────────────
def _open_capture(source: str) -> cv2.VideoCapture:
    """
    Opens a VideoCapture from:
      - Integer index ("0", "1") → USB webcam
      - RTSP URL ("rtsp://...")   → IP CCTV camera
      - File path ("video.mp4")  → Recorded footage for testing
    """
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        # GStreamer pipeline hint for smoother RTSP on Pi 5
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    # Prefer 720p to keep CPU headroom on Pi 5 (YOLO11n ONNX still ~20ms/frame)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag for live streams
    return cap


# ─── Per-Camera Worker Thread ────────────────────────────────────────────────
def _camera_worker(
    camera_id: str,
    source: str,
    detector: PersonDetector,
    tracker: ByteTracker,
    multicam_manager: MultiCamManager,
    floor: int,
):
    """
    Runs in its own thread — one thread per CCTV camera.
    Each thread reads frames, runs YOLO11n detection, ByteTrack tracking,
    and lazy OSNet ReID for cross-camera identity, then pushes occupancy
    count to Supabase every SYNC_INTERVAL_SEC.
    """
    logger.info(f"[{camera_id}] Opening stream: {source!r}")
    cap = _open_capture(source)

    if not cap.isOpened():
        logger.error(f"[{camera_id}] Failed to open stream: {source!r}. Thread exiting.")
        return

    logger.info(f"[{camera_id}] Stream opened. Starting detection loop.")

    frame_count = 0
    last_sync = 0.0
    last_active_tracks = []
    # Streak debounce — same MIN_HITS / MISS_GRACE logic as pipeline.py
    _MIN_HITS   = 3
    _MISS_GRACE = 5
    _streaks: Dict[int, int] = {}
    _misses:  Dict[int, int] = {}
    _confirmed: set = set()

    while _running:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"[{camera_id}] Frame read failed — reconnecting in 2s...")
            cap.release()
            time.sleep(2.0)
            cap = _open_capture(source)
            continue

        frame_count += 1
        t0 = time.time()

        try:
            # ── Step 1: Person Detection (YOLO11n / YOLO26n ONNX) ──
            detections = detector.detect(frame)

            # ── Step 2: ByteTrack Single-Camera Tracking ──
            local_tracks = tracker.update(detections, frame=frame)

            # ── Step 3: Multi-Camera Global ID + Lazy ReID ──
            global_tracks = multicam_manager.process_camera_tracks(
                camera_id=camera_id,
                tracks=local_tracks,
                frame=frame
            )

            # ── Step 4: Streak debounce with grace period (ADD=3 frames, REMOVE=5 misses) ──
            seen = {t.track_id for t in global_tracks}
            for tid in seen:
                _streaks[tid] = _streaks.get(tid, 0) + 1
                _misses[tid]  = 0
                if _streaks[tid] >= _MIN_HITS:
                    _confirmed.add(tid)
            for tid in list(_streaks):
                if tid not in seen:
                    _misses[tid] = _misses.get(tid, 0) + 1
                    if _misses[tid] >= _MISS_GRACE:
                        _confirmed.discard(tid)
                        _streaks.pop(tid, None)
                        _misses.pop(tid, None)

            last_active_tracks = [t for t in global_tracks if t.track_id in _confirmed]
        except Exception as e:
            logger.error(f"[{camera_id}] Error in frame {frame_count} processing: {e}", exc_info=True)
            continue

        elapsed_ms = (time.time() - t0) * 1000

        # ── Step 5: Non-blocking Supabase Occupancy Push (debounced count) ──
        now = time.time()
        if (now - last_sync) >= config.SYNC_INTERVAL_SEC:
            last_sync = now
            count = len(last_active_tracks)  # confirmed (debounced) count
            push_occupancy_background(camera_id, floor, count)
            logger.info(
                f"[{camera_id}] Floor {floor} | {count} confirmed people | "
                f"Detection: {elapsed_ms:.1f}ms"
            )

    cap.release()
    logger.info(f"[{camera_id}] Thread stopped cleanly.")


# ─── Main Entry Point ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Cafe Multi-Cam Person Tracking Agent (Raspberry Pi 5)")
    parser.add_argument("--single", action="store_true", help="Run single camera only (CAM_1)")
    parser.add_argument("--cam", type=str, default=None, help="Override CAM_1 source (index or RTSP URL)")
    parser.add_argument("--show", action="store_true", help="Show live visualization window (only for desktop/dev mode)")
    args = parser.parse_args()

    print("═══════════════════════════════════════════════════════════")
    print("   CAFE PERSON TRACKING AGENT — Raspberry Pi 5 Edition")
    print("   YOLO11n + ByteTrack + OSNet ReID + Supabase Real-time")
    print("═══════════════════════════════════════════════════════════")

    # ── Load Models (shared across cameras — thread-safe for inference) ──
    logger.info("Loading YOLO11n ONNX detector...")
    detector = PersonDetector(conf_thresh=config.DETECTION_CONF_THRESH, use_onnx=True)

    logger.info("Loading MultiCamManager (OSNet ReID)...")
    multicam_manager = MultiCamManager()

    # ── Determine Camera Sources ──
    cam_sources = {}
    if args.single or args.cam is not None:
        src = args.cam or config.DEFAULT_CAMERAS_CONFIG["CAM_1"]["rtsp_url"]
        cam_sources = {"CAM_1": str(src)}
    else:
        for cam_id, cam_cfg in config.DEFAULT_CAMERAS_CONFIG.items():
            cam_sources[cam_id] = str(cam_cfg["rtsp_url"])

    logger.info(f"Active cameras: {cam_sources}")

    # ── Start One Thread Per Camera ──
    threads = []
    for cam_id, source in cam_sources.items():
        cam_cfg = config.DEFAULT_CAMERAS_CONFIG.get(cam_id, {})
        floor = cam_cfg.get("floor", 1)
        tracker = ByteTracker(camera_id=cam_id)

        t = threading.Thread(
            target=_camera_worker,
            args=(cam_id, source, detector, tracker, multicam_manager, floor),
            name=f"CamThread-{cam_id}",
            daemon=True,
        )
        t.start()
        threads.append(t)
        logger.info(f"Started thread for {cam_id} (Floor {floor}) ← source: {source!r}")

    logger.info("All camera threads running. Press Ctrl+C to stop.")

    # ── Keep main thread alive until SIGINT/SIGTERM ──
    try:
        while _running:
            time.sleep(0.5)
    finally:
        logger.info("Waiting for camera threads to finish...")
        for t in threads:
            t.join(timeout=5.0)
        logger.info("Person Tracking Agent stopped. Goodbye!")


if __name__ == "__main__":
    main()
