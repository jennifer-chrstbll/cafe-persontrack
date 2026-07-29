import time
import cv2
import numpy as np
import argparse
from typing import List

from models.detector import PersonDetector
from tracker.byte_track import ByteTracker, STrack
from api_client import CRMBackendClient, sync_telemetry_background
import config

def run_single_cam_demo(camera_source: str = "0", camera_id: str = "CAM_1", target_fps: int = 30):
    print("=========================================================")
    print("   CAFE PERSON TRACKING (SINGLE CAMERA WEBCAM DEMO)")
    print("=========================================================")
    print(f"Camera Source: {camera_source}")
    print(f"Target FPS Mode: {target_fps} FPS")
    print("Press 'q' or ESC in visualizer window to stop.")
    print("---------------------------------------------------------")

    detector = PersonDetector(conf_thresh=config.DETECTION_CONF_THRESH, use_onnx=True)
    tracker = ByteTracker(camera_id=camera_id)
    backend_client = CRMBackendClient()

    if camera_source.isdigit():
        cap = cv2.VideoCapture(int(camera_source))
    elif os.path.exists(camera_source):
        cap = cv2.VideoCapture(camera_source)
    else:
        print(f"[demo_single_cam] Error: Cannot open camera source {camera_source}")
        return

    if not cap.isOpened():
        print(f"[demo_single_cam] Error: Camera source {camera_source} could not be opened.")
        return

    # Set 1280x720 webcam resolution if available
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_sync = 0.0
    frame_count = 0
    frame_stride = max(1, int(30 / max(1, target_fps)))
    active_tracks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[demo_single_cam] End of video stream or camera disconnected.")
            break

        frame_count += 1
        t_start = time.time()

        # Execute Person Detection & ByteTrack tracking ONLY on sampled frame stride
        if frame_count % frame_stride == 0:
            # Step 1: Person Detection
            detections = detector.detect(frame)

            # Step 2: ByteTrack Tracking with OSNet ReID Appearance Persistence
            active_tracks = tracker.update(detections, frame=frame)

        elapsed = time.time() - t_start
        fps = 1.0 / max(1e-5, elapsed)

        # Step 3: Non-blocking Background Sync to Backend
        now = time.time()
        if (now - last_sync) >= config.SYNC_INTERVAL_SEC:
            last_sync = now
            tracks_payload = [
                {
                    "camera_id": camera_id,
                    "raw_track_id": f"GT-{t.track_id:04d}",
                    "pos_x": round(t.centroid[0], 2),
                    "pos_y": round(t.centroid[1], 2),
                    "velocity_x": round(t.velocity_x, 2),
                    "velocity_y": round(t.velocity_y, 2),
                    "status": "ACTIVE"
                }
                for t in active_tracks
            ]
            sync_telemetry_background(backend_client, camera_id, 1, tracks_payload)

        # Step 4: Visual Overlay Rendering
        out_frame = frame.copy()
        for track in active_tracks:
            x1, y1, x2, y2 = [int(v) for v in track.tlbr]
            cx, cy = [int(v) for v in track.centroid]
            track_label = f"Track #{track.track_id} ({track.score:.2f})"

            # Unique track color
            color_hash = hash(track.track_id) & 0xFFFFFF
            color = (color_hash & 0xFF, (color_hash >> 8) & 0xFF, (color_hash >> 16) & 0xFF)

            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(out_frame, (cx, cy), 5, (0, 0, 255), -1)

            # Draw velocity direction arrow
            vx_end = int(cx + track.velocity_x * 8)
            vy_end = int(cy + track.velocity_y * 8)
            cv2.arrowedLine(out_frame, (cx, cy), (vx_end, vy_end), (255, 255, 0), 2, tipLength=0.3)

            # Text tag
            t_size = cv2.getTextSize(track_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(out_frame, (x1, y1 - 25), (x1 + t_size[0] + 10, y1), color, -1)
            cv2.putText(out_frame, track_label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # HUD Stats Header
        h, w = out_frame.shape[:2]
        cv2.rectangle(out_frame, (0, 0), (w, 40), (25, 25, 25), -1)
        hud_text = f"SINGLE CAM DEMO | Mode: {target_fps} FPS (Stride 1/{frame_stride}) | People: {len(active_tracks)}"
        cv2.putText(out_frame, hud_text, (20, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow(f"Kafe Person Tracking - Single Cam Webcam Demo ({target_fps} FPS Mode)", out_frame)

        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam single-camera demo stopped.")


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description="Single-Camera Person Tracking Webcam Demo")
    parser.add_argument("--cam", type=str, default="0", help="Webcam index (e.g. 0) or video file path")
    parser.add_argument("--fps", type=int, default=30, help="Target processing FPS (e.g. 5, 15, or 30)")
    args = parser.parse_args()
    run_single_cam_demo(args.cam, target_fps=args.fps)
