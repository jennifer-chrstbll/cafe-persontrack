import os
import time
import cv2
import numpy as np
import argparse
from typing import List

from pipeline import MultiCamPipeline
from scripts.create_cctv_demo import create_demo_videos
import config

def get_video_capture(source: str) -> cv2.VideoCapture:
    """Returns cv2.VideoCapture object for webcam index, video file path, or stream."""
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    elif os.path.exists(source):
        return cv2.VideoCapture(source)
    else:
        print(f"[demo_sim] Warning: Source '{source}' not found as video file or webcam index.")
        return None


def generate_synthetic_frame(frame_idx: int, camera_id: str, width: int = 1280, height: int = 720) -> np.ndarray:
    """Generates synthetic video frame simulating cafe ceiling CCTV view with moving people."""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 40
    
    cv2.rectangle(frame, (100, 100), (300, 300), (60, 60, 60), -1)
    cv2.putText(frame, "Table 1", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
    
    cv2.rectangle(frame, (500, 100), (700, 300), (60, 60, 60), -1)
    cv2.putText(frame, "Table 2", (550, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

    p1_x = int((frame_idx * 5) % (width - 100)) + 50
    p1_y = 400
    cv2.circle(frame, (p1_x, p1_y), 25, (100, 200, 255), -1)
    cv2.rectangle(frame, (p1_x - 30, p1_y - 60), (p1_x + 30, p1_y + 60), (0, 165, 255), 2)

    if camera_id == "CAM_1":
        p2_x = int(width * 0.82)
        p2_y = int(height * 0.75) + int(np.sin(frame_idx * 0.1) * 20)
        cv2.circle(frame, (p2_x, p2_y), 25, (255, 100, 200), -1)
        cv2.rectangle(frame, (p2_x - 30, p2_y - 60), (p2_x + 30, p2_y + 60), (255, 0, 255), 2)

    return frame


def run_simulation(source_cam1: str, source_cam2: str):
    print("=========================================================")
    print("   CAFE PERSON TRACKING & MULTI-CAM REID SIMULATOR")
    print("=========================================================")
    print(f"Cam 1 Source: {source_cam1}")
    print(f"Cam 2 Source: {source_cam2}")
    print("Press 'q' or ESC in visualizer window to stop.")
    print("---------------------------------------------------------")

    pipeline = MultiCamPipeline(camera_ids=["CAM_1", "CAM_2"], use_onnx=True)

    cap1 = get_video_capture(source_cam1)
    cap2 = get_video_capture(source_cam2)

    frame_idx = 0

    while True:
        frame_idx += 1
        
        # Read frame for CAM_1
        frame1 = None
        if cap1 is not None and cap1.isOpened():
            ret1, frame1 = cap1.read()
            if not ret1:
                # Loop video stream for continuous demonstration
                cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret1, frame1 = cap1.read()
        
        if frame1 is None:
            frame1 = generate_synthetic_frame(frame_idx, "CAM_1")

        # Read frame for CAM_2
        frame2 = None
        if cap2 is not None and cap2.isOpened():
            ret2, frame2 = cap2.read()
            if not ret2:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret2, frame2 = cap2.read()

        if frame2 is None:
            frame2 = generate_synthetic_frame(frame_idx + 30, "CAM_2")

        t_start = time.time()
        
        # Process both camera feeds through pipeline
        tracks1 = pipeline.process_frame("CAM_1", frame1)
        tracks2 = pipeline.process_frame("CAM_2", frame2)
        
        elapsed = time.time() - t_start
        fps = 1.0 / max(1e-5, elapsed)

        # Draw tracking & transition zone overlays
        out1 = pipeline.draw_tracks(frame1, "CAM_1", tracks1, fps)
        out2 = pipeline.draw_tracks(frame2, "CAM_2", tracks2, fps)

        # Stack camera windows side by side
        h1, w1 = out1.shape[:2]
        h2, w2 = out2.shape[:2]
        if (h1, w1) != (h2, w2):
            out2 = cv2.resize(out2, (w1, h1))
            
        combined_view = np.hstack([out1, out2])
        
        cv2.imshow("Cafe Multi-Camera Person Tracking & Lazy ReID", combined_view)

        key = cv2.waitKey(40) & 0xFF
        if key == 27 or key == ord('q'):
            break

    if cap1: cap1.release()
    if cap2: cap2.release()
    cv2.destroyAllWindows()
    print("Simulation stopped.")


if __name__ == "__main__":
    out_dir = os.path.join(config.BASE_DIR, "demo_videos")
    default_v1 = os.path.join(out_dir, "cctv_fl1.mp4")
    default_v2 = os.path.join(out_dir, "cctv_fl2.mp4")

    # Auto-generate video files if they don't exist yet
    if not os.path.exists(default_v1) or not os.path.exists(default_v2):
        create_demo_videos()

    parser = argparse.ArgumentParser(description="Multi-Camera CCTV Simulator & ReID Visualizer")
    parser.add_argument("--cam1", type=str, default=default_v1, help="Video path or webcam index for Cam 1 (FL1)")
    parser.add_argument("--cam2", type=str, default=default_v2, help="Video path or webcam index for Cam 2 (FL2)")
    args = parser.parse_args()

    run_simulation(args.cam1, args.cam2)
