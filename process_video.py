"""
process_video.py -- CCTV Person Tracking (YOLO26n / YOLO11n + ByteTrack + OSNet ReID)
=======================================================================================
Usage:
  python process_video.py --video cctv_test.mp4                        # Optimal default: YOLO26n, imgsz=640, skip=2
  python process_video.py --video cctv_test.mp4 --skip 3              # Ultra fast (RPi5 recommendation)
  python process_video.py --video cctv_test.mp4 --model yolo11         # Revert to YOLO11n
  python process_video.py --video cctv_test.mp4 --far-pass             # Enable 2nd-pass for distant small people
  python process_video.py --video cctv_test.mp4 --no-show              # Headless mode (SSH/server)
"""
import os, time, argparse, cv2, numpy as np
from models.detector import PersonDetector
from tracker.byte_track import ByteTracker
import config

_CONF         = 0.18
_LOW_CONF     = 0.05
_TRACK_THRESH = 0.25
_TRACK_BUFFER = 150   # was 250 — reduced to prevent ghost tracks from persisting too long
_MATCH_THRESH = 0.65  # was 0.60 — tighter gate for fused cost
_REID_THRESH  = 0.62  # was 0.50 — stricter reconnect threshold for OSNet x0.25
_REID_WEIGHT      = 0.55  # was 0.35 — raise appearance weight to fix ID-swap-on-crossing
_REID_EASY_THRESH = 0.30  # IoU cost threshold for Stage 2A unambiguous match (no ReID)
_MIN_HITS         = 3
_MISS_GRACE       = 5     # consecutive misses before removing confirmed track


def _apply_config(conf):
    config.DETECTION_CONF_THRESH  = conf
    config.LOW_CONF_THRESH        = _LOW_CONF
    config.TRACK_THRESH           = _TRACK_THRESH
    config.TRACK_BUFFER           = _TRACK_BUFFER
    config.MATCH_THRESH           = _MATCH_THRESH
    config.REID_SIMILARITY_THRESH = _REID_THRESH
    config.REID_COST_WEIGHT       = _REID_WEIGHT
    config.REID_EASY_THRESH       = _REID_EASY_THRESH


def draw(frame, tracks, confirmed, fi, tf, fps, n_unique, model_label):
    out = frame.copy()
    H, W = out.shape[:2]
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.tlbr]
        cx, cy = [int(v) for v in t.centroid]
        ok  = t.track_id in confirmed
        ch  = hash(t.track_id) & 0xFFFFFF
        col = (ch & 0xFF, (ch >> 8) & 0xFF, (ch >> 16) & 0xFF)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2 if ok else 1)
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
        cv2.arrowedLine(out, (cx, cy),
            (int(cx + t.velocity_x * 6), int(cy + t.velocity_y * 6)),
            (255, 255, 0), 2, tipLength=0.3)
        lbl = "ID %d [%s] %.2f" % (t.track_id, "OK" if ok else "?", t.score)
        tsz = cv2.getTextSize(lbl, 0, 0.52, 1)[0]
        cv2.rectangle(out, (x1, y1 - 22), (x1 + tsz[0] + 8, y1), col, -1)
        cv2.putText(out, lbl, (x1 + 4, y1 - 6), 0, 0.52, (255, 255, 255), 1)
    aktif = sum(1 for t in tracks if t.track_id in confirmed)
    cv2.rectangle(out, (0, 0), (W, 48), (15, 15, 15), -1)
    cv2.putText(out,
        "Frame %d/%d | Aktif: %d | Unik: %d | %s" % (fi, tf, aktif, n_unique, model_label),
        (12, 17), 0, 0.50, (0, 255, 0), 1)
    cv2.putText(out,
        "FPS %.1f | Tracks: %d | [OK]=counted [?]=new/unconfirmed" % (fps, len(tracks)),
        (12, 38), 0, 0.44, (180, 180, 180), 1)
    return out


def process(video_in,
            video_out="results/output_processed.mp4",
            show=True,
            conf=_CONF,
            frame_skip=2,
            imgsz=640,
            model=None,
            far_pass=False):
    if not os.path.exists(video_in):
        print("[Error] Video tidak ditemukan:", video_in)
        return

    _apply_config(conf)
    config.YOLO_INPUT_SIZE = imgsz
    config.ENABLE_FAR_REGION_PASS = far_pass
    os.makedirs(os.path.dirname(video_out) or ".", exist_ok=True)

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        print("[Error] Gagal buka video")
        return

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    TF     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    active_model = (model or getattr(config, "ACTIVE_MODEL", "yolo26")).lower()
    model_label  = "YOLO26n (NMS-free)" if active_model == "yolo26" else "YOLO11n"

    print("=" * 65)
    print("CCTV TRACKING  model=%s  conf=%.2f  imgsz=%d  skip=%d  far_pass=%s" % (
        model_label, conf, imgsz, frame_skip, far_pass))
    print("Fused IoU+ReID w=%.2f  match=%.2f  reid=%.2f" % (
        _REID_WEIGHT, _MATCH_THRESH, _REID_THRESH))
    print("%dx%d @%.1ffps | %d frames" % (W, H, fps_in, TF))
    print("=" * 65)

    writer   = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))
    detector = PersonDetector(conf_thresh=conf, use_onnx=True,
                              input_size=imgsz, model_name=model)
    tracker  = ByteTracker(camera_id="CAM_EVAL", fps=fps_in)

    fi = 0
    t0 = time.time()
    streaks   = {}
    misses    = {}
    confirmed = set()
    unique    = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1
        tt = time.time()

        if fi % frame_skip == 0:
            dets   = detector.detect(frame)
            tracks = tracker.update(dets, frame=frame)
        else:
            # Skipped frame: Kalman predict only (~0.2ms, no YOLO/ReID)
            for t in tracker.tracked_stracks:
                t.predict()
            tracks = [t for t in tracker.tracked_stracks if t.is_activated]

        seen = {t.track_id for t in tracks}
        for tid in seen:
            streaks[tid] = streaks.get(tid, 0) + 1
            misses[tid]  = 0
            if streaks[tid] >= _MIN_HITS:
                confirmed.add(tid)
                unique.add(tid)
        for tid in list(streaks):
            if tid not in seen:
                misses[tid] = misses.get(tid, 0) + 1
                if misses[tid] >= _MISS_GRACE:
                    confirmed.discard(tid)
                    streaks.pop(tid, None)
                    misses.pop(tid, None)

        spd = 1.0 / max(1e-5, time.time() - tt)
        ann = draw(frame, tracks, confirmed, fi, TF, spd, len(unique), model_label)
        writer.write(ann)

        if show:
            cv2.imshow("CCTV Tracking", ann)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                print("Stopped by user.")
                break

        if fi % 50 == 0:
            ak = sum(1 for t in tracks if t.track_id in confirmed)
            print("  [%5.1f%%] f%d/%d | Aktif:%d | Unik:%d | %.1ffps" % (
                fi / TF * 100, fi, TF, ak, len(unique), spd), flush=True)

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    e = time.time() - t0
    print("=" * 65)
    print("Done %.1fs | Unik(OK>=%d): %d | %s" % (e, _MIN_HITS, len(unique), video_out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CCTV Person Tracking Evaluation")
    p.add_argument("--video",    default="cctv_test.mp4",
                   help="Path to input video file")
    p.add_argument("--output",   default="results/output_processed.mp4",
                   help="Path to output annotated video")
    p.add_argument("--conf",     type=float, default=_CONF,
                   help="Detection confidence threshold (default: %.2f)" % _CONF)
    p.add_argument("--imgsz",    type=int,   default=640,
                   help="YOLO input size: 640=default (fast/edge), 960=balanced, 1280=high-res")
    p.add_argument("--skip",     type=int,   default=2,
                   help="Frame skip: 2=default (2x faster), 1=every frame, 3=RPi5 recommended")
    p.add_argument("--far-pass", action="store_true",
                   help="Enable 2nd-pass detection on far/top region for distant people (higher recall, ~2x slower)")
    p.add_argument("--model",    default=None,
                   help="Model to use: yolo26 (default, NMS-free) or yolo11")
    p.add_argument("--no-show",  action="store_true",
                   help="Disable display window (use on RPi5 over SSH)")
    a = p.parse_args()
    process(a.video, a.output, not a.no_show, a.conf, a.skip, a.imgsz, a.model, a.far_pass)
