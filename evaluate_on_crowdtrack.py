"""
evaluate_on_crowdtrack.py
--------------------------
Menjalankan pipeline produksi ASLI kamu (models.detector.PersonDetector +
tracker.botsort_tracker.BotSortTracker) di atas satu sequence format MOT,
lalu menilai hasilnya terhadap ground truth pakai py-motmetrics.

Taruh file ini di root folder cafe-persontrack (sejajar dengan config.py).

Install dulu:
    pip install motmetrics --break-system-packages

Cara pakai:
    python evaluate_on_crowdtrack.py --seq CrowdTrack-MOT/train/track0001

Untuk bandingkan YOLO11n vs YOLO26n, set env var dulu sebelum jalankan
(script ini otomatis baca config.DETECTOR_BACKEND, sama seperti demo_single_cam.py):
    $env:DETECTOR_BACKEND="yolo26n"
    python evaluate_on_crowdtrack.py --seq CrowdTrack-MOT/train/track0001

Opsional, buat tes cepat dulu (cuma proses N frame pertama):
    python evaluate_on_crowdtrack.py --seq CrowdTrack-MOT/train/track0001 --max-frames 100
"""
import argparse
import configparser
import glob
import os
import sys

import cv2
import numpy as np

# --- Compat shim: motmetrics 1.4.0 still calls np.asfarray, removed in NumPy 2.0 ---
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

try:
    import motmetrics as mm
except ImportError:
    sys.exit(
        "Package 'motmetrics' belum ter-install. Jalankan dulu:\n"
        "    pip install motmetrics --break-system-packages"
    )

import config
from models.detector import PersonDetector
from tracker.botsort_tracker import BotSortTracker


def load_gt(gt_path: str) -> dict:
    """
    Baca gt.txt format MOT: frame,id,x,y,w,h,conf,class,visibility
    -> {frame: [(id,x,y,w,h), ...]}

    Mengikuti konvensi resmi MOTChallenge evaluation:
    - conf==0 berarti baris ini ditandai "ignore", tidak dihitung sama sekali.
    - class harus == 1 (pedestrian). Kelas lain (2=person_on_vehicle,
      7=static_person, 8=distractor, 12=reflection, dst) BUKAN target nyata --
      biasanya manekin, pantulan cermin, dsb -- dan harus dibuang, kalau tidak
      GT count akan meledak dan FN jadi menyesatkan (kasus MOT17-04 yang
      settingnya di mall, penuh manekin & cermin).

    Kalau file gt.txt cuma 6 kolom (mis. dataset CrowdTrack-MOT yang sudah
    single-class 'person' semua), filter ini otomatis dilewati -- semua baris
    tetap dipakai seperti sebelumnya.
    """
    gt = {}
    for line in open(gt_path, "r"):
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        frame = int(float(parts[0]))
        tid = int(float(parts[1]))
        x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))

        if len(parts) >= 8:
            conf = float(parts[6])
            cls = int(float(parts[7]))
            if conf == 0:
                continue
            if cls != 1:
                continue

        gt.setdefault(frame, []).append((tid, x, y, w, h))
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="Path folder sequence, mis. CrowdTrack-MOT/train/track0001")
    ap.add_argument("--max-frames", type=int, default=None, help="Batasi jumlah frame untuk tes cepat")
    args = ap.parse_args()

    img_dir = os.path.join(args.seq, "img1")
    gt_path = os.path.join(args.seq, "gt", "gt.txt")
    seqinfo_path = os.path.join(args.seq, "seqinfo.ini")

    if not os.path.isdir(img_dir):
        sys.exit(f"Folder frame tidak ditemukan: {img_dir}")
    if not os.path.isfile(gt_path):
        sys.exit(f"Ground truth tidak ditemukan: {gt_path}")

    fps = 25.0
    if os.path.isfile(seqinfo_path):
        cp = configparser.ConfigParser()
        cp.read(seqinfo_path)
        fps = float(cp.get("Sequence", "frameRate", fallback="25"))

    frame_files = sorted(
        glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    )
    if args.max_frames:
        frame_files = frame_files[: args.max_frames]
    if not frame_files:
        sys.exit(f"Tidak ada file .jpg/.png di {img_dir}")

    gt = load_gt(gt_path)

    print("=" * 60)
    print(f"Detector backend : {config.ACTIVE_MODEL}  ({config.DETECTOR_BACKEND})")
    print(f"Detector path    : {config.YOLO_MODEL_PATH}")
    print(f"OSNet path       : {config.OSNET_MODEL_PATH}")
    print(f"Sequence         : {args.seq}  ({len(frame_files)} frame)")
    print("=" * 60)

    detector = PersonDetector(conf_thresh=config.DETECTION_CONF_THRESH, use_onnx=True)
    tracker = BotSortTracker(camera_id="EVAL", fps=fps)

    acc = mm.MOTAccumulator(auto_id=True)
    total_dets = 0
    total_gt = 0

    for i, fpath in enumerate(frame_files, start=1):
        frame = cv2.imread(fpath)
        if frame is None:
            continue

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        total_dets += len(detections)
        total_gt += len(gt.get(i, []))

        hyp_ids, hyp_boxes = [], []
        for t in tracks:
            x1, y1, x2, y2 = t.tlbr
            hyp_ids.append(t.track_id)
            hyp_boxes.append([x1, y1, x2 - x1, y2 - y1])

        gt_rows = gt.get(i, [])
        gt_ids = [r[0] for r in gt_rows]
        gt_boxes = [[r[1], r[2], r[3], r[4]] for r in gt_rows]

        if gt_boxes and hyp_boxes:
            distances = mm.distances.iou_matrix(gt_boxes, hyp_boxes, max_iou=0.5)
        else:
            distances = np.empty((len(gt_boxes), len(hyp_boxes)))

        acc.update(gt_ids, hyp_ids, distances)

        if i % 25 == 0 or i == len(frame_files):
            print(f"  memproses frame {i}/{len(frame_files)}...", end="\r")

    n_frames_processed = len(frame_files)
    print(f"\n\nRata-rata deteksi/frame : {total_dets / n_frames_processed:.1f}")
    print(f"Rata-rata GT/frame      : {total_gt / n_frames_processed:.1f}")
    if total_dets < total_gt * 0.7:
        print(">> Deteksi jauh di bawah ground truth -> kemungkinan besar masalah RECALL DETECTOR, bukan tracker.")

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "mota", "motp", "idf1",
            "num_switches", "num_false_positives", "num_misses", "num_fragmentations",
        ],
        name="hasil",
    )
    print("=== HASIL EVALUASI ===")
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    ))
    print(
        "\nCatatan baca cepat:\n"
        "  IDF1  -> makin tinggi makin bagus (identitas konsisten, sedikit ID switch)\n"
        "  IDSW  -> jumlah ID switch mentah, makin KECIL makin bagus (ini yang paling relevan buat kamu)\n"
        "  MOTA  -> akurasi keseluruhan (deteksi + asosiasi), makin tinggi makin bagus\n"
    )


if __name__ == "__main__":
    main()