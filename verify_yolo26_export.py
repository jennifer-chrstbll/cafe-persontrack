"""
verify_yolo26_export.py
------------------------
Memastikan weights/yolo26n.onnx BENAR-BENAR model YOLO26 NMS-free yang sah,
bukan file salah/ke-mislabel, sebelum kita percaya hasil eval yang memakainya.

Jalankan:
    python verify_yolo26_export.py
"""
import hashlib
import os

import onnx
import onnxruntime as ort


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(path):
    print(f"\n=== {path} ===")
    if not os.path.exists(path):
        print("  TIDAK DITEMUKAN.")
        return
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Ukuran file : {size_mb:.2f} MB")
    print(f"  SHA256      : {sha256sum(path)}")

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print(f"  Input name  : {inp.name}  shape={inp.shape}")
    for o in sess.get_outputs():
        print(f"  Output name : {o.name}  shape={o.shape}")

    model = onnx.load(path)
    print(f"  Jumlah node graph : {len(model.graph.node)}")
    # Cari node yang menandakan ada NMS eksplisit di dalam graph (ciri YOLO11
    # klasik biasanya TIDAK punya NMS di dalam graph -- itu dilakukan manual
    # di detector.py. Tapi kalau model YOLO26 asli, cek shape output-nya harus
    # sekitar (1, 300, 6) -- kalau (1, 84, 8400) itu format YOLO11, BUKAN YOLO26!)


print("Verifikasi model deteksi...")
inspect("weights/yolo11n.onnx")
inspect("weights/yolo26n.onnx")

print(
    "\n--- CARA BACA ---\n"
    "1. Kalau SHA256 kedua file SAMA PERSIS -> ini file yang sama/ke-copy, BUKAN model beda. Masalah!\n"
    "2. Output shape yolo11n.onnx yang benar : sekitar (1, 84, 8400)\n"
    "3. Output shape yolo26n.onnx yang benar : sekitar (1, 300, 6)  <- NMS-free\n"
    "   Kalau yolo26n.onnx shape-nya malah (1, 84, 8400) juga -> itu SEBENARNYA model YOLO11\n"
    "   yang di-mislabel, dan detector.py salah membaca datanya sebagai kalau NMS-free.\n"
)