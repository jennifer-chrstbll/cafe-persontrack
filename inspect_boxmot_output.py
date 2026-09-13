"""
Jalankan ini untuk melihat PERSIS apa yang dikembalikan oleh BotSort.update()
di versi boxmot yang ter-install, sebelum kita percaya wrapper kita benar.
"""
import numpy as np
from pathlib import Path
from boxmot import BotSort
import config

print(f"OSNET_ACTIVE_PATH = {config.OSNET_ACTIVE_PATH}")
print(f"File exists? {Path(config.OSNET_ACTIVE_PATH).exists()}")

tracker = BotSort(
    reid_weights=Path(config.OSNET_ACTIVE_PATH),
    device="cpu",
    half=False,
)

# Dummy frame: 640x480 hitam
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Dummy detection: satu "orang" kotak di tengah, format Nx6 [x1,y1,x2,y2,conf,cls]
dets = np.array([[200, 150, 300, 400, 0.9, 0]], dtype=np.float32)

result = tracker.update(dets, frame)

print("\n--- HASIL ---")
print("Tipe object hasil:", type(result))
print("Isi mentah:", result)

if hasattr(result, "shape"):
    print("Shape:", result.shape)
if hasattr(result, "__len__"):
    print("Panjang:", len(result))
    if len(result) > 0:
        first = result[0]
        print("Tipe elemen pertama:", type(first))
        print("Isi elemen pertama:", first)
        if hasattr(first, "__dict__"):
            print("Attribute elemen pertama:", vars(first))
        if hasattr(first, "__dataclass_fields__"):
            print("Dataclass fields:", first.__dataclass_fields__.keys())