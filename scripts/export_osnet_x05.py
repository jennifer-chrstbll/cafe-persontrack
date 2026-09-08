"""
Export OSNet x0.5 ke ONNX untuk dipakai sebagai ReID model yang lebih akurat.

OSNet x0.5 channels: [32, 128, 192, 256]  (vs x0.25: [16, 64, 96, 128])
- ~2x lebih besar dari x0.25
- mAP MSMT17: ~29% vs ~23% untuk x0.25
- Inference time di CPU: ~2-3ms vs ~1ms per crop (masih cukup cepat di RPi5)

Jalankan:
    python scripts/export_osnet_x05.py

Output:
    weights/osnet_x0_5_msmt17.onnx
    
Setelah export, ubah config.py:
    OSNET_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'osnet_x0_5_msmt17.onnx')
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("[ERROR] PyTorch tidak ditemukan. Install: pip install torch")
    sys.exit(1)


# ─── OSNet x0.5 Architecture ────────────────────────────────────────────────
# Channel config: [32, 128, 192, 256] — 2x dari x0.25 [16, 64, 96, 128]

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1,
                              stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class OSBlock(nn.Module):
    """Omni-Scale Block (full version, dipakai di x0.5 ke atas)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 4
        self.conv1 = Conv1x1(in_channels, mid)
        # Dua skala omni: 3x3 normal + 3x3 depthwise (untuk tangkap konteks multi-skala)
        self.conv2a = ConvLayer(mid, mid, 3, padding=1)
        self.conv2b = ConvLayer(mid, mid, 3, padding=1, groups=mid)
        self.conv2c = ConvLayer(mid, mid, 3, padding=1)
        self.conv3  = Conv1x1(mid * 3, out_channels)
        self.gate   = nn.Sequential(nn.Linear(out_channels, out_channels // 4),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(out_channels // 4, out_channels),
                                     nn.Sigmoid())
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1(in_channels, out_channels)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x1  = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        out = self.conv3(torch.cat([x2a, x2b, x2c], dim=1))
        # Channel-wise attention gate
        gap = out.mean(dim=[2, 3])
        w   = self.gate(gap).unsqueeze(-1).unsqueeze(-1)
        out = out * w
        return F.relu(out + identity)


class OSNetx0_5(nn.Module):
    """OSNet-x0_5: width multiplier 0.5, feature_dim=512."""
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        # Channels: stem=32, layers=[128, 192, 256]
        self.conv1       = ConvLayer(3,  32, 7, stride=2, padding=3)
        self.maxpool     = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1      = nn.Sequential(OSBlock(32,  128), OSBlock(128, 128))
        self.layer2      = nn.Sequential(OSBlock(128, 192), OSBlock(192, 192))
        self.layer3      = nn.Sequential(OSBlock(192, 256), OSBlock(256, 256))
        self.conv2       = Conv1x1(256, 512)
        self.global_avg  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc          = nn.Linear(512, feature_dim)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        x = self.global_avg(x)
        x = x.view(x.size(0), -1)
        v = self.fc(x)
        return F.normalize(v, p=2, dim=1)


def export_onnx(pth_path: str | None, out_path: str, feature_dim: int = 512):
    print("=" * 60)
    print("  OSNet x0.5 -> ONNX Export")
    print("=" * 60)

    model = OSNetx0_5(feature_dim=feature_dim)
    model.eval()

    # Load pre-trained weights jika ada
    if pth_path and os.path.exists(pth_path):
        print(f"[INFO] Loading weights dari: {pth_path}")
        state = torch.load(pth_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Hapus prefix 'module.' jika ada (dari DataParallel training)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] Missing keys: {len(missing)} — model mungkin dari arsitektur berbeda")
        print(f"[INFO] Weights loaded.")
    else:
        print("[WARN] Tidak ada .pth weights — menggunakan bobot random (HANYA untuk test export).")
        print("[WARN] Model tanpa pre-trained weights TIDAK akan akurat untuk ReID!")

    # Dummy input: batch=1, 3 channel, 256x128 (standard ReID crop size)
    dummy = torch.zeros(1, 3, 256, 128)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[INFO] Exporting ke: {out_path}")

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[OK] Export selesai! File: {out_path} ({size_mb:.1f} MB)")

    # Verifikasi output shape
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        dummy_np = np.zeros((1, 3, 256, 128), dtype=np.float32)
        out = sess.run(None, {"input": dummy_np})
        print(f"[OK] Verifikasi ONNX: output shape = {out[0].shape}  (harusnya (1, {feature_dim}))")
    except Exception as e:
        print(f"[WARN] Verifikasi gagal: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export OSNet x0.5 ke ONNX")
    parser.add_argument(
        "--pth",
        default=None,
        help="Path ke .pth/.pt pre-trained weights OSNet x0.5 (opsional). "
             "Kalau tidak ada, export architecture-only (bobot random)."
    )
    parser.add_argument(
        "--out",
        default=os.path.join(BASE_DIR, "weights", "osnet_x0_5_msmt17.onnx"),
        help="Output path file .onnx"
    )
    parser.add_argument("--feature-dim", type=int, default=512)
    args = parser.parse_args()

    export_onnx(args.pth, args.out, args.feature_dim)

    print()
    print("-" * 60)
    print("Langkah selanjutnya:")
    print("1. Edit config.py:")
    print("   OSNET_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'osnet_x0_5_msmt17.onnx')")
    print("2. Run ulang: python process_video.py --video cctv_test.mp4 --no-show")
    print("-" * 60)
