# -*- coding: utf-8 -*-
"""
Export OSNet x0.5 ke ONNX -- architecture EXACT-MATCH torchreid kaiyangzhou/osnet.

Architecture verified dari inspeksi state_dict HuggingFace kaiyangzhou/osnet:
  - OSBlock punya 4 skala: conv2a (1 LiteConv), conv2b (2), conv2c (3), conv2d (4)
  - Tiap skala disum (bukan concat), lalu digabung lewat gate (Aggregation Gate)
  - Gate: fc1 [mid -> 4], fc2 [4 -> mid] (squeeze-excitation style, 4 skala)
  - conv3 input = mid (bukan mid*4) karena output AG sudah di-pool ke mid
  - Channels x0.5: conv1=32, conv2/3/4 stages=[128,192,256], conv5=256, fc=512

Jalankan:
    python scripts/export_osnet_x05.py --pth weights\\osnet_x0_5_msmt17.pth

Output:
    weights/osnet_x0_5_msmt17.onnx

Update config.py setelah export:
    OSNET_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'osnet_x0_5_msmt17.onnx')
"""

import os, sys, argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("[ERROR] PyTorch tidak ditemukan.")
    sys.exit(1)


# ---- Building blocks (names match torchreid exactly) -----------------------

class LiteConv(nn.Module):
    """Depthwise-separable lite conv.
    State dict keys: .conv1.weight, .conv2.weight, .bn.{weight,bias,running_*}
    """
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 1, bias=False)          # pointwise
        self.conv2 = nn.Conv2d(c, c, 3, padding=1,           # depthwise
                               groups=c, bias=False)
        self.bn    = nn.BatchNorm2d(c)

    def forward(self, x):
        return F.relu(self.bn(self.conv2(self.conv1(x))), inplace=True)


class ConvLayer(nn.Module):
    """Conv2d + BN2d [+ ReLU].
    State dict keys: .conv.weight, .bn.{weight,bias,running_*}
    """
    def __init__(self, in_c, out_c, k, stride=1, pad=0, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=stride,
                              padding=pad, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self._relu = relu

    def forward(self, x):
        x = self.bn(self.conv(x))
        return F.relu(x, inplace=True) if self._relu else x


class AggregationGate(nn.Module):
    """Omni-scale feature aggregation gate.
    n_scales varies per stage: conv2=2, conv3=3, conv4=4 (= out_c // 64).
    State dict keys: gate.fc1.{weight,bias}, gate.fc2.{weight,bias}
    """
    def __init__(self, mid, n_scales):
        super().__init__()
        self.n = n_scales
        self.fc1 = nn.Conv2d(mid, n_scales, 1, bias=True)
        self.fc2 = nn.Conv2d(n_scales, mid,  1, bias=True)

    def forward(self, feat_list):
        summed = torch.stack(feat_list, dim=1).sum(dim=1)   # [B, mid, H, W]
        w = F.softmax(self.fc1(summed), dim=1)              # [B, n, H, W]
        w = self.fc2(w)                                     # [B, mid, H, W]
        out = torch.zeros_like(feat_list[0])
        for f in feat_list:
            out = out + f * w
        return F.relu(out, inplace=True)


class OSBlock(nn.Module):
    """OSNet omni-scale block with variable scales + aggregation gate.

    n_scales = out_c // 64  (verified from state_dict):
      conv2 (out=128, mid=32): 2 scales -- conv2a (x1), conv2b (x2)
      conv3 (out=192, mid=48): 3 scales -- conv2a (x1), conv2b (x2), conv2c (x3)
      conv4 (out=256, mid=64): 4 scales -- conv2a (x1), conv2b (x2), conv2c (x3), conv2d (x4)
    gate fc1: [n_scales, mid, 1, 1]  fc2: [mid, n_scales, 1, 1]
    conv3 input = mid (after gate aggregation, not concat)
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        mid      = out_c // 4
        n_scales = out_c // 64   # 128//64=2, 192//64=3, 256//64=4

        self.conv1 = ConvLayer(in_c, mid, 1)
        # Build exactly n_scales parallel streams
        self.conv2a = LiteConv(mid)                                           # scale 1: depth=1
        if n_scales >= 2:
            self.conv2b = nn.Sequential(LiteConv(mid), LiteConv(mid))        # scale 2: depth=2
        if n_scales >= 3:
            self.conv2c = nn.Sequential(LiteConv(mid), LiteConv(mid), LiteConv(mid))  # scale 3
        if n_scales >= 4:
            self.conv2d = nn.Sequential(LiteConv(mid), LiteConv(mid),        # scale 4: depth=4
                                         LiteConv(mid), LiteConv(mid))
        self.gate  = AggregationGate(mid, n_scales)
        self.conv3 = ConvLayer(mid, out_c, 1, relu=False)   # input=mid (after gate, not concat)

        self.downsample = None
        if in_c != out_c:
            self.downsample = ConvLayer(in_c, out_c, 1, relu=False)

        self._n_scales = n_scales

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x1 = self.conv1(x)
        scales = [self.conv2a(x1)]
        if self._n_scales >= 2:
            scales.append(self.conv2b(x1))
        if self._n_scales >= 3:
            scales.append(self.conv2c(x1))
        if self._n_scales >= 4:
            scales.append(self.conv2d(x1))
        agg = self.gate(scales)
        out = self.conv3(agg)
        return F.relu(out + identity, inplace=True)


class OSNetx0_5(nn.Module):
    """OSNet x0.5 -- channels: stem=32, stages=[128,192,256], head=256, fc=512."""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.conv1       = ConvLayer(3, 32, 7, stride=2, pad=3)
        self.maxpool     = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2       = nn.Sequential(OSBlock(32,  128), OSBlock(128, 128))
        self.pool2       = nn.MaxPool2d(2, stride=2)
        self.conv3       = nn.Sequential(OSBlock(128, 192), OSBlock(192, 192))
        self.pool3       = nn.MaxPool2d(2, stride=2)
        self.conv4       = nn.Sequential(OSBlock(192, 256), OSBlock(256, 256))

        self.conv5       = ConvLayer(256, 256, 1)
        self.global_avg  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc          = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_avg(x).view(x.size(0), -1)
        v = self.fc(x)
        return F.normalize(v, p=2, dim=1)


# ---- Export ----------------------------------------------------------------

def export(pth_path, out_path, feature_dim=512):
    print("=" * 60)
    print("OSNet x0.5 -> ONNX Export (torchreid-exact architecture)")
    print("=" * 60)

    model = OSNetx0_5(feature_dim=feature_dim)
    model.eval()

    if pth_path and os.path.exists(pth_path):
        print(f"[INFO] Loading: {pth_path}")
        raw   = torch.load(pth_path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw)
        state = {k.replace("module.", ""): v
                 for k, v in state.items()
                 if not k.startswith("classifier")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] {len(missing)} missing keys (first 5):")
            for k in missing[:5]:
                print(f"  {k}")
        print(f"[OK] Loaded. Unexpected (dropped): {len(unexpected)}")
    else:
        print("[WARN] No .pth file - RANDOM weights!")

    dummy = torch.zeros(1, 3, 256, 128)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"[INFO] Exporting to: {out_path}")

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12, do_constant_folding=True,
    )
    mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[OK] Exported: {out_path}  ({mb:.1f} MB)")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        x    = np.zeros((1, 3, 256, 128), dtype=np.float32)
        out  = sess.run(None, {"input": x})
        norm = float(np.linalg.norm(out[0][0]))
        print(f"[OK] Inference: shape={out[0].shape}, L2-norm={norm:.4f} (should be ~1.0)")
    except Exception as e:
        print(f"[WARN] Inference check: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pth", default=os.path.join(BASE_DIR, "weights", "osnet_x0_5_msmt17.pth"))
    p.add_argument("--out", default=os.path.join(BASE_DIR, "weights", "osnet_x0_5_msmt17.onnx"))
    p.add_argument("--feature-dim", type=int, default=512)
    args = p.parse_args()
    export(args.pth, args.out, args.feature_dim)
    print()
    print("-" * 60)
    print("Next: update config.py ->")
    print("  OSNET_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'osnet_x0_5_msmt17.onnx')")
    print("-" * 60)
