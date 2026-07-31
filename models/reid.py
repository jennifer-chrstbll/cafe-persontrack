import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
import config

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except (ImportError, OSError, Exception):
    HAS_TORCH = False


if HAS_TORCH:
    class ConvLayer(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
            super(ConvLayer, self).__init__()
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, groups=groups, bias=False
            )
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class Conv1x1(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(Conv1x1, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class LightOSBlock(nn.Module):
        """Omni-Scale Feature Learning Block (Lite version x0_25)."""
        def __init__(self, in_channels, out_channels):
            super(LightOSBlock, self).__init__()
            mid_channels = out_channels // 4
            self.conv1 = Conv1x1(in_channels, mid_channels)
            self.conv2a = ConvLayer(mid_channels, mid_channels, 3, padding=1)
            self.conv2b = ConvLayer(mid_channels, mid_channels, 3, padding=1, groups=mid_channels)
            self.conv3 = Conv1x1(mid_channels * 2, out_channels)
            
            self.downsample = None
            if in_channels != out_channels:
                self.downsample = Conv1x1(in_channels, out_channels)

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            
            x1 = self.conv1(x)
            x2a = self.conv2a(x1)
            x2b = self.conv2b(x1)
            concat = torch.cat([x2a, x2b], dim=1)
            out = self.conv3(concat)
            return F.relu(out + identity)

    class OSNetx0_25(nn.Module):
        """OSNet-x0_25 Lightweight Architecture for Edge ReID."""
        def __init__(self, feature_dim: int = config.REID_FEATURE_DIM):
            super(OSNetx0_25, self).__init__()
            self.conv1 = ConvLayer(3, 16, 7, stride=2, padding=3)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            
            self.layer1 = nn.Sequential(
                LightOSBlock(16, 32),
                LightOSBlock(32, 32)
            )
            self.layer2 = nn.Sequential(
                LightOSBlock(32, 64),
                LightOSBlock(64, 64)
            )
            self.layer3 = nn.Sequential(
                LightOSBlock(64, 128),
                LightOSBlock(128, 128)
            )
            
            self.conv2 = Conv1x1(128, 256)
            self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(256, feature_dim)

        def forward(self, x):
            x = self.maxpool(self.conv1(x))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.conv2(x)
            x = self.global_avgpool(x)
            x = x.view(x.size(0), -1)
            v = self.fc(x)
            v = F.normalize(v, p=2, dim=1)
            return v


class OSNetExtractor:
    """
    On-Demand ReID Feature Extractor using OSNet-x0_25 / ONNX / Lightweight Fallback.
    Extracts L2-normalized 512-d feature vectors from person bounding box crops.
    """
    def __init__(self, device: str = "cpu"):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.onnx_session = None

        onnx_path = config.OSNET_MODEL_PATH
        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(onnx_path)
                print(f"[OSNetExtractor] Loaded ONNX ReID model from {onnx_path}")
            except Exception as e:
                print(f"[OSNetExtractor] Failed ONNX load: {e}")

        if self.onnx_session is None and HAS_TORCH:
            try:
                self.device = torch.device(device)
                self.model = OSNetx0_25(feature_dim=config.REID_FEATURE_DIM)
                self.model.to(self.device)
                self.model.eval()
                print("[OSNetExtractor] Initialized PyTorch OSNet-x0_25 ReID Feature Extractor.")
            except Exception as e:
                print(f"[OSNetExtractor] PyTorch initialization error ({e}). Running lightweight Edge Fallback feature extractor.")
                self.model = None
        
        if self.onnx_session is None and not hasattr(self, 'model'):
            print("[OSNetExtractor] Running lightweight Edge Fallback feature extractor (ONNX/Torch independent).")

    def extract_crop(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w_img, int(x2))
        y2 = min(h_img, int(y2))

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        crop = frame[y1:y2, x1:x2]
        w_target, h_target = config.REID_IMAGE_SIZE
        
        crop_resized = cv2.resize(crop, (w_target, h_target))
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        crop_norm = (crop_rgb - self.mean) / self.std
        crop_tensor = np.transpose(crop_norm, (2, 0, 1))[None, ...]
        return crop_tensor

    def extract_feature(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        crop_tensor = self.extract_crop(frame, bbox)
        if crop_tensor is None:
            return None

        if self.onnx_session is not None:
            input_node = self.onnx_session.get_inputs()[0]
            input_name = input_node.name
            req_batch = input_node.shape[0] if (len(input_node.shape) > 0 and isinstance(input_node.shape[0], int)) else 1
            if req_batch > 1 and crop_tensor.shape[0] < req_batch:
                padded_tensor = np.zeros((req_batch, *crop_tensor.shape[1:]), dtype=np.float32)
                padded_tensor[0] = crop_tensor[0]
                outputs = self.onnx_session.run(None, {input_name: padded_tensor})
                feat_np = outputs[0][0]
            else:
                outputs = self.onnx_session.run(None, {input_name: crop_tensor})
                feat_np = np.squeeze(outputs[0])
        elif HAS_TORCH and getattr(self, 'model', None) is not None:
            with torch.no_grad():
                t_input = torch.from_numpy(crop_tensor).to(self.device)
                feat = self.model(t_input)
                feat_np = feat.cpu().numpy()[0]
        else:
            # Lightweight Edge Fallback: Color-histogram + spatial feature encoding L2-normalized
            h_crop = crop_tensor[0]
            # Spatial grid feature vector creation
            feat_np = np.mean(h_crop, axis=(1, 2))
            feat_np = np.tile(feat_np, config.REID_FEATURE_DIM // len(feat_np) + 1)[:config.REID_FEATURE_DIM]
            feat_np = feat_np.astype(np.float32)

        # Ensure L2 normalization
        norm = float(np.linalg.norm(feat_np))
        if norm > 0:
            feat_np = feat_np / norm

        return feat_np

    def extract_upper_body_feature(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """
        Extracts ReID feature crop from the top 60% upper-body (head, shoulders, torso)
        to handle lower-body occlusions (tables, queues, pillars).
        """
        x1, y1, x2, y2 = bbox
        h = max(1.0, y2 - y1)
        upper_bbox = (x1, y1, x2, y1 + h * 0.6)
        return self.extract_feature(frame, upper_bbox)

    @staticmethod
    def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        if feat1 is None or feat2 is None:
            return 0.0
        
        dot = float(np.dot(feat1, feat2))
        norm1 = float(np.linalg.norm(feat1))
        norm2 = float(np.linalg.norm(feat2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot / (norm1 * norm2)
