import os
import sys
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def ensure_weights_dir():
    weights_dir = os.path.join(config.BASE_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    return weights_dir

def export_yolo11n():
    """Export YOLO11n model to ONNX format using Ultralytics."""
    try:
        from ultralytics import YOLO
        weights_dir = ensure_weights_dir()
        pt_path = config.YOLO_PT_PATH
        onnx_path = config.YOLO_MODEL_PATH

        print(f"[export_models] Loading YOLO11n model from ultralytics...")
        model = YOLO("yolo11n.pt")
        
        # Save PyTorch weights if not present
        if not os.path.exists(pt_path):
            model.save(pt_path)
            
        # Export to ONNX if ONNX file does not exist
        if not os.path.exists(onnx_path):
            print(f"[export_models] Exporting YOLO11n to ONNX format at {onnx_path}...")
            model.export(format="onnx", imgsz=640, simplify=True)
            # Ultralytics exports to current dir or next to pt, let's move/verify path
            default_onnx = pt_path.replace(".pt", ".onnx")
            if os.path.exists(default_onnx) and default_onnx != onnx_path:
                os.rename(default_onnx, onnx_path)
        print("[export_models] YOLO11n ONNX export complete!")
    except Exception as e:
        print(f"[export_models] Warning: Could not export YOLO11n ONNX: {e}")
        print("[export_models] Will fallback to PyTorch YOLO inference.")

def main():
    ensure_weights_dir()
    print("=== Model Downloader & ONNX Exporter ===")
    export_yolo11n()

if __name__ == "__main__":
    main()
