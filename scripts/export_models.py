import os
import sys
import shutil

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

        print("[export_models] Loading YOLO11n model...")
        model = YOLO("yolo11n.pt")
        if not os.path.exists(pt_path):
            model.save(pt_path)

        if not os.path.exists(onnx_path):
            print(f"[export_models] Exporting YOLO11n to ONNX: {onnx_path}...")
            exported = model.export(format="onnx", imgsz=640, simplify=True, dynamic=True)
            if str(exported) != onnx_path and os.path.exists(str(exported)):
                shutil.move(str(exported), onnx_path)
        print("[export_models] YOLO11n ONNX ready!")
    except Exception as e:
        print(f"[export_models] Warning: YOLO11n export error: {e}")

def export_yolo26n():
    """Export YOLO26n model to ONNX format using Ultralytics."""
    try:
        from ultralytics import YOLO
        weights_dir = ensure_weights_dir()
        pt_path = getattr(config, 'YOLO26_PT_PATH', os.path.join(weights_dir, 'yolo26n.pt'))
        onnx_path = getattr(config, 'YOLO26_MODEL_PATH', os.path.join(weights_dir, 'yolo26n.onnx'))

        print("[export_models] Loading YOLO26n model...")
        model = YOLO("yolo26n.pt")
        if not os.path.exists(pt_path):
            model.save(pt_path)

        if not os.path.exists(onnx_path):
            print(f"[export_models] Exporting YOLO26n to ONNX: {onnx_path}...")
            exported = model.export(format="onnx", imgsz=640, simplify=True, dynamic=True)
            if str(exported) != onnx_path and os.path.exists(str(exported)):
                shutil.move(str(exported), onnx_path)
        print("[export_models] YOLO26n ONNX ready!")
    except Exception as e:
        print(f"[export_models] Warning: YOLO26n export error: {e}")

def main():
    ensure_weights_dir()
    print("=== Model Downloader & ONNX Exporter ===")
    export_yolo11n()
    export_yolo26n()

if __name__ == "__main__":
    main()
