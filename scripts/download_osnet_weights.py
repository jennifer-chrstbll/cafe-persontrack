import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# Pretrained OSNet-x0_25 MSMT17 weights URL (HuggingFace)
OSNET_MSMT17_URL = "https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt"

def download_osnet_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    pth_path = os.path.join(WEIGHTS_DIR, "osnet_x0_25_msmt17.pth")

    if not os.path.exists(pth_path) or os.path.getsize(pth_path) < 1000000:
        print(f"[download_osnet] Downloading official MSMT17 pretrained OSNet-x0_25 weights...")
        try:
            urllib.request.urlretrieve(OSNET_MSMT17_URL, pth_path)
            print(f"[download_osnet] Download complete! Path: {pth_path} (Size: {os.path.getsize(pth_path)} bytes)")
        except Exception as e:
            print(f"[download_osnet] Download failed: {e}")
            return False
    else:
        print(f"[download_osnet] Pretrained OSNet-x0_25 weights ready at {pth_path}")

    return True

if __name__ == "__main__":
    download_osnet_weights()
