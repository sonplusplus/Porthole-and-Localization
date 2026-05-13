import time
import numpy as np
import cv2
import onnxruntime as ort
from ultralytics import YOLO

def test_yolo_pt():
    model = YOLO("models/yolov8m.pt")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    t0 = time.time()
    results = model(dummy, verbose=False)
    ms = (time.time() - t0) * 1000
    print(f"YOLOv8 inference: {ms:.1f}ms")

def test_onnxruntime():
    print("Testing ONNX Runtime CPU...")
    providers = ort.get_available_providers()
    print(f"  Available providers: {providers}")
    assert "CPUExecutionProvider" in providers
    print("CPUExecutionProvider OK")

def test_depth_onnx():
    print("Testing Depth Anything V2 ONNX...")
    try:
        sess = ort.InferenceSession(
            "models/depth_anything_v2_vits.onnx",
            providers=["CPUExecutionProvider"]
        )
        inp = sess.get_inputs()[0]
        dummy = np.random.randn(1, 3, 518, 518).astype(np.float32)
        t0 = time.time()
        out = sess.run(None, {inp.name: dummy})
        ms = (time.time() - t0) * 1000
        print(f"Depth inference: {ms:.1f}ms | output shape: {out[0].shape}")
    except Exception as e:
        print(f"Depth model not ready yet: {e}")

def test_opencv():
    print("Testing OpenCV...")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"OpenCV {cv2.__version__} OK")

if __name__ == "__main__":
    print("=" * 10)
    print("SMOKE TEST — EV Pothole Localization")
    test_opencv()
    test_onnxruntime()
    test_yolo_pt()
    test_depth_onnx()
    print("=" * 10)
    print("Phase 0 complete — sẵn sàng Phase 1!")