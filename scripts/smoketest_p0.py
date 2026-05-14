from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


POTHOLE_ONNX = Path("models/yolov8s_pothole.onnx")
DEPTH_ONNX = Path("models/depth_anything_v2_vits.onnx")


def test_yolo_onnx():
    print("Testing YOLOv8 pothole ONNX on CPU...")
    if not POTHOLE_ONNX.exists():
        raise FileNotFoundError(f"Missing pothole ONNX model: {POTHOLE_ONNX}")

    model = YOLO(str(POTHOLE_ONNX), task="segment")
    dummy = np.zeros((448, 448, 3), dtype=np.uint8)
    t0 = time.time()
    results = model.predict(dummy, imgsz=448, device="cpu", verbose=False)
    ms = (time.time() - t0) * 1000
    print(f"YOLOv8 ONNX CPU inference: {ms:.1f}ms | results={len(results)}")


def test_onnxruntime():
    print("Testing ONNX Runtime CPU...")
    providers = ort.get_available_providers()
    print(f"  Available providers: {providers}")
    assert "CPUExecutionProvider" in providers
    gpu_like = [provider for provider in providers if provider != "CPUExecutionProvider"]
    if gpu_like:
        print(f"  Non-CPU providers are installed but this smoke test forces CPU: {gpu_like}")
    print("CPUExecutionProvider OK")


def test_depth_onnx():
    print("Testing Depth Anything V2 ONNX...")
    if not DEPTH_ONNX.exists():
        raise FileNotFoundError(f"Missing depth ONNX model: {DEPTH_ONNX}")

    sess = ort.InferenceSession(
        str(DEPTH_ONNX),
        providers=["CPUExecutionProvider"],
    )
    inp = sess.get_inputs()[0]
    dummy = np.random.randn(1, 3, 518, 518).astype(np.float32)
    t0 = time.time()
    out = sess.run(None, {inp.name: dummy})
    ms = (time.time() - t0) * 1000
    print(f"Depth ONNX CPU inference: {ms:.1f}ms | output shape: {out[0].shape}")


def test_opencv():
    print("Testing OpenCV...")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"OpenCV {cv2.__version__} OK | gray_shape={gray.shape}")


if __name__ == "__main__":
    print("=" * 10)
    print("SMOKE TEST - EV Pothole Localization")
    test_opencv()
    test_onnxruntime()
    test_yolo_onnx()
    test_depth_onnx()
    print("=" * 10)
    print("Phase 0 complete - ready for Phase 1.")
