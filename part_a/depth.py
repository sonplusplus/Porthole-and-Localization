import cv2
import math
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import Optional, Tuple

from .calibration import CameraParams, IPMTransformer

@dataclass
class PotholeMetrics:
    depth_m:      float
    depth_delta_m: float       # metric difference between pothole and surrounding road
    depth_rel:    float        # độ sâu tương đối so với mặt đường xung quanh [0–1]
    area_m2:      float        
    area_ratio:   float        # tỷ lệ diện tích / khu vực ROI
    severity:     str          # "minor" | "moderate" | "severe"
    severity_idx: int          
    centroid_xy:  Tuple[float, float]   # centroid (X, Y) trong ground frame (m)


class DepthEstimator:
    """
    Depth Anything V2 ViT-S ONNX wrapper.

    Usage:
        de = DepthEstimator("models/depth_anything_v2_vits.onnx", cam)
        depth_m = de.infer_metric(frame)           # (H, W) float32, metres
        metrics = de.estimate_pothole(frame, mask) # PotholeMetrics
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Input size Depth Anything V2 ViT-S
    _INPUT_H = 518
    _INPUT_W = 518

    # Default severity thresholds use area_ratio for backward compatibility.
    _SEV_THRESH_RATIO = [(0.005, "minor", 0), (0.025, "moderate", 1)]
    _SEV_THRESH_M2 = [(0.02, "minor", 0), (0.08, "moderate", 1)]

    def __init__(
        self,
        onnx_path: str,
        cam: CameraParams,
        ipm: Optional[IPMTransformer] = None,
        providers: Optional[list] = None,
    ):
        self.cam = cam
        self.ipm = ipm or IPMTransformer(cam)

        # ONNX session
        _providers = providers or ['CPUExecutionProvider']
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4

        self._session = ort.InferenceSession(
            onnx_path, sess_options=sess_opts, providers=_providers
        )
        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name


    #core logic
    def infer_relative(self, frame: np.ndarray) -> np.ndarray:
        """
        Chạy Depth Anything V2.
        Returns: depth map shape (H, W), dtype float32, relative [0..1]
                 (0 = gần nhất, 1 = xa nhất)
        """
        orig_h, orig_w = frame.shape[:2]
        inp = self._preprocess(frame)
        raw = self._session.run([self._output_name], {self._input_name: inp})[0]

        # raw output: (1, H, W) hoặc (1, 1, H, W)
        depth = raw.squeeze()

        # Normalize về [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)

        # Resize về kích thước gốc
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth.astype(np.float32)

    def infer_metric(self, frame: np.ndarray) -> np.ndarray:
        """
        Trả về metric depth map (m).
        Scale dùng ground-plane assumption + camera height.
        """
        depth_rel = self.infer_relative(frame)
        return self.scale_to_metric(depth_rel, frame.shape[:2])

    def scale_to_metric(
        self,
        depth_rel: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Convert relative depth → metric depth (m).

        Method: Ground plane regression.
        - Sample ~bottom 20% của frame (thường là mặt đường)
        - Tính Z_gt tại các pixel đó từ geometry:
              Z_gt(v) = H_camera / tan(pitch + pixel_angle(v))
        - Fit scale factor: Z_metric = scale * depth_rel + shift
          bằng RANSAC linear regression trên ground samples.
        """
        H, W = frame_shape

        # Start ground sampling below the horizon implied by camera pitch.
        # This keeps sky/traffic near the horizon out of the metric scale fit.
        v_horizon = self.cam.cy - self.cam.fy * math.tan(self.cam.pitch)
        v_horizon = float(np.clip(v_horizon, 0, H - 1))
        v_ground_top = v_horizon + 0.8 * (H - v_horizon)
        v_start = int(np.clip(v_ground_top, H * 0.5, max(H - 10, 0)))

        vs = np.arange(v_start, H).astype(np.float32)
        pixel_angles = np.arctan((vs - self.cam.cy) / self.cam.fy)
        total_angles = self.cam.pitch + pixel_angles

        # Z_gt cho mỗi hàng pixel (ground plane assumption)
        valid = np.abs(total_angles) > 1e-4
        Z_gt = np.where(valid,
                        self.cam.h_camera / np.tan(np.abs(total_angles)),
                        np.nan)
        Z_gt = np.clip(Z_gt, 0.5, 50.0)

        # Depth values trong ground strip. Crop horizontal edges to avoid kerbs,
        # adjacent vehicles, and image borders dominating the fit.
        h_margin = W // 6
        c0 = min(h_margin, max(W - 1, 0))
        c1 = max(c0 + 1, W - h_margin)
        depth_strip = depth_rel[v_start:, c0:c1]
        Z_gt_strip  = Z_gt[:, None].repeat(depth_strip.shape[1], axis=1)

        d_flat = depth_strip.ravel()
        z_flat = Z_gt_strip.ravel()

        #filter outliers (<0.01m or inf)
        mask_valid = (d_flat > 0.01) & np.isfinite(z_flat)
        if mask_valid.sum() < 10:
            scale = self.cam.h_camera / max(float(np.median(depth_rel[v_start:])), 1e-6)
            return (depth_rel * scale).astype(np.float32)

        d_v = d_flat[mask_valid]
        z_v = z_flat[mask_valid]

        # RANSAC linear fit: z = a*d + b
        a, b = self._ransac_linear(d_v, z_v)
        depth_metric = a * depth_rel + b
        depth_metric = np.clip(depth_metric, 0.1, 80.0)
        return depth_metric.astype(np.float32)
    
    #pothole 
    def estimate_pothole(
        self,
        frame: np.ndarray,
        mask: np.ndarray,               
        depth_metric: Optional[np.ndarray] = None,
        severity_mode: str = "area_ratio",
    ) -> PotholeMetrics:
        if depth_metric is None:
            depth_metric = self.infer_metric(frame)

        # Normalize mask → binary
        bin_mask = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask.astype(np.uint8)

        H, W = frame.shape[:2]
        frame_area = H * W

        depth_in_mask = depth_metric[bin_mask == 1]
        if len(depth_in_mask) == 0:
            return PotholeMetrics(0, 0, 0, 0, 0, "minor", 0, (0.0, 0.0))

        depth_mean, road_depth, depth_delta_m = self._estimate_local_depth_delta(depth_metric, bin_mask)
        depth_rel_val = depth_delta_m / (road_depth + 1e-6)

        mask_bev = self.ipm.warp_mask(bin_mask)
        area_m2  = self.ipm.pixel_area_to_m2(mask_bev)
        area_ratio = float(bin_mask.sum()) / frame_area

        M = cv2.moments(bin_mask)
        if M["m00"] > 0:
            cu = M["m10"] / M["m00"]
            cv_ = M["m01"] / M["m00"]
            centroid_xy = self.ipm.image_to_ground(cu, cv_)
        else:
            centroid_xy = (0.0, 0.0)

        severity, sev_idx = self._classify_severity(area_ratio, area_m2, severity_mode)

        return PotholeMetrics(
            depth_m      = depth_mean,
            depth_delta_m= depth_delta_m,
            depth_rel    = depth_rel_val,
            area_m2      = area_m2,
            area_ratio   = area_ratio,
            severity     = severity,
            severity_idx = sev_idx,
            centroid_xy  = centroid_xy,
        )

    def _estimate_local_depth_delta(self, depth_metric: np.ndarray, bin_mask: np.ndarray) -> Tuple[float, float, float]:
        """
        Estimate pothole drop against a local road model.

        The old mean(mask) vs median(ring) comparison is sensitive to mask
        borders and vertical perspective gradients. This method erodes the
        mask to focus on the pothole interior, fits a local plane on the
        surrounding road ring, then compares depths at matching pixel
        locations.
        """
        ys, xs = np.nonzero(bin_mask)
        if len(xs) == 0:
            return 0.0, 1.0, 0.0

        H, W = depth_metric.shape[:2]
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bbox_size = max(x2 - x1 + 1, y2 - y1 + 1)

        ring_size = self._odd_int(np.clip(round(bbox_size * 0.20), 15, 61))
        erode_size = self._odd_int(np.clip(round(bbox_size * 0.08), 3, 21))

        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_size, ring_size))
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))

        core_mask = cv2.erode(bin_mask, erode_kernel)
        min_core_area = max(20, int(0.15 * int(bin_mask.sum())))
        if int(core_mask.sum()) < min_core_area:
            core_mask = bin_mask

        dilated = cv2.dilate(bin_mask, ring_kernel)
        surround = (dilated.astype(bool) & ~bin_mask.astype(bool))

        road_model = self._fit_local_road_plane(depth_metric, surround)
        cy, cx = np.nonzero(core_mask)
        core_depth = depth_metric[cy, cx].astype(np.float32)
        valid_core = np.isfinite(core_depth) & (core_depth > 0)
        if valid_core.sum() == 0:
            depth_in_mask = depth_metric[bin_mask == 1]
            depth_in_mask = depth_in_mask[np.isfinite(depth_in_mask)]
            fallback = float(np.median(depth_in_mask)) if len(depth_in_mask) else 0.0
            return fallback, max(fallback, 1e-6), 0.0

        cy = cy[valid_core]
        cx = cx[valid_core]
        core_depth = core_depth[valid_core]
        road_at_core = road_model(cx, cy)

        valid = np.isfinite(road_at_core) & (road_at_core > 0)
        if valid.sum() == 0:
            road_depth = float(np.median(core_depth))
            return road_depth, road_depth, 0.0

        core_depth = core_depth[valid]
        road_at_core = road_at_core[valid]
        residual = np.abs(core_depth - road_at_core)
        residual = self._trim_percentiles(residual, 5.0, 95.0)

        pothole_depth = float(np.median(core_depth))
        road_depth = float(np.median(road_at_core))
        depth_delta_m = float(np.median(residual)) if len(residual) else abs(road_depth - pothole_depth)
        return pothole_depth, road_depth, depth_delta_m

    def _fit_local_road_plane(self, depth_metric: np.ndarray, surround: np.ndarray):
        sy, sx = np.nonzero(surround)
        z = depth_metric[sy, sx].astype(np.float32)
        valid = np.isfinite(z) & (z > 0)
        sy = sy[valid]
        sx = sx[valid]
        z = z[valid]

        if len(z) < 30:
            road_depth = float(np.median(z)) if len(z) else 1.0
            return lambda x, y: np.full_like(np.asarray(x, dtype=np.float32), road_depth, dtype=np.float32)

        z = self._trim_percentiles(z, 5.0, 95.0)
        if len(z) < 30:
            road_depth = float(np.median(z)) if len(z) else 1.0
            return lambda x, y: np.full_like(np.asarray(x, dtype=np.float32), road_depth, dtype=np.float32)

        # Rebuild coordinates for the same percentile window.
        raw_z = depth_metric[sy, sx].astype(np.float32)
        lo, hi = np.percentile(raw_z, [5.0, 95.0])
        keep = (raw_z >= lo) & (raw_z <= hi)
        sy = sy[keep]
        sx = sx[keep]
        z = raw_z[keep]

        H, W = depth_metric.shape[:2]
        x_norm = (sx.astype(np.float32) - W * 0.5) / max(float(W), 1.0)
        y_norm = (sy.astype(np.float32) - H * 0.5) / max(float(H), 1.0)
        A = np.column_stack([x_norm, y_norm, np.ones_like(x_norm)])

        try:
            coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
            fitted = A @ coeff
            abs_res = np.abs(z - fitted)
            mad = float(np.median(abs_res))
            inlier_thresh = max(0.03, 3.0 * mad)
            inliers = abs_res <= inlier_thresh
            if inliers.sum() >= 30:
                coeff, *_ = np.linalg.lstsq(A[inliers], z[inliers], rcond=None)
        except np.linalg.LinAlgError:
            road_depth = float(np.median(z))
            return lambda x, y: np.full_like(np.asarray(x, dtype=np.float32), road_depth, dtype=np.float32)

        def road_model(x, y):
            x_arr = np.asarray(x, dtype=np.float32)
            y_arr = np.asarray(y, dtype=np.float32)
            x_n = (x_arr - W * 0.5) / max(float(W), 1.0)
            y_n = (y_arr - H * 0.5) / max(float(H), 1.0)
            return (coeff[0] * x_n + coeff[1] * y_n + coeff[2]).astype(np.float32)

        return road_model

    @staticmethod
    def _trim_percentiles(values: np.ndarray, low: float, high: float) -> np.ndarray:
        clean = np.asarray(values, dtype=np.float32)
        clean = clean[np.isfinite(clean)]
        if len(clean) < 4:
            return clean
        lo, hi = np.percentile(clean, [low, high])
        trimmed = clean[(clean >= lo) & (clean <= hi)]
        return trimmed if len(trimmed) else clean

    @staticmethod
    def _odd_int(value: float) -> int:
        size = max(1, int(value))
        return size if size % 2 == 1 else size + 1

    #visualize
    def visualize(
        self,
        frame: np.ndarray,
        depth_metric: np.ndarray,
        mask: Optional[np.ndarray] = None,
        metrics: Optional[PotholeMetrics] = None,
    ) -> np.ndarray:
        """Overlay depth colormap + pothole info lên frame."""
        # Depth colormap
        d_vis = cv2.normalize(depth_metric, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_INFERNO)
        out = cv2.addWeighted(frame, 0.5, d_color, 0.5, 0)

        if mask is not None:
            bin_mask = (mask > 127) if mask.max() > 1 else mask.astype(bool)
            out[bin_mask] = (out[bin_mask] * 0.4 + np.array([0, 0, 200]) * 0.6).astype(np.uint8)

        if metrics is not None:
            SEV_COLOR = {"minor": (0, 255, 0), "moderate": (0, 165, 255), "severe": (0, 0, 255)}
            color = SEV_COLOR.get(metrics.severity, (255, 255, 255))
            txt = (f"Drop:{metrics.depth_delta_m:.2f}m  "
                   f"Area:{metrics.area_m2:.3f}m2  "
                   f"[{metrics.severity.upper()}]")
            cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2, cv2.LINE_AA)
        return out

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR frame → ONNX input tensor (1, 3, 518, 518)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._INPUT_W, self._INPUT_H),
                             interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - self._MEAN) / self._STD
        tensor = tensor.transpose(2, 0, 1)[None]   # (1, 3, H, W)
        return np.ascontiguousarray(tensor)

    @staticmethod
    def _ransac_linear(
        x: np.ndarray, y: np.ndarray,
        n_iter: int = 50, threshold: float = 0.5, min_inliers: int = 20
    ) -> Tuple[float, float]:
        """RANSAC linear regression: y = a*x + b. Returns (a, b)."""
        best_a, best_b, best_count = 1.0, 0.0, 0
        n = len(x)
        rng = np.random.default_rng(42)

        for _ in range(n_iter):
            idx = rng.choice(n, 2, replace=False)
            x1, x2 = x[idx[0]], x[idx[1]]
            y1, y2 = y[idx[0]], y[idx[1]]
            if abs(x2 - x1) < 1e-6:
                continue
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
            inliers = np.abs(y - (a * x + b)) < threshold
            if inliers.sum() > best_count:
                best_count = inliers.sum()
                if inliers.sum() >= min_inliers:
                    # Refit on all inliers
                    best_a = float(np.polyfit(x[inliers], y[inliers], 1)[0])
                    best_b = float(np.polyfit(x[inliers], y[inliers], 1)[1])

        return best_a, best_b

    @classmethod
    def _classify_severity(cls, area_ratio: float, area_m2: float, mode: str = "area_ratio") -> Tuple[str, int]:
        if mode == "area_m2":
            value = area_m2
            thresholds = cls._SEV_THRESH_M2
        elif mode == "area_ratio":
            value = area_ratio
            thresholds = cls._SEV_THRESH_RATIO
        else:
            raise ValueError(f"Unsupported severity_mode: {mode}")

        for thresh, label, idx in thresholds:
            if value < thresh:
                return label, idx
        return "severe", 2

#quick test
if __name__ == '__main__':
    import os, time

    MODEL_PATH = "models/depth_anything_v2_vits.onnx"
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}, using dummy test")
        cam = CameraParams(fx=800, fy=800, cx=640, cy=360,
                           h_camera=1.2, pitch=np.deg2rad(5))
        ipm = IPMTransformer(cam)
        print("IPM + CameraParams instantiated OK (no model)")
    else:
        cam = CameraParams(fx=800, fy=800, cx=640, cy=360,
                           width=1280, height=720,
                           h_camera=1.2, pitch=np.deg2rad(5))
        ipm = IPMTransformer(cam)
        de  = DepthEstimator(MODEL_PATH, cam, ipm)
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        depth = de.infer_metric(frame)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"depth shape: {depth.shape}  min={depth.min():.2f}m  max={depth.max():.2f}m")
        print(f"inference time: {elapsed:.1f}ms")

        mask = np.zeros((720, 1280), dtype=np.uint8)
        mask[500:580, 560:720] = 1
        m = de.estimate_pothole(frame, mask, depth)
        print(f"PotholeMetrics: depth={m.depth_m:.2f}m  area={m.area_m2:.4f}m²  severity={m.severity}")
