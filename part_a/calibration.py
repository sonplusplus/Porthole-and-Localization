import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class CameraParams:
    """Intrinsic + mounting params cho 1 camera monocular."""
    fx: float = 800.0       # focal length x
    fy: float = 800.0       # focal length y
    cx: float = 640.0       # principal point x 
    cy: float = 360.0       # principal point y  
    width:  int = 1280
    height: int = 720

    # Distortion coefficients [k1, k2, p1, p2, k3]
    dist_coeffs: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float64)
    )

    h_camera: float = 1.20   
    pitch:    float = 0.0    
    roll:     float = 0.0   

    @property
    def K(self) -> np.ndarray:
        return np.array([
            [self.fx,  0,       self.cx],
            [0,        self.fy, self.cy],
            [0,        0,       1      ]
        ], dtype=np.float64)

    @property
    def K_inv(self) -> np.ndarray:
        return np.linalg.inv(self.K)


class IPMTransformer:
    """
    Coordinate system BEV:
        - Origin = điểm ngay dưới camera trên mặt đường
        - +X = right 
        - +Y = forward 
    """

    # BEV output canvas (pixels)
    BEV_W = 400   # horizontal  (covers ±bev_range_x m)
    BEV_H = 600   # vertical    (covers 0 → bev_range_y m forward)

    def __init__(
        self,
        cam: CameraParams,
        bev_range_x: float = 5.0,   # metres bên trái/phải
        bev_range_y: float = 15.0,  # metres về phía trước
    ):
        self.cam = cam
        self.bev_range_x = bev_range_x
        self.bev_range_y = bev_range_y

        # metres per pixel trong BEV
        self.mx = (2 * bev_range_x) / self.BEV_W   # m/px horizontal
        self.my = bev_range_y / self.BEV_H           # m/px vertical

        # Tính homography H: image => BEV
        self._H, self._H_inv = self._compute_homography()


    @property
    def meters_per_pixel(self) -> Tuple[float, float]:
        """(mx, my) — metres per pixel trong BEV image."""
        return self.mx, self.my

    def warp(self, frame: np.ndarray) -> np.ndarray:
        """Warp frame gốc → BEV image (H×W×3)."""
        return cv2.warpPerspective(
            frame, self._H, (self.BEV_W, self.BEV_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

    def warp_mask(self, mask: np.ndarray) -> np.ndarray:
        """Warp binary mask gốc → BEV mask."""
        return cv2.warpPerspective(
            mask.astype(np.uint8), self._H,
            (self.BEV_W, self.BEV_H),
            flags=cv2.INTER_NEAREST
        )

    def pixel_area_to_m2(self, mask_bev: np.ndarray) -> float:
        """Tính diện tích thực (m²) từ binary mask trong BEV space."""
        n_px = float(np.count_nonzero(mask_bev))
        return n_px * self.mx * self.my

    def image_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        """
        Chuyển điểm ảnh gốc (u, v) → tọa độ mặt đường (X, Y) theo m.
        Dựa trên ground-plane assumption + camera height.
        """
        # Undistort point
        pt = cv2.undistortPoints(
            np.array([[[u, v]]], dtype=np.float32),
            self.cam.K, self.cam.dist_coeffs, P=self.cam.K
        )[0][0]
        u_n, v_n = pt

        # Normalised image ray
        ray = self.cam.K_inv @ np.array([u_n, v_n, 1.0])

        # Rotate by pitch
        R_pitch = _rot_x(self.cam.pitch)
        ray_w = R_pitch @ ray

        # Intersect with ground plane Z=0 in camera-world frame
        # Camera is at height h above ground; looking along -Z (OpenCV convention)
        # Plane equation: Y_world = -h  (camera Y-axis points down)
        if abs(ray_w[1]) < 1e-6:
            return 0.0, 0.0   # parallel to ground
        t = self.cam.h_camera / ray_w[1]
        X = t * ray_w[0]
        Y = t * ray_w[2]   # forward
        return float(X), float(Y)

    def bev_pixel_to_ground(self, bev_u: float, bev_v: float) -> Tuple[float, float]:
        """BEV pixel → ground (X, Y) m."""
        X = (bev_u - self.BEV_W / 2) * self.mx
        Y = (self.BEV_H - bev_v) * self.my
        return X, Y

    #homography computation
    def _compute_homography(self):
        """
        Compute H bằng cách chọn 4 điểm đặc trưng trên mặt đường:
          - 2 điểm gần (y_near m trước xe)
          - 2 điểm xa  (y_far  m trước xe)
        và map sang BEV canvas tương ứng.
        """
        y_near = 2.0
        y_far  = min(10.0, self.bev_range_y * 0.85)
        x_half = self.bev_range_x * 0.7

        # Ground points (X, Y) m → image (u, v)
        ground_pts = np.float32([
            [-x_half, y_near],
            [ x_half, y_near],
            [ x_half, y_far ],
            [-x_half, y_far ],
        ])
        img_pts = np.float32([
            self._ground_to_image(gp) for gp in ground_pts
        ])

        def g2bev(gp):
            bev_u = (gp[0] + self.bev_range_x) / (2 * self.bev_range_x) * self.BEV_W
            bev_v = self.BEV_H - (gp[1] / self.bev_range_y) * self.BEV_H
            return [bev_u, bev_v]

        bev_pts = np.float32([g2bev(gp) for gp in ground_pts])

        H, _ = cv2.findHomography(img_pts, bev_pts)
        H_inv, _ = cv2.findHomography(bev_pts, img_pts)
        return H, H_inv

    def _ground_to_image(self, ground_xy: np.ndarray) -> Tuple[float, float]:
        """Ground (X, Y) m → image (u, v) pixel."""
        X, Y = ground_xy
        # Camera ở độ cao h, ground plane cách camera h m theo -Y camera
        R_pitch = _rot_x(-self.cam.pitch)   # inverse pitch
        world = np.array([X, -self.cam.h_camera, Y])   # OpenCV cam coords
        cam = R_pitch @ world
        if cam[2] <= 0:
            return (self.cam.cx, self.cam.cy)
        u = self.cam.fx * cam[0] / cam[2] + self.cam.cx
        v = self.cam.fy * cam[1] / cam[2] + self.cam.cy
        return (float(u), float(v))


class EpipolarMonitor:
    """
    (Optional — Bonus C2) Cảnh báo khi camera bị lệch calibration do xe rung.
    Estimate pitch drift bằng cách theo dõi horizon line qua frame.
    """

    def __init__(self, cam: CameraParams, alert_threshold_deg: float = 2.0):
        self.cam = cam
        self.threshold_rad = np.deg2rad(alert_threshold_deg)
        self._baseline_pitch: Optional[float] = None

    def update(self, frame: np.ndarray) -> dict:
        """
        Ước lượng pitch hiện tại từ horizon detection.
        Returns: {'pitch_est': float, 'drift_deg': float, 'alert': bool}
        """
        pitch_est = self._estimate_pitch_from_horizon(frame)
        if self._baseline_pitch is None:
            self._baseline_pitch = pitch_est

        drift = abs(pitch_est - self._baseline_pitch)
        return {
            'pitch_est': pitch_est,
            'drift_deg': np.rad2deg(drift),
            'alert': drift > self.threshold_rad
        }

    def _estimate_pitch_from_horizon(self, frame: np.ndarray) -> float:
        """Dùng Hough lines trên vùng sky để tìm vanishing point / horizon."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi = gray[:h // 3, :]   # top third
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40,
                                minLineLength=w // 5, maxLineGap=20)
        if lines is None:
            return self.cam.pitch

        # Horizontal lines → estimate horizon row
        horizon_rows = []
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = abs(np.arctan2(y2 - y1, x2 - x1))
            if angle < np.deg2rad(10):   # near-horizontal
                horizon_rows.append((y1 + y2) / 2)

        if not horizon_rows:
            return self.cam.pitch

        horizon_v = np.median(horizon_rows)
        # pitch = arctan((cy - horizon_v) / fy)
        pitch_est = np.arctan((self.cam.cy - horizon_v) / self.cam.fy)
        return float(pitch_est)

#helpers
def _rot_x(angle_rad: float) -> np.ndarray:
    """Rotation matrix around X-axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ], dtype=np.float64)


def load_calibration_from_yaml(path: str) -> CameraParams:
    """
    Load calibration từ file YAML (output của cv2.calibrateCamera).
    Dùng khi đã chạy checkerboard calibration thực tế.
    """
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    K = np.array(data['camera_matrix']['data']).reshape(3, 3)
    D = np.array(data['distortion_coefficients']['data'])
    h_cam = data.get('h_camera', 1.2)
    pitch = np.deg2rad(data.get('pitch_deg', 0.0))
    return CameraParams(
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        width=data.get('width', 1280),
        height=data.get('height', 720),
        dist_coeffs=D,
        h_camera=h_cam,
        pitch=pitch,
    )
#quick test
if __name__ == '__main__':
    cam = CameraParams(
        fx=800, fy=800, cx=640, cy=360,
        width=1280, height=720,
        h_camera=1.2,
        pitch=np.deg2rad(5),
    )
    ipm = IPMTransformer(cam)

    # Test ground projection
    X, Y = ipm.image_to_ground(640, 500)   # bottom-center
    print(f"image(640,500) → ground ({X:.2f}m, {Y:.2f}m)")

    # Test BEV warp 
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    bev = ipm.warp(dummy)
    print(f"BEV output shape: {bev.shape}")

    #mock mask, area test
    dummy_mask = np.zeros((IPMTransformer.BEV_H, IPMTransformer.BEV_W), dtype=np.uint8)
    dummy_mask[200:250, 180:220] = 1   
    area = ipm.pixel_area_to_m2(dummy_mask)
    mx, my = ipm.meters_per_pixel
    print(f"Mask area: {area:.4f} m²  (mx={mx:.4f}, my={my:.4f} m/px)")
