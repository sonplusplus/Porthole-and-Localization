import math
from typing import Optional, Tuple

import cv2
import numpy as np

from part_a.calibration import CameraParams
from .schema import DeltaPose, Pose2D, empty_delta, empty_pose

class OrbVisualOdometry:
    """ORB essential-matrix VO baseline with external scale from KITTI GPS/OXTS.

    DeltaPose follows the Phase 3 contract: dx/dy are body-frame motion.
    The accumulated `pose_local` returned by this class is local/world-frame.
    """

    def __init__(
        self,
        cam: CameraParams,
        max_features: int = 2000,
        min_matches: int = 30,
        min_inliers: int = 15,
        ratio_test: float = 0.72,
        fast_threshold: int = 10,
    ) -> None:
        self.cam = cam
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.ratio_test = ratio_test
        self.orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=fast_threshold)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_kp = None
        self.prev_des = None
        self.pose = empty_pose()

    def update(self, frame: np.ndarray, scale_hint: float = 0.0) -> Tuple[Pose2D, DeltaPose]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if self.prev_gray is None or self.prev_des is None or des is None:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return self.pose, empty_delta()

        matches = self._match(self.prev_des, des)
        if len(matches) < self.min_matches:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return self.pose, DeltaPose(0.0, 0.0, 0.0, scale_hint, len(matches), 0, False)

        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
        essential, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            self.cam.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if essential is None or mask is None:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return self.pose, DeltaPose(0.0, 0.0, 0.0, scale_hint, len(matches), 0, False)

        inliers, rot, trans, pose_mask = cv2.recoverPose(essential, pts1, pts2, self.cam.K)
        if int(inliers) < self.min_inliers:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return self.pose, DeltaPose(0.0, 0.0, 0.0, scale_hint, len(matches), int(inliers), False)

        scale = float(scale_hint if scale_hint > 0.01 else 1.0)
        dx_body = float(trans[0, 0]) * scale
        dy_body = float(trans[2, 0]) * scale
        dtheta = _yaw_from_rotation(rot)

        c = math.cos(self.pose.theta)
        s = math.sin(self.pose.theta)
        dx_world = c * dx_body - s * dy_body
        dy_world = s * dx_body + c * dy_body
        self.pose = Pose2D(
            x=self.pose.x + dx_world,
            y=self.pose.y + dy_world,
            theta=_wrap_angle(self.pose.theta + dtheta),
        )
        delta = DeltaPose(
            dx=dx_body,
            dy=dy_body,
            dtheta=dtheta,
            scale=scale,
            matches=len(matches),
            inliers=int(inliers),
            valid=True,
        )
        self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
        return self.pose, delta

    def _match(self, prev_des: np.ndarray, des: np.ndarray):
        raw = self.matcher.knnMatch(prev_des, des, k=2)
        good = []
        for pair in raw:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.ratio_test * n.distance:
                good.append(m)
        return good


def _yaw_from_rotation(rot: np.ndarray) -> float:
    return float(math.atan2(rot[0, 2], rot[2, 2]))


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2 * math.pi) - math.pi)
