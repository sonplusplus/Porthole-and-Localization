import math
import unittest
from typing import Optional

from part_b.dead_reckoning import WheelImuDeadReckoner, choose_motion_delta, integrate_body_delta
from part_b.gps import GpsIntegrityMonitor
from part_b.handover import GpsHandoverManager, GpsLossSimulator
from part_b.schema import DeltaPose, GpsSample, Pose2D, ground_delta_from_points


def gps_sample(valid: bool = True, hdop: Optional[float] = 1.0, satellites: Optional[int] = 8) -> GpsSample:
    return GpsSample(
        lat=10.0,
        lon=106.0,
        alt=5.0,
        speed_mps=8.0,
        hdop=hdop,
        satellites=satellites,
        valid=valid,
    )


class PartBCoreTests(unittest.TestCase):
    def test_gps_integrity_monitor_degrades_before_lost(self) -> None:
        monitor = GpsIntegrityMonitor(lost_after_invalid=2)

        self.assertEqual(monitor.update(gps_sample()), "good")
        self.assertEqual(monitor.update(gps_sample(valid=False)), "degraded")
        self.assertEqual(monitor.update(gps_sample(valid=False)), "lost")
        self.assertEqual(monitor.update(gps_sample(hdop=8.0)), "degraded")

    def test_gps_loss_simulator_has_degraded_window_then_loss(self) -> None:
        simulator = GpsLossSimulator(start_frame=10, end_frame=12, degraded_frames=1)

        self.assertTrue(simulator.apply(gps_sample(), 9).valid)

        degraded = simulator.apply(gps_sample(), 10)
        self.assertTrue(degraded.valid)
        self.assertGreaterEqual(degraded.hdop or 0.0, simulator.degraded_hdop)
        self.assertLessEqual(degraded.satellites or 99, simulator.degraded_satellites)

        lost = simulator.apply(gps_sample(), 11)
        self.assertFalse(lost.valid)
        self.assertEqual(lost.satellites, 0)

    def test_handover_latches_last_good_pose_on_degradation(self) -> None:
        manager = GpsHandoverManager()
        pose = Pose2D(x=2.0, y=3.0, theta=0.25)

        good = manager.update(gps_sample(), gps_xy=(2.0, 3.0), pose_before_update=pose, gps_state="good")
        self.assertEqual(good.mode, "gps_fused")

        degraded = manager.update(
            gps_sample(hdop=8.0, satellites=3),
            gps_xy=(2.5, 3.5),
            pose_before_update=pose,
            gps_state="degraded",
        )
        self.assertEqual(degraded.transition, "good->degraded")
        self.assertEqual(degraded.latched_xy, (2.0, 3.0))

    def test_dead_reckoning_and_motion_selection_are_deterministic(self) -> None:
        reckoner = WheelImuDeadReckoner()
        self.assertFalse(reckoner.update(timestamp=1.0, speed_mps=5.0, gyro_z_rad_s=0.1).valid)

        delta = reckoner.update(timestamp=1.2, speed_mps=5.0, gyro_z_rad_s=0.1)
        self.assertTrue(delta.valid)
        self.assertAlmostEqual(delta.dy, 1.0)
        self.assertAlmostEqual(delta.dtheta, 0.02)

        vo_delta = DeltaPose(dx=0.5, dy=0.8, dtheta=0.3, scale=0.8, matches=40, inliers=20, valid=True)
        fused, source = choose_motion_delta("vo_wheel_imu", vo_delta=vo_delta, wheel_imu_delta=delta)
        self.assertEqual(source, "vo_wheel_imu")
        self.assertAlmostEqual(fused.dy, delta.dy)
        self.assertLessEqual(abs(fused.dx), abs(delta.dy) * 0.25)

    def test_body_delta_integration_and_ground_delta(self) -> None:
        pose = integrate_body_delta(
            Pose2D(x=0.0, y=0.0, theta=math.pi / 2),
            DeltaPose(dx=0.0, dy=2.0, dtheta=0.0, scale=2.0, matches=0, inliers=0, valid=True),
        )

        self.assertAlmostEqual(pose.x, -2.0)
        self.assertAlmostEqual(pose.y, 0.0, places=6)
        self.assertAlmostEqual(ground_delta_from_points((0.0, 0.0), (3.0, 4.0)), 5.0)


if __name__ == "__main__":
    unittest.main()
