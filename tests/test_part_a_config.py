import argparse
import math
import unittest
import warnings

from part_a.config import DEFAULT_CAMERA, load_camera_from_args, placeholder_camera


class PartAConfigTests(unittest.TestCase):
    def test_placeholder_camera_uses_shared_defaults(self) -> None:
        cam = placeholder_camera()

        self.assertEqual(cam.width, DEFAULT_CAMERA.width)
        self.assertEqual(cam.height, DEFAULT_CAMERA.height)
        self.assertAlmostEqual(cam.fx, DEFAULT_CAMERA.fx)
        self.assertAlmostEqual(cam.h_camera, DEFAULT_CAMERA.camera_height)
        self.assertAlmostEqual(cam.pitch, math.radians(DEFAULT_CAMERA.pitch_deg))

    def test_load_camera_from_args_warns_without_calibration(self) -> None:
        args = argparse.Namespace(
            calib=None,
            fx=900.0,
            fy=910.0,
            cx=500.0,
            cy=300.0,
            width=1000,
            height=600,
            camera_height=1.35,
            pitch_deg=4.5,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cam = load_camera_from_args(args)

        self.assertTrue(any("without --calib" in str(item.message) for item in caught))
        self.assertAlmostEqual(cam.fx, 900.0)
        self.assertAlmostEqual(cam.fy, 910.0)
        self.assertEqual(cam.width, 1000)
        self.assertAlmostEqual(cam.pitch, math.radians(4.5))


if __name__ == "__main__":
    unittest.main()
