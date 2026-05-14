import math
import warnings
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from part_a.calibration import CameraParams
from .schema import GpsSample, ImuSample, Phase3Frame


KITTI_DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
_KITTI_FALLBACK_PARAMS = {
    "width": 1242,
    "height": 375,
    "fx": 721.5377,
    "fy": 721.5377,
    "cx": 609.5593,
    "cy": 172.854,
}


def _kitti_fallback_with_warning(reason: str) -> CameraParams:
    warnings.warn(
        f"Using KITTI default intrinsics ({reason}). Results will be wrong for non-KITTI cameras. "
        "Provide a calib zip or calib_cam_to_cam.txt.",
        UserWarning,
        stacklevel=3,
    )
    return CameraParams(**_KITTI_FALLBACK_PARAMS)


@dataclass
class KittiSequenceRef:
    sequence_id: str
    sync_path: Path
    calib_path: Optional[Path]


@dataclass
class KittiRawSample:
    meta: Phase3Frame
    frame: np.ndarray
    local_xy: Optional[Tuple[float, float]]


def discover_kitti_sequences(data_root: str = "data") -> List[KittiSequenceRef]:
    root = Path(data_root)
    refs: List[KittiSequenceRef] = []
    for sync_zip in sorted(root.glob("*_sync.zip")):
        sequence_id = sync_zip.name.replace("_sync.zip", "")
        calib_zip = root / f"{sequence_id}_calib.zip"
        refs.append(
            KittiSequenceRef(
                sequence_id=sequence_id,
                sync_path=sync_zip,
                calib_path=calib_zip if calib_zip.exists() else None,
            )
        )

    for sync_dir in sorted(root.rglob("*_sync")):
        if not sync_dir.is_dir():
            continue
        sequence_id = sync_dir.name.replace("_sync", "")
        date_dir = sync_dir.parent
        calib_file = date_dir / "calib_cam_to_cam.txt"
        refs.append(
            KittiSequenceRef(
                sequence_id=sequence_id,
                sync_path=sync_dir,
                calib_path=calib_file if calib_file.exists() else None,
            )
        )
    return refs


class KittiRawSequence:
    """Read KITTI Raw sync data from either a zip file or an extracted directory."""

    def __init__(
        self,
        sync_path: str,
        calib_path: Optional[str] = None,
        camera: str = "image_02",
    ) -> None:
        self.sync_path = Path(sync_path)
        self.calib_path = Path(calib_path) if calib_path else None
        self.camera = camera
        self.sequence_id = self.sync_path.name.replace("_sync.zip", "").replace("_sync", "")

        self._zip: Optional[zipfile.ZipFile] = None
        self._zip_names: List[str] = []
        self._sync_prefix = ""
        if self.sync_path.suffix.lower() == ".zip":
            self._zip = zipfile.ZipFile(self.sync_path)
            self._zip_names = self._zip.namelist()
            self._sync_prefix = self._find_zip_sync_prefix()

        self.frame_files = self._list_frame_files()
        self.frame_timestamps = self._read_timestamps(f"{self.camera}/timestamps.txt")
        self.oxts_timestamps = self._read_timestamps("oxts/timestamps.txt")
        self.oxts_files = self._list_oxts_files()
        self.camera_params = self._load_camera_params()

    def __len__(self) -> int:
        return min(len(self.frame_files), len(self.frame_timestamps), len(self.oxts_files))

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def iter_samples(self, max_frames: Optional[int] = None) -> Iterator[KittiRawSample]:
        origin: Optional[Tuple[float, float]] = None
        total = len(self)
        if max_frames is not None:
            total = min(total, max_frames)

        for idx in range(total):
            timestamp = self.frame_timestamps[idx]
            oxts = self._read_oxts(self.oxts_files[idx])
            gps, imu = _parse_oxts(oxts)
            xy = gps_to_local_xy(gps.lat, gps.lon, origin)
            if origin is None:
                origin = xy
                xy = (0.0, 0.0)

            frame_name = self.frame_files[idx]
            frame = self._read_image(frame_name)
            meta = Phase3Frame(
                sequence_id=self.sequence_id,
                frame_index=idx,
                timestamp=(timestamp - self.frame_timestamps[0]).total_seconds(),
                timestamp_iso=timestamp.isoformat(),
                frame_path=self._display_path(frame_name),
                gps=gps,
                imu=imu,
            )
            yield KittiRawSample(meta=meta, frame=frame, local_xy=xy)

    def _find_zip_sync_prefix(self) -> str:
        for name in self._zip_names:
            if name.endswith(f"{self.camera}/timestamps.txt"):
                return name[: -len(f"{self.camera}/timestamps.txt")]
        raise FileNotFoundError(f"Could not find {self.camera}/timestamps.txt in {self.sync_path}")

    def _list_frame_files(self) -> List[str]:
        suffix = f"{self.camera}/data/"
        if self._zip is not None:
            files = [
                name
                for name in self._zip_names
                if name.startswith(self._sync_prefix + suffix) and name.lower().endswith(".png")
            ]
            return sorted(files)
        frame_dir = self.sync_path / self.camera / "data"
        return sorted(str(path) for path in frame_dir.glob("*.png"))

    def _list_oxts_files(self) -> List[str]:
        suffix = "oxts/data/"
        if self._zip is not None:
            files = [
                name
                for name in self._zip_names
                if name.startswith(self._sync_prefix + suffix) and name.lower().endswith(".txt")
            ]
            return sorted(files)
        oxts_dir = self.sync_path / "oxts" / "data"
        return sorted(str(path) for path in oxts_dir.glob("*.txt"))

    def _read_timestamps(self, rel_path: str) -> List[datetime]:
        text = self._read_text(rel_path)
        stamps = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            stamps.append(_parse_kitti_timestamp(line))
        return stamps

    def _read_oxts(self, path: str) -> List[float]:
        if self._zip is not None:
            text = self._zip.read(path).decode("utf-8")
        else:
            text = Path(path).read_text(encoding="utf-8")
        return [float(item) for item in text.strip().split()]

    def _read_text(self, rel_path: str) -> str:
        if self._zip is not None:
            return self._zip.read(self._sync_prefix + rel_path).decode("utf-8")
        return (self.sync_path / rel_path).read_text(encoding="utf-8")

    def _read_image(self, path: str) -> np.ndarray:
        if self._zip is not None:
            raw = np.frombuffer(self._zip.read(path), dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        else:
            frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"Could not read KITTI frame: {path}")
        return frame

    def _display_path(self, path: str) -> str:
        if self._zip is not None:
            return f"{self.sync_path}!{path}"
        return str(Path(path).resolve())

    def _load_camera_params(self) -> CameraParams:
        if self.calib_path is None or not self.calib_path.exists():
            return _kitti_fallback_with_warning("calib path not provided or not found")

        if self.calib_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(self.calib_path) as zf:
                name = next((n for n in zf.namelist() if n.endswith("calib_cam_to_cam.txt")), None)
                if name is None:
                    return _kitti_fallback_with_warning("calib_cam_to_cam.txt missing in calib zip")
                text = zf.read(name).decode("utf-8")
        elif self.calib_path.is_file():
            text = self.calib_path.read_text(encoding="utf-8")
        else:
            calib_file = self.calib_path / "calib_cam_to_cam.txt"
            if not calib_file.exists():
                return _kitti_fallback_with_warning("calib_cam_to_cam.txt missing")
            text = calib_file.read_text(encoding="utf-8")

        data = _parse_calib_text(text)
        key = "P_rect_02" if self.camera == "image_02" else "P_rect_03"
        values = data.get(key)
        if values is None or len(values) < 12:
            return _kitti_fallback_with_warning(f"{key} missing or too short")
        p = np.array(values, dtype=np.float64).reshape(3, 4)
        width, height = _parse_size(data.get(f"S_rect_{self.camera[-2:]}", []))
        return CameraParams(
            fx=float(p[0, 0]),
            fy=float(p[1, 1]),
            cx=float(p[0, 2]),
            cy=float(p[1, 2]),
            width=width,
            height=height,
        )


def _parse_calib_text(text: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        vals = []
        for item in raw.strip().split():
            try:
                vals.append(float(item))
            except ValueError:
                pass
        out[key.strip()] = vals
    return out


def _parse_kitti_timestamp(value: str) -> datetime:
    # KITTI timestamps often store nanoseconds. Python datetime keeps
    # microseconds, so trim the fractional component to six digits.
    if "." not in value:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    head, frac = value.split(".", 1)
    frac = (frac + "000000")[:6]
    return datetime.strptime(f"{head}.{frac}", KITTI_DATE_FORMAT)


def _parse_size(values: List[float]) -> Tuple[int, int]:
    if len(values) >= 2:
        return int(values[0]), int(values[1])
    return 1242, 375


def _parse_oxts(values: List[float]) -> Tuple[GpsSample, ImuSample]:
    lat, lon, alt = values[0], values[1], values[2]
    vn, ve = values[6], values[7]
    speed = float((vn * vn + ve * ve) ** 0.5)
    ax, ay, az = values[11], values[12], values[13]
    gx, gy, gz = values[17], values[18], values[19]
    pos_accuracy = values[23] if len(values) > 23 else None
    satellites = int(values[25]) if len(values) > 25 else None
    gps = GpsSample(
        lat=lat,
        lon=lon,
        alt=alt,
        speed_mps=speed,
        hdop=pos_accuracy,
        satellites=satellites,
        valid=math.isfinite(lat) and math.isfinite(lon) and abs(lat) > 1e-9 and abs(lon) > 1e-9,
    )
    imu = ImuSample(ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz)
    return gps, imu


def gps_to_local_xy(
    lat: float,
    lon: float,
    origin: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    radius = 6378137.0
    lat_rad = math.radians(lat)
    x = radius * math.radians(lon) * math.cos(lat_rad)
    y = radius * math.radians(lat)
    if origin is None:
        return x, y
    return x - origin[0], y - origin[1]
