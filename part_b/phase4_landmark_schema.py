import math
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Point3D:
    x: float
    y: float
    z: float = 0.0
    quality: float = 0.0


@dataclass
class VisualDescriptor:
    kind: str
    vector: List[float] = field(default_factory=list)
    quality: float = 0.0
    text_raw: Optional[str] = None
    text_norm: Optional[str] = None


@dataclass
class LandmarkRecord:
    id: str
    class_name: str
    p_3D: Point3D
    d_visual: VisualDescriptor
    t_first: float
    t_last: float
    n_obs: int
    source: str = "phase4"
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        row = asdict(self)
        row["class"] = row.pop("class_name")
        return row

    @classmethod
    def from_jsonable(cls, row: Dict[str, Any]) -> "LandmarkRecord":
        data = dict(row)
        data["class_name"] = data.pop("class", data.get("class_name"))
        data["p_3D"] = Point3D(**data["p_3D"])
        data["d_visual"] = VisualDescriptor(**data["d_visual"])
        return cls(**data)


@dataclass
class LandmarkObservation:
    observation_id: str
    class_name: str
    p_3D: Point3D
    d_visual: VisualDescriptor
    timestamp: float
    frame_index: int
    bbox_xyxy: Tuple[int, int, int, int]
    source: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LandmarkMatch:
    landmark_id: str
    score: float
    reason: str


def normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = unicodedata.normalize("NFC", value)
    text = " ".join(text.strip().lower().split())
    return text or None


def normalize_text_ascii(value: Optional[str]) -> Optional[str]:
    text = normalize_text(value)
    if text is None:
        return None
    without_marks = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in without_marks if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", without_marks)


def euclidean_distance(a: Point3D, b: Point3D) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for av, bv in zip(a, b):
        dot += av * bv
        na += av * av
        nb += bv * bv
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom <= 1e-12:
        return 0.0
    return float(dot / denom)

