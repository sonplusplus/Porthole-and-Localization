import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .landmark_schema import (
    LandmarkMatch,
    LandmarkObservation,
    LandmarkRecord,
    Point3D,
    VisualDescriptor,
    cosine_similarity,
    euclidean_distance,
    normalize_text_ascii,
)


_CLASS_MATCH_DISTANCE: Dict[str, float] = {
    "pothole": 3.0,
    "traffic_sign": 8.0,
    "street_name_sign": 15.0,
}
_DEFAULT_MATCH_DISTANCE = 10.0


class LandmarkDatabase:
    """Persistent Phase 4 landmark DB keyed by stable task-level landmark IDs."""

    def __init__(
        self,
        sequence_id: str,
        match_distance_m: float = _DEFAULT_MATCH_DISTANCE,
        descriptor_threshold: float = 0.82,
        class_match_distance: Optional[Dict[str, float]] = None,
    ) -> None:
        self.sequence_id = sequence_id
        self.match_distance_m = match_distance_m
        self.descriptor_threshold = descriptor_threshold
        self.class_match_distance: Dict[str, float] = dict(_CLASS_MATCH_DISTANCE)
        if class_match_distance:
            self.class_match_distance.update(class_match_distance)
        self.records: Dict[str, LandmarkRecord] = {}
        self._next_by_class: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.records)

    def add(self, record: LandmarkRecord) -> None:
        self.records[record.id] = record
        self._note_existing_id(record.id, record.class_name)

    def upsert(self, obs: LandmarkObservation) -> Tuple[LandmarkRecord, Optional[LandmarkMatch]]:
        match = self.find_best_match(obs)
        if match is None:
            record = LandmarkRecord(
                id=self._new_id(obs.class_name),
                class_name=obs.class_name,
                p_3D=obs.p_3D,
                d_visual=obs.d_visual,
                t_first=obs.timestamp,
                t_last=obs.timestamp,
                n_obs=1,
                source=obs.source,
                attributes=dict(obs.attributes),
            )
            record.attributes["last_bbox_xyxy"] = list(obs.bbox_xyxy)
            self.add(record)
            return record, None

        record = self.records[match.landmark_id]
        record.p_3D = _weighted_point(record.p_3D, obs.p_3D, record.n_obs)
        record.d_visual = _weighted_descriptor(record.d_visual, obs.d_visual, record.n_obs)
        record.t_last = obs.timestamp
        record.n_obs += 1
        record.attributes.update(obs.attributes)
        record.attributes["last_bbox_xyxy"] = list(obs.bbox_xyxy)
        return record, match

    def find_best_match(self, obs: LandmarkObservation) -> Optional[LandmarkMatch]:
        max_dist = self.class_match_distance.get(obs.class_name, self.match_distance_m)
        best: Optional[LandmarkMatch] = None
        for record in self.records.values():
            if record.class_name != obs.class_name:
                continue

            distance = euclidean_distance(record.p_3D, obs.p_3D)
            if distance > max_dist:
                continue

            score, reason = self._association_score(record, obs, distance, max_dist)
            if score <= 0.0:
                continue
            if best is None or score > best.score:
                best = LandmarkMatch(record.id, score, reason)
        return best

    def query(
        self,
        class_name: Optional[str] = None,
        center: Optional[Point3D] = None,
        radius_m: Optional[float] = None,
    ) -> List[LandmarkRecord]:
        out: List[LandmarkRecord] = []
        for record in self.records.values():
            if class_name is not None and record.class_name != class_name:
                continue
            if center is not None and radius_m is not None:
                if euclidean_distance(record.p_3D, center) > radius_m:
                    continue
            out.append(record)
        return out

    def to_jsonl(self, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for record in sorted(self.records.values(), key=lambda item: item.id):
                f.write(json.dumps(record.to_jsonable(), ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, sequence_id: str, path: str) -> "LandmarkDatabase":
        db = cls(sequence_id=sequence_id)
        in_path = Path(path)
        if not in_path.exists():
            return db
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                db.add(LandmarkRecord.from_jsonable(json.loads(line)))
        return db

    def _association_score(
        self,
        record: LandmarkRecord,
        obs: LandmarkObservation,
        distance: float,
        max_dist: float,
    ) -> Tuple[float, str]:
        text_score = _text_score(record.d_visual, obs.d_visual)
        descriptor_score = cosine_similarity(record.d_visual.vector, obs.d_visual.vector)
        distance_score = max(0.0, 1.0 - distance / max(max_dist, 1e-6))

        if obs.class_name == "street_name_sign" and text_score >= 0.99:
            return 0.70 + 0.30 * distance_score, "same_text_nearby"

        if descriptor_score >= self.descriptor_threshold:
            return 0.65 * descriptor_score + 0.35 * distance_score, "descriptor_nearby"

        if text_score > 0.0 and distance <= max_dist * 0.5:
            return 0.55 * text_score + 0.45 * distance_score, "similar_text_nearby"

        return 0.0, "no_match"

    def _new_id(self, class_name: str) -> str:
        safe_sequence = _safe_token(self.sequence_id)
        safe_class = _safe_token(class_name)
        next_index = self._next_by_class.get(safe_class, 1)
        self._next_by_class[safe_class] = next_index + 1
        return f"L_{safe_sequence}_{safe_class}_{next_index:06d}"

    def _note_existing_id(self, landmark_id: str, class_name: str) -> None:
        safe_class = _safe_token(class_name)
        match = re.search(r"_(\d+)$", landmark_id)
        if match is None:
            return
        index = int(match.group(1))
        self._next_by_class[safe_class] = max(self._next_by_class.get(safe_class, 1), index + 1)


def iter_landmarks(path: str) -> Iterable[LandmarkRecord]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield LandmarkRecord.from_jsonable(json.loads(line))


def _safe_token(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def _text_score(a: VisualDescriptor, b: VisualDescriptor) -> float:
    if not a.text_norm or not b.text_norm:
        return 0.0
    if a.text_norm == b.text_norm:
        return 1.0
    a_ascii = normalize_text_ascii(a.text_norm)
    b_ascii = normalize_text_ascii(b.text_norm)
    if a_ascii and b_ascii and a_ascii == b_ascii:
        return 0.95
    return 0.0


def _weighted_point(old: Point3D, new: Point3D, old_count: int) -> Point3D:
    count = max(old_count, 1)
    total = count + 1
    return Point3D(
        x=(old.x * count + new.x) / total,
        y=(old.y * count + new.y) / total,
        z=(old.z * count + new.z) / total,
        quality=max(old.quality, new.quality),
    )


def _weighted_descriptor(
    old: VisualDescriptor,
    new: VisualDescriptor,
    old_count: int,
) -> VisualDescriptor:
    if len(old.vector) != len(new.vector) or not old.vector:
        vector = list(new.vector or old.vector)
    else:
        count = max(old_count, 1)
        total = count + 1
        vector = [(ov * count + nv) / total for ov, nv in zip(old.vector, new.vector)]

    return VisualDescriptor(
        kind=new.kind or old.kind,
        vector=vector,
        quality=max(old.quality, new.quality),
        text_raw=new.text_raw or old.text_raw,
        text_norm=new.text_norm or old.text_norm,
    )
