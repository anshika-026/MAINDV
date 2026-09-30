import math
import time
from dataclasses import dataclass
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MAX_AGE = 12
DEFAULT_MIN_HITS = 1

DEFAULT_IOU_THRESHOLD = 0.20
DEFAULT_DISTANCE_THRESHOLD = 150.0

PERSON_DISTANCE_THRESHOLD = 220.0
OBJECT_DISTANCE_THRESHOLD = 160.0

VELOCITY_SMOOTHING = 0.65
PREDICTION_ENABLED = True

MIN_MATCH_SCORE = 0.08

MAX_PERSON_JUMP = 300.0
MAX_OBJECT_JUMP = 240.0


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _normalise_name(value):
    return str(value or "").strip().lower()


def _bbox_to_center(bbox):
    if not bbox or len(bbox) != 4:
        return 0.0, 0.0

    x1, y1, x2, y2 = map(
        _safe_float,
        bbox,
    )

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def _bbox_width(bbox):
    if not bbox or len(bbox) != 4:
        return 0.0

    return max(
        0.0,
        _safe_float(bbox[2])
        - _safe_float(bbox[0]),
    )


def _bbox_height(bbox):
    if not bbox or len(bbox) != 4:
        return 0.0

    return max(
        0.0,
        _safe_float(bbox[3])
        - _safe_float(bbox[1]),
    )


def _bbox_area(bbox):
    return (
        _bbox_width(bbox)
        * _bbox_height(bbox)
    )


def _intersection_area(
    bbox_a,
    bbox_b,
):
    if (
        not bbox_a
        or not bbox_b
        or len(bbox_a) != 4
        or len(bbox_b) != 4
    ):
        return 0.0

    ax1, ay1, ax2, ay2 = map(
        _safe_float,
        bbox_a,
    )

    bx1, by1, bx2, by2 = map(
        _safe_float,
        bbox_b,
    )

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    return (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )


def calculate_iou(
    bbox_a,
    bbox_b,
):
    intersection = _intersection_area(
        bbox_a,
        bbox_b,
    )

    if intersection <= 0:
        return 0.0

    area_a = _bbox_area(
        bbox_a
    )

    area_b = _bbox_area(
        bbox_b
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def center_distance(
    bbox_a,
    bbox_b,
):
    ax, ay = _bbox_to_center(
        bbox_a
    )

    bx, by = _bbox_to_center(
        bbox_b
    )

    return math.sqrt(
        (ax - bx) ** 2
        + (ay - by) ** 2
    )


def _box_from_detection(
    detection,
):
    if not isinstance(
        detection,
        dict,
    ):
        return [
            0,
            0,
            0,
            0,
        ]

    bbox = detection.get(
        "bbox"
    )

    if (
        isinstance(
            bbox,
            (list, tuple),
        )
        and len(bbox) == 4
    ):
        return list(bbox)

    return [
        detection.get("x1", 0),
        detection.get("y1", 0),
        detection.get("x2", 0),
        detection.get("y2", 0),
    ]


# ============================================================
# TRACK
# ============================================================

@dataclass
class Track:

    track_id: int

    bbox: list

    name: str

    confidence: float = 0.0

    color: str = "unknown"

    status: Any = None

    helmet: str = "unknown"
    helmet_confidence: float = 0.0

    mask: str = "unknown"
    mask_confidence: float = 0.0

    specs: str = "unknown"
    specs_confidence: float = 0.0

    vest: str = "unknown"
    vest_confidence: float = 0.0

    gloves: str = "unknown"
    gloves_confidence: float = 0.0

    age: int = 1

    hits: int = 1

    missed: int = 0

    created_at: float = 0.0

    updated_at: float = 0.0

    velocity_x: float = 0.0
    velocity_y: float = 0.0

    last_center_x: float = 0.0
    last_center_y: float = 0.0

    confirmed: bool = False

    def __post_init__(self):
        if self.created_at <= 0:
            self.created_at = time.time()

        if self.updated_at <= 0:
            self.updated_at = (
                self.created_at
            )

        (
            self.last_center_x,
            self.last_center_y,
        ) = _bbox_to_center(
            self.bbox
        )

        self.confirmed = (
            self.hits
            >= DEFAULT_MIN_HITS
        )


# ============================================================
# OBJECT TRACKER
# ============================================================

class ObjectTracker:

    def __init__(
        self,
        max_age=DEFAULT_MAX_AGE,
        min_hits=DEFAULT_MIN_HITS,
        iou_threshold=DEFAULT_IOU_THRESHOLD,
        distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
    ):
        self.max_age = max(
            1,
            int(max_age),
        )

        self.min_hits = max(
            1,
            int(min_hits),
        )

        self.iou_threshold = max(
            0.0,
            float(iou_threshold),
        )

        self.distance_threshold = max(
            1.0,
            float(distance_threshold),
        )

        self.tracks = {}

        self.next_track_id = 1

        self.frame_count = 0

        self.last_update_time = (
            time.time()
        )

    # ========================================================
    # CREATE TRACK
    # ========================================================

    def _create_track(
        self,
        detection,
    ):
        bbox = _box_from_detection(
            detection
        )

        name = _normalise_name(
            detection.get(
                "name",
                "object",
            )
        )

        confidence = _safe_float(
            detection.get(
                "confidence",
                0.0,
            )
        )

        (
            center_x,
            center_y,
        ) = _bbox_to_center(
            bbox
        )

        now = time.time()

        track = Track(
            track_id=self.next_track_id,

            bbox=bbox,

            name=name,

            confidence=confidence,

            color=detection.get(
                "color",
                "unknown",
            ),

            status=detection.get(
                "status",
                None,
            ),

            helmet=detection.get(
                "helmet",
                "unknown",
            ),

            helmet_confidence=_safe_float(
                detection.get(
                    "helmet_confidence",
                    0.0,
                )
            ),

            mask=detection.get(
                "mask",
                "unknown",
            ),

            mask_confidence=_safe_float(
                detection.get(
                    "mask_confidence",
                    0.0,
                )
            ),

            specs=detection.get(
                "specs",
                "unknown",
            ),

            specs_confidence=_safe_float(
                detection.get(
                    "specs_confidence",
                    0.0,
                )
            ),

            vest=detection.get(
                "vest",
                "unknown",
            ),

            vest_confidence=_safe_float(
                detection.get(
                    "vest_confidence",
                    0.0,
                )
            ),

            gloves=detection.get(
                "gloves",
                "unknown",
            ),

            gloves_confidence=_safe_float(
                detection.get(
                    "gloves_confidence",
                    0.0,
                )
            ),

            age=1,
            hits=1,
            missed=0,

            created_at=now,
            updated_at=now,

            last_center_x=center_x,
            last_center_y=center_y,

            confirmed=(
                1 >= self.min_hits
            ),
        )

        self.tracks[
            self.next_track_id
        ] = track

        self.next_track_id += 1

        return track

    # ========================================================
    # DISTANCE THRESHOLD
    # ========================================================

    def _get_distance_threshold(
        self,
        track,
        detection,
    ):
        name = _normalise_name(
            detection.get(
                "name",
                track.name,
            )
        )

        if name == "person":
            return max(
                self.distance_threshold,
                PERSON_DISTANCE_THRESHOLD,
            )

        return max(
            self.distance_threshold,
            OBJECT_DISTANCE_THRESHOLD,
        )

    # ========================================================
    # PREDICTED BOX
    # ========================================================

    def _predicted_bbox(
        self,
        track,
    ):
        if not PREDICTION_ENABLED:
            return list(
                track.bbox
            )

        dx = track.velocity_x
        dy = track.velocity_y

        return [
            track.bbox[0] + dx,
            track.bbox[1] + dy,
            track.bbox[2] + dx,
            track.bbox[3] + dy,
        ]

    # ========================================================
    # MATCH SCORE
    # ========================================================

    def _calculate_match_score(
        self,
        track,
        detection,
    ):
        detection_name = (
            _normalise_name(
                detection.get(
                    "name",
                    "",
                )
            )
        )

        track_name = (
            _normalise_name(
                track.name
            )
        )

        # Different object classes
        # must never share a track.
        if (
            track_name
            != detection_name
        ):
            return None

        bbox = _box_from_detection(
            detection
        )

        if (
            not bbox
            or _bbox_area(bbox) <= 0
        ):
            return None

        predicted_bbox = (
            self._predicted_bbox(
                track
            )
        )

        iou_current = calculate_iou(
            track.bbox,
            bbox,
        )

        iou_predicted = calculate_iou(
            predicted_bbox,
            bbox,
        )

        iou = max(
            iou_current,
            iou_predicted,
        )

        distance_current = (
            center_distance(
                track.bbox,
                bbox,
            )
        )

        predicted_center = (
            _bbox_to_center(
                predicted_bbox
            )
        )

        detection_center = (
            _bbox_to_center(
                bbox
            )
        )

        distance_predicted = math.sqrt(
            (
                predicted_center[0]
                - detection_center[0]
            ) ** 2
            +
            (
                predicted_center[1]
                - detection_center[1]
            ) ** 2
        )

        distance = min(
            distance_current,
            distance_predicted,
        )

        threshold = (
            self._get_distance_threshold(
                track,
                detection,
            )
        )

        max_jump = (
            MAX_PERSON_JUMP
            if track_name == "person"
            else MAX_OBJECT_JUMP
        )

        # Impossible jump.
        if distance > max_jump:
            return None

        # No spatial relationship.
        if (
            iou < self.iou_threshold
            and distance > threshold
        ):
            return None

        iou_score = min(
            1.0,
            max(
                0.0,
                iou,
            ),
        )

        distance_score = max(
            0.0,
            1.0
            - (
                distance
                / max(
                    threshold,
                    1.0,
                )
            ),
        )

        old_width = _bbox_width(
            track.bbox
        )

        old_height = _bbox_height(
            track.bbox
        )

        new_width = _bbox_width(
            bbox
        )

        new_height = _bbox_height(
            bbox
        )

        width_similarity = (
            min(
                old_width,
                new_width,
            )
            / max(
                old_width,
                new_width,
                1.0,
            )
        )

        height_similarity = (
            min(
                old_height,
                new_height,
            )
            / max(
                old_height,
                new_height,
                1.0,
            )
        )

        size_score = (
            width_similarity
            + height_similarity
        ) / 2.0

        # Person tracking prioritizes spatial overlap.
        if track_name == "person":
            score = (
                iou_score * 0.55
                + distance_score * 0.30
                + size_score * 0.15
            )
        else:
            score = (
                iou_score * 0.50
                + distance_score * 0.35
                + size_score * 0.15
            )

        if score < MIN_MATCH_SCORE:
            return None

        return score

    # ========================================================
    # UPDATE TRACK
    # ========================================================

    def _update_track(
        self,
        track,
        detection,
    ):
        bbox = _box_from_detection(
            detection
        )

        (
            old_center_x,
            old_center_y,
        ) = (
            track.last_center_x,
            track.last_center_y,
        )

        (
            new_center_x,
            new_center_y,
        ) = _bbox_to_center(
            bbox
        )

        raw_velocity_x = (
            new_center_x
            - old_center_x
        )

        raw_velocity_y = (
            new_center_y
            - old_center_y
        )

        track.velocity_x = (
            track.velocity_x
            * (
                1.0
                - VELOCITY_SMOOTHING
            )
            +
            raw_velocity_x
            * VELOCITY_SMOOTHING
        )

        track.velocity_y = (
            track.velocity_y
            * (
                1.0
                - VELOCITY_SMOOTHING
            )
            +
            raw_velocity_y
            * VELOCITY_SMOOTHING
        )

        track.bbox = bbox

        track.confidence = _safe_float(
            detection.get(
                "confidence",
                track.confidence,
            )
        )

        # ----------------------------------------------------
        # Color
        # ----------------------------------------------------

        color = detection.get(
            "color",
            None,
        )

        if (
            color
            and str(color).lower()
            not in (
                "",
                "unknown",
                "none",
            )
        ):
            track.color = color

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        status = detection.get(
            "status",
            None,
        )

        if (
            status is not None
            and str(status).lower()
            not in (
                "",
                "unknown",
                "none",
            )
        ):
            track.status = status

        # ----------------------------------------------------
        # PPE
        # ----------------------------------------------------

        self._update_ppe_field(
            track,
            detection,
            "helmet",
        )

        self._update_ppe_field(
            track,
            detection,
            "mask",
        )

        self._update_ppe_field(
            track,
            detection,
            "specs",
        )

        self._update_ppe_field(
            track,
            detection,
            "vest",
        )

        self._update_ppe_field(
            track,
            detection,
            "gloves",
        )

        track.age += 1

        track.hits += 1

        track.missed = 0

        track.updated_at = time.time()

        track.last_center_x = (
            new_center_x
        )

        track.last_center_y = (
            new_center_y
        )

        track.confirmed = (
            track.hits
            >= self.min_hits
        )

    # ========================================================
    # PPE FIELD UPDATE
    # ========================================================

    def _update_ppe_field(
        self,
        track,
        detection,
        category,
    ):
        value = detection.get(
            category,
            None,
        )

        if value is None:
            return

        value = str(
            value
        ).strip().lower()

        if value in (
            "",
            "none",
            "unknown",
            "unk",
        ):
            return

        confidence = _safe_float(
            detection.get(
                f"{category}_confidence",
                0.0,
            )
        )

        current_value = str(
            getattr(
                track,
                category,
                "unknown",
            )
            or "unknown"
        ).lower()

        current_confidence = (
            _safe_float(
                getattr(
                    track,
                    f"{category}_confidence",
                    0.0,
                )
            )
        )

        # Never replace a strong PPE result
        # with a weaker result.
        if (
            current_value == "unknown"
            or confidence >= current_confidence
        ):
            setattr(
                track,
                category,
                value,
            )

            setattr(
                track,
                f"{category}_confidence",
                confidence,
            )

    # ========================================================
    # MISSED
    # ========================================================

    def _mark_missed(
        self,
        track,
    ):
        track.missed += 1
        track.age += 1

    # ========================================================
    # REMOVE OLD
    # ========================================================

    def _remove_old_tracks(self):
        remove_ids = []

        for (
            track_id,
            track,
        ) in self.tracks.items():

            if (
                track.missed
                > self.max_age
            ):
                remove_ids.append(
                    track_id
                )

        for track_id in remove_ids:
            self.tracks.pop(
                track_id,
                None,
            )

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        detections,
    ):
        self.frame_count += 1

        self.last_update_time = (
            time.time()
        )

        if not isinstance(
            detections,
            list,
        ):
            detections = []

        # ----------------------------------------------------
        # Validate detections
        # ----------------------------------------------------

        valid_detections = []

        for detection in detections:

            if not isinstance(
                detection,
                dict,
            ):
                continue

            bbox = _box_from_detection(
                detection
            )

            if (
                not bbox
                or _bbox_area(bbox) <= 0
            ):
                continue

            name = _normalise_name(
                detection.get(
                    "name",
                    "",
                )
            )

            if not name:
                continue

            valid_detections.append(
                detection
            )

        detections = (
            valid_detections
        )

        # ----------------------------------------------------
        # First frame
        # ----------------------------------------------------

        if not self.tracks:

            for detection in detections:
                self._create_track(
                    detection
                )

            return self.get_results()

        # ----------------------------------------------------
        # No detections
        # ----------------------------------------------------

        if not detections:

            for track in (
                self.tracks.values()
            ):
                self._mark_missed(
                    track
                )

            self._remove_old_tracks()

            return self.get_results()

        unmatched_detections = set(
            range(
                len(detections)
            )
        )

        unmatched_tracks = set(
            self.tracks.keys()
        )

        candidates = []

        # ----------------------------------------------------
        # Generate candidates
        # ----------------------------------------------------

        for (
            track_id,
            track,
        ) in self.tracks.items():

            for (
                detection_index,
                detection,
            ) in enumerate(
                detections
            ):

                score = (
                    self._calculate_match_score(
                        track,
                        detection,
                    )
                )

                if score is None:
                    continue

                candidates.append(
                    (
                        score,
                        track_id,
                        detection_index,
                    )
                )

        # ----------------------------------------------------
        # Highest score first
        # ----------------------------------------------------

        candidates.sort(
            key=lambda item: (
                item[0],
                self.tracks[
                    item[1]
                ].hits,
            ),
            reverse=True,
        )

        matched_pairs = []

        for (
            score,
            track_id,
            detection_index,
        ) in candidates:

            if (
                track_id
                not in unmatched_tracks
            ):
                continue

            if (
                detection_index
                not in unmatched_detections
            ):
                continue

            matched_pairs.append(
                (
                    track_id,
                    detection_index,
                )
            )

            unmatched_tracks.remove(
                track_id
            )

            unmatched_detections.remove(
                detection_index
            )

        # ----------------------------------------------------
        # Update matched tracks
        # ----------------------------------------------------

        for (
            track_id,
            detection_index,
        ) in matched_pairs:

            track = self.tracks.get(
                track_id
            )

            if track is None:
                continue

            self._update_track(
                track,
                detections[
                    detection_index
                ],
            )

        # ----------------------------------------------------
        # Missed old tracks
        # ----------------------------------------------------

        for track_id in (
            unmatched_tracks
        ):

            track = self.tracks.get(
                track_id
            )

            if track is None:
                continue

            self._mark_missed(
                track
            )

        # ----------------------------------------------------
        # New tracks
        # ----------------------------------------------------

        for detection_index in (
            unmatched_detections
        ):

            self._create_track(
                detections[
                    detection_index
                ]
            )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        self._remove_old_tracks()

        return self.get_results()

    # ========================================================
    # RESULTS
    # ========================================================

    def get_results(
        self,
        include_missed=False,
    ):
        results = []

        for track in (
            self.tracks.values()
        ):

            # A single missed frame used to hide a track instantly (any
            # missed > 0) - with a probabilistic model that flickers frame
            # to frame, this is exactly the "detection shows then vanishes
            # right away" bug reported live. Tolerating a couple of misses
            # before dropping it smooths that out without changing when a
            # track is removed entirely (see _remove_old_tracks).
            if (
                not include_missed
                and track.missed > 2
            ):
                continue

            bbox = list(
                track.bbox
            )

            results.append({
                "track_id": track.track_id,

                "name": track.name,

                "bbox": bbox,

                "x1": _safe_int(
                    bbox[0]
                ),

                "y1": _safe_int(
                    bbox[1]
                ),

                "x2": _safe_int(
                    bbox[2]
                ),

                "y2": _safe_int(
                    bbox[3]
                ),

                "confidence": round(
                    track.confidence,
                    3,
                ),

                "color": track.color,

                "status": track.status,

                "helmet": track.helmet,
                "helmet_confidence": round(
                    track.helmet_confidence,
                    3,
                ),

                "mask": track.mask,
                "mask_confidence": round(
                    track.mask_confidence,
                    3,
                ),

                "specs": track.specs,
                "specs_confidence": round(
                    track.specs_confidence,
                    3,
                ),

                "vest": track.vest,
                "vest_confidence": round(
                    track.vest_confidence,
                    3,
                ),

                "gloves": track.gloves,
                "gloves_confidence": round(
                    track.gloves_confidence,
                    3,
                ),

                "age": track.age,

                "hits": track.hits,

                "missed": track.missed,

                "confirmed": track.confirmed,

                "velocity": {
                    "x": round(
                        track.velocity_x,
                        2,
                    ),
                    "y": round(
                        track.velocity_y,
                        2,
                    ),
                },

                "center": {
                    "x": round(
                        track.last_center_x,
                        2,
                    ),
                    "y": round(
                        track.last_center_y,
                        2,
                    ),
                },
            })

        return results

    # ========================================================
    # ACTIVE TRACKS
    # ========================================================

    def get_active_tracks(self):
        return [
            track
            for track
            in self.tracks.values()
            if track.missed == 0
        ]

    # ========================================================
    # GET TRACK
    # ========================================================

    def get_track(
        self,
        track_id,
    ):
        return self.tracks.get(
            _safe_int(
                track_id
            )
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):
        self.tracks.clear()

        self.next_track_id = 1

        self.frame_count = 0

        self.last_update_time = (
            time.time()
        )

    # ========================================================
    # DELETE TRACK
    # ========================================================

    def delete_track(
        self,
        track_id,
    ):
        self.tracks.pop(
            _safe_int(
                track_id
            ),
            None,
        )

    # ========================================================
    # COUNT
    # ========================================================

    def count(
        self,
        name=None,
    ):
        active = [
            track
            for track
            in self.tracks.values()
            if track.missed == 0
        ]

        if name is None:
            return len(active)

        target = _normalise_name(
            name
        )

        return sum(
            1
            for track in active
            if _normalise_name(
                track.name
            ) == target
        )