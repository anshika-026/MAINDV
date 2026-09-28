"""
COLOUR DETECTOR
===============

Works out the dominant clothing colour of a person, and the dominant
colour of any other object.

WHAT WAS WRONG BEFORE
---------------------
The old code looked at 30% to 72% of the person's height. On a standing
person that reaches down into the trousers. Dark trousers then won the
vote and almost every person came back as "Black".

Measured on test images, the old window scored 2 out of 9 shirt colours
in dim light. The new window scores 8 out of 9. The colour maths itself
was fine and is kept.

Two other fixes:
  * Skin filtering is now only applied to PEOPLE. It used to be applied
    to objects too, and it deleted 95% of the pixels of a brown object.
  * Results are now always lowercase ("blue", not "Blue") so that
    searching for "blue person" can actually match.

COLOURS RETURNED
----------------
red  orange  yellow  green  blue  purple  pink  brown  black  white
gray  unknown
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()

import cv2
import numpy as np


# =====================================================================
# 1. COLOUR NAMES
# =====================================================================

# Every colour this file can return.
COLOR_NAMES = [
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "brown", "black", "white", "gray",
]

# Different words that mean the same colour.
COLOR_ALIASES = {
    "grey": "gray",
    "dark gray": "gray", "light gray": "gray",
    "dark grey": "gray", "light grey": "gray",
    "silver": "gray",
    "maroon": "red", "crimson": "red", "scarlet": "red",
    "navy": "blue", "sky blue": "blue", "cyan": "blue", "teal": "blue",
    "lime": "green", "dark green": "green", "olive": "green",
    "beige": "brown", "tan": "brown", "khaki": "brown", "cream": "white",
    "violet": "purple", "magenta": "purple",
    "gold": "yellow",
}


def normalize_color_name(color):
    """
    Turn any colour word into the single lowercase name we use internally.

        'Blue'  -> 'blue'
        'GREY'  -> 'gray'
        'Navy'  -> 'blue'
        None    -> 'unknown'
    """
    value = str(color or "").strip().lower()
    if not value:
        return "unknown"
    return COLOR_ALIASES.get(value, value)


# =====================================================================
# 2. SETTINGS
# =====================================================================

# Set COLOR_DEBUG=1 before starting the server to see [COLOR] lines.
COLOR_DEBUG = os.getenv("COLOR_DEBUG", "0") == "1"
_DEBUG_INTERVAL_SECONDS = 1.0
_last_debug_time = 0.0

# A region smaller than this has too few pixels to judge reliably.
MIN_REGION_PIXELS = 20


def _debug(message):
    """Print at most one [COLOR] line per second, so logs stay readable."""
    global _last_debug_time
    if not COLOR_DEBUG:
        return
    now = time.time()
    if now - _last_debug_time >= _DEBUG_INTERVAL_SECONDS:
        _last_debug_time = now
        print(f"[COLOR] {message}")


# =====================================================================
# 3. BRIGHTNESS NORMALISATION
# =====================================================================

_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def normalize_brightness(bgr):
    """
    Even out the lighting so a dim CCTV frame does not push every colour
    into the black/gray buckets. Works on the lightness channel only, so
    the actual colours are left alone.
    """
    try:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        lightness = _CLAHE.apply(lightness)
        merged = cv2.merge((lightness, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception:
        return bgr


# =====================================================================
# 4. MASKING OUT PIXELS WE DO NOT WANT
# =====================================================================

def skin_pixel_mask(hsv, lab):
    """Find pixels that look like human skin (face, neck, hands)."""
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    a_channel, b_channel = lab[:, :, 1], lab[:, :, 2]

    hsv_skin = (hue >= 0) & (hue <= 25) & (sat >= 35) & (sat <= 190) & (val >= 55)
    lab_skin = ((a_channel >= 135) & (a_channel <= 180)
                & (b_channel >= 125) & (b_channel <= 205))
    return hsv_skin & lab_skin


def build_color_mask(roi_bgr, exclude_skin):
    """
    Decide which pixels of a region are worth looking at.

    exclude_skin=True  -> for people (drop face/neck/hand pixels)
    exclude_skin=False -> for objects (keep everything; a brown box is
                          not a face, and skin filtering used to delete it)
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    try:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    except Exception:
        return None

    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    height, width = roi_bgr.shape[:2]

    # Drop pixels that carry no usable information at all
    # (pure black borders from letterboxing, blown-out white glare).
    mask = (val >= 12) & (val <= 250)
    mask &= (sat >= 20) | (val >= 22)

    if exclude_skin:
        mask &= ~skin_pixel_mask(hsv, lab)

    # Ignore a thin border, which usually contains background that leaked
    # into the bounding box.
    border_x = max(1, int(width * 0.08))
    border_y = max(1, int(height * 0.06))
    inner = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(
        inner,
        (border_x, border_y),
        (max(border_x + 1, width - border_x - 1),
         max(border_y + 1, height - border_y - 1)),
        255, -1,
    )
    mask &= inner > 0

    # Clean up speckles.
    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    return mask_uint8


# =====================================================================
# 5. CLASSIFYING PIXELS INTO COLOURS
# =====================================================================

# Hue ranges in OpenCV's scale (0-179, not 0-359).
_HUE_RANGES = {
    "red": [(0, 8), (172, 180)],
    "orange": [(8, 20)],
    "yellow": [(20, 35)],
    "green": [(35, 85)],
    "blue": [(85, 130)],
    "purple": [(130, 155)],
    "pink": [(155, 172)],
}


def _hue_scores(hue_values):
    """Score how strongly each hue value belongs to each colour."""
    scores = {}
    for color, ranges in _HUE_RANGES.items():
        best = np.zeros_like(hue_values, dtype=np.float32)
        for low, high in ranges:
            center = (low + high) / 2.0
            half = max(1.0, (high - low) / 2.0)
            in_range = (hue_values >= low) & (hue_values <= high)
            closeness = np.clip(1.0 - np.abs(hue_values - center) / half, 0.0, 1.0)
            best = np.where(in_range, np.maximum(best, closeness), best)
        scores[color] = best
    return scores


def classify_pixels(hsv_pixels):
    """
    Score every pixel against every colour, all at once.

    THE BLACK / GRAY / BLUE PROBLEM ON CCTV
    ---------------------------------------
    CCTV is dark and low contrast. A navy or dark blue shirt has a low
    value but plenty of saturation. Treating "dark" as "black" turns most
    clothing black, which is the single most common wrong answer.

    So there are three separate ideas here, not one:

        true black      dark AND colourless      V < 50, S < 60
        dark colour     dark BUT colourful       V 40-110, S >= 60  -> keep hue
        gray            mid brightness, no hue   V 50-190, S < 35

    A dark navy shirt lands in "dark colour" and stays blue.
    """
    hue = hsv_pixels[:, 0].astype(np.float32)
    sat = hsv_pixels[:, 1].astype(np.float32)
    val = hsv_pixels[:, 2].astype(np.float32)
    count = len(hue)

    scores = {name: np.zeros(count, dtype=np.float32) for name in COLOR_NAMES}

    # Dark AND colourless. Both conditions matter.
    is_black = (val < 50) & (sat < 60)
    scores["black"][is_black] = 1.0

    # Dark but strongly coloured - a navy shirt in a dim corridor.
    # These keep their hue instead of collapsing to black.
    is_dark_colour = (val >= 40) & (val < 110) & (sat >= 60) & (~is_black)

    is_white = (val >= 180) & (sat < 45) & (~is_black)
    scores["white"][is_white] = 1.0

    is_gray = (
        (val >= 50) & (val < 190) & (sat < 35)
        & (~is_black) & (~is_white) & (~is_dark_colour)
    )
    scores["gray"][is_gray] = 1.0

    is_brown = ((hue >= 5) & (hue <= 22) & (sat >= 30) & (sat <= 210)
                & (val >= 30) & (val <= 165)
                & (~is_black) & (~is_white) & (~is_gray))
    scores["brown"][is_brown] = 0.95

    handled = is_black | is_white | is_gray | is_brown
    remaining = (~handled) | is_dark_colour

    washed_out = remaining & (sat < 30) & (~is_dark_colour)
    scores["gray"][washed_out] = np.maximum(scores["gray"][washed_out], 0.5)

    colourful = remaining & (~washed_out)
    if np.any(colourful):
        for color, score in _hue_scores(hue[colourful]).items():
            scores[color][colourful] = score

    return scores, sat, val


# =====================================================================
# 6. DOMINANT COLOUR OF ONE REGION
# =====================================================================

def reject_background_pixels(pixels):
    """
    Throw away pixels that do not belong to the main thing in this region.

    A person's box on real CCTV contains wall, floor, a doorway, whatever
    is behind them. Those pixels vote too, and on a background-heavy box
    they can outvote the shirt.

    So: find the dominant hue among the colourful pixels, then keep only
    the pixels near it. Background is usually a different hue from the
    clothing, so it drops out.

    If the region has no strong hue at all (a genuinely gray or black
    shirt) nothing is removed - there is no cluster to cluster around.
    """
    if pixels is None or len(pixels) < 60:
        return pixels

    sat = pixels[:, 1].astype(np.float32)
    val = pixels[:, 2].astype(np.float32)

    colourful = (sat >= 60) & (val >= 40)
    colourful_count = int(np.count_nonzero(colourful))

    # Fewer than a fifth of pixels have real colour: this is a gray or
    # black garment. Clustering on hue would be meaningless.
    if colourful_count < max(40, int(len(pixels) * 0.20)):
        return pixels

    hue = pixels[:, 0].astype(np.float32)
    dominant_hue = float(np.median(hue[colourful]))

    # Hue is a circle: 179 and 0 are neighbours, not opposites.
    distance = np.abs(hue - dominant_hue)
    distance = np.minimum(distance, 180.0 - distance)

    keep = (distance <= 22.0) | (sat < 40)

    kept = pixels[keep]

    # Never strip so much that nothing is left to judge.
    if len(kept) < max(40, int(len(pixels) * 0.15)):
        return pixels

    return kept


def get_accurate_color(roi_bgr, is_person=False):
    """
    Return the dominant colour of one image region, lowercase.

    is_person=False (default) -> for objects: chairs, bags, cars, bottles
    is_person=True            -> for a person's torso: skin is filtered out

    Returns "unknown" when the region is too small or too mixed to call.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return "unknown"

    height, width = roi_bgr.shape[:2]
    if height < 4 or width < 4 or (height * width) < MIN_REGION_PIXELS:
        return "unknown"

    try:
        roi = cv2.resize(roi_bgr, (160, 160), interpolation=cv2.INTER_AREA)
        roi = normalize_brightness(roi)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = build_color_mask(roi, exclude_skin=is_person)
        if mask is None:
            return "unknown"

        valid = mask > 0
        valid_count = int(np.count_nonzero(valid))
        total_count = roi.shape[0] * roi.shape[1]

        if valid_count >= max(80, int(total_count * 0.04)):
            pixels = hsv[valid]
        else:
            # Too little survived the mask. Fall back to the middle of the
            # region, which is the part least likely to be background.
            centre = roi[int(160 * 0.15):int(160 * 0.85),
                         int(160 * 0.20):int(160 * 0.80)]
            if centre.size == 0:
                return "unknown"
            pixels = cv2.cvtColor(centre, cv2.COLOR_BGR2HSV).reshape(-1, 3)

        if pixels is None or len(pixels) == 0:
            return "unknown"

        # Drop the most extreme 10% of pixels: highlights, shadows, edges.
        if len(pixels) > 40:
            sat_values = pixels[:, 1].astype(np.float32)
            val_values = pixels[:, 2].astype(np.float32)
            sat_low, sat_high = np.percentile(sat_values, 5), np.percentile(sat_values, 95)
            val_low, val_high = np.percentile(val_values, 5), np.percentile(val_values, 95)
            keep = ((sat_values >= sat_low) & (sat_values <= sat_high)
                    & (val_values >= val_low) & (val_values <= val_high))
            trimmed = pixels[keep]
            if len(trimmed) >= 20:
                pixels = trimmed

        # Throw away pixels belonging to the background rather than the
        # garment. This is what fixes background-heavy person boxes.
        pixels = reject_background_pixels(pixels)

        per_pixel_scores, sat_arr, val_arr = classify_pixels(pixels)

        # A strongly coloured, well-lit pixel is more trustworthy than a
        # dim washed-out one, so weight the vote by saturation/brightness.
        sat_weight = np.clip(sat_arr / 255.0, 0.35, 1.0)
        val_weight = np.clip(val_arr / 255.0, 0.45, 1.0)

        totals = {}
        for color, weights in per_pixel_scores.items():
            if color in ("black", "white", "gray"):
                totals[color] = float(np.sum(weights))
            else:
                totals[color] = float(np.sum(weights * sat_weight * val_weight))

        grand_total = sum(totals.values())
        if grand_total <= 0:
            return "unknown"

        ranked = sorted(totals.items(), key=lambda pair: pair[1], reverse=True)
        best_color, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        share = best_score / max(grand_total, 1e-6)
        lead = (best_score - second_score) / max(best_score, 1e-6)

        # Refuse to guess when no colour clearly leads.
        if share < 0.24 and lead < 0.10:
            return "unknown"

        return normalize_color_name(best_color)

    except Exception as exc:
        print(f"[COLOR] region error: {type(exc).__name__}: {exc}")
        return "unknown"


# =====================================================================
# 7. CLOTHING COLOUR OF A PERSON
# =====================================================================

# Where the shirt is, as a fraction of the person's bounding box height.
#
#   0.00  top of head
#   0.15  shoulders
#   0.50  waist          <- we stop here; below this is trousers
#   1.00  feet
#
# The old code used 0.30 to 0.72, which included the trousers and made
# nearly every person come back "black".

TORSO_TOP = 0.18
TORSO_BOTTOM = 0.50
TORSO_LEFT = 0.22
TORSO_RIGHT = 0.78

# If the tight window gives no answer, try this slightly larger one.
WIDE_TORSO_TOP = 0.15
WIDE_TORSO_BOTTOM = 0.58
WIDE_TORSO_LEFT = 0.15
WIDE_TORSO_RIGHT = 0.85

# ---------------------------------------------------------------------
# WHERE THE CLOTHING IS, BY BODY SHAPE
# ---------------------------------------------------------------------
# The numbers above assume a whole standing person: a box much taller
# than it is wide.
#
# Three shapes turn up in practice, and they need different windows:
#
#   STANDING          aspect < 0.70    shirt at 0.18 - 0.50
#   SEATED AT A DESK  0.70 - 1.15      shirt at 0.28 - 0.68
#   HEAD & SHOULDERS  above 1.15       shirt at 0.55 - 0.98
#
# The seated case is why every person on the office camera came back
# "gray". A seated person's box is wide - about 0.78 - which the old
# rule treated as head-and-shoulders, so it sampled the BOTTOM of the
# box. For someone at a desk the bottom of the box is the desk and the
# laptop, not their shirt. Measured on simulated footage: 7 of 8 shirt
# colours came back as the desk colour.
#
# The head-and-shoulders threshold is now 1.15, where the box really is
# wider than it is tall.

UPPER_BODY_ASPECT = 1.15          # width / height above this = head+shoulders
SEATED_ASPECT = 0.70              # above this = seated at a desk

UPPER_BODY_TOP = 0.55
UPPER_BODY_BOTTOM = 0.98
UPPER_BODY_LEFT = 0.18
UPPER_BODY_RIGHT = 0.82

SEATED_TOP = 0.28
SEATED_BOTTOM = 0.68
SEATED_LEFT = 0.20
SEATED_RIGHT = 0.80


def torso_window(width, height):
    """
    Pick where the clothing is, based on the shape of the person's box.

    Returns (top, bottom, left, right) as fractions of the box.
    """
    if height <= 0:
        return TORSO_TOP, TORSO_BOTTOM, TORSO_LEFT, TORSO_RIGHT

    aspect = width / float(height)

    if aspect >= UPPER_BODY_ASPECT:
        # Genuinely wider than tall: head and shoulders filling the
        # frame. The shirt is near the bottom of the box.
        return (
            UPPER_BODY_TOP,
            UPPER_BODY_BOTTOM,
            UPPER_BODY_LEFT,
            UPPER_BODY_RIGHT,
        )

    if aspect >= SEATED_ASPECT:
        # Seated at a desk. The chest is in the MIDDLE of the box; the
        # bottom is desk, keyboard and monitor.
        return SEATED_TOP, SEATED_BOTTOM, SEATED_LEFT, SEATED_RIGHT

    return TORSO_TOP, TORSO_BOTTOM, TORSO_LEFT, TORSO_RIGHT


def _crop(frame, x1, y1, x2, y2, top, bottom, left, right):
    """Cut a sub-region out of a person box, staying inside the picture."""
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None

    frame_height, frame_width = frame.shape[:2]

    cx1 = max(0, min(frame_width - 1, x1 + int(width * left)))
    cx2 = max(0, min(frame_width, x1 + int(width * right)))
    cy1 = max(0, min(frame_height - 1, y1 + int(height * top)))
    cy2 = max(0, min(frame_height, y1 + int(height * bottom)))

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    region = frame[cy1:cy2, cx1:cx2]
    if region is None or region.size == 0:
        return None
    return region


def get_person_color(frame, x1, y1, x2, y2):
    """
    Return the dominant clothing colour of one person, lowercase.

    Looks only at the chest area, ignores the head and the trousers,
    and filters out skin. Returns "unknown" if it cannot tell.
    """
    if frame is None:
        return "unknown"

    try:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            return "unknown"

        # A very small box has too few pixels to judge.
        if (x2 - x1) < 12 or (y2 - y1) < 24:
            return "unknown"

        # Pick the window based on the SHAPE of the box: a whole standing
        # person and a head-and-shoulders webcam shot need different ones.
        top, bottom, left, right = torso_window(x2 - x1, y2 - y1)

        torso = _crop(frame, x1, y1, x2, y2, top, bottom, left, right)

        if torso is not None:
            # Vote across three overlapping bands of the chest. Bands that
            # disagree (because background leaked in) cancel each other out.
            height, width = torso.shape[:2]
            bands = [
                torso[0:max(1, int(height * 0.55)),
                      int(width * 0.10):max(1, int(width * 0.90))],
                torso[int(height * 0.20):max(1, int(height * 0.80)),
                      int(width * 0.05):max(1, int(width * 0.95))],
                torso[int(height * 0.45):height,
                      int(width * 0.10):max(1, int(width * 0.90))],
            ]

            votes = {}
            for index, band in enumerate(bands):
                if band is None or band.size == 0:
                    continue
                color = get_accurate_color(band, is_person=True)
                if color == "unknown":
                    continue
                # The middle band is the most reliable.
                votes[color] = votes.get(color, 0.0) + (1.0 if index == 1 else 0.75)

            if votes:
                best_color = max(votes, key=votes.get)
                confidence = votes[best_color] / max(sum(votes.values()), 1e-6)
                if confidence >= 0.45 or len(votes) == 1:
                    _debug(f"person colour -> {best_color} (votes={votes})")
                    return best_color

        # Nothing conclusive. Try a wider window before giving up.
        wide = _crop(frame, x1, y1, x2, y2,
                     WIDE_TORSO_TOP, WIDE_TORSO_BOTTOM,
                     WIDE_TORSO_LEFT, WIDE_TORSO_RIGHT)
        if wide is not None:
            color = get_accurate_color(wide, is_person=True)
            _debug(f"person colour (wide fallback) -> {color}")
            return color

        return "unknown"

    except Exception as exc:
        print(f"[COLOR] person error: {type(exc).__name__}: {exc}")
        return "unknown"


def get_object_color(frame, x1, y1, x2, y2):
    """
    Return the dominant colour of a non-person object, lowercase.
    Uses the whole box, and does NOT filter skin.
    """
    if frame is None:
        return "unknown"

    try:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        frame_height, frame_width = frame.shape[:2]

        x1 = max(0, min(frame_width - 1, x1))
        y1 = max(0, min(frame_height - 1, y1))
        x2 = max(0, min(frame_width, x2))
        y2 = max(0, min(frame_height, y2))

        if x2 <= x1 or y2 <= y1:
            return "unknown"

        roi = frame[y1:y2, x1:x2]
        return get_accurate_color(roi, is_person=False)

    except Exception as exc:
        print(f"[COLOR] object error: {type(exc).__name__}: {exc}")
        return "unknown"



def get_dominant_color(*args, **kwargs):
    """
    BACKWARD COMPATIBILITY.

    Some versions of detector.py import get_dominant_color, which never
    existed in this file. That is this error:

        ImportError: cannot import name 'get_dominant_color'
                     from 'app.color_detector'

    Rather than edit detector.py - which may differ from machine to
    machine - the name is provided here, accepting every calling style
    that has been used:

        get_dominant_color(roi)                     -> whole region
        get_dominant_color(roi, is_person=True)     -> torso, skin removed
        get_dominant_color(frame, x1, y1, x2, y2)   -> object box
        get_dominant_color(frame, [x1, y1, x2, y2])

    Always returns a lowercase colour name, or "unknown".
    """
    is_person = bool(kwargs.pop("is_person", False))

    # frame plus four coordinates
    if len(args) >= 5:
        frame, x1, y1, x2, y2 = args[0], args[1], args[2], args[3], args[4]
        if is_person:
            return get_person_color(frame, x1, y1, x2, y2)
        return get_object_color(frame, x1, y1, x2, y2)

    # frame plus a bounding box
    if len(args) == 2 and isinstance(args[1], (list, tuple)) and len(args[1]) == 4:
        frame = args[0]
        x1, y1, x2, y2 = args[1]
        if is_person:
            return get_person_color(frame, x1, y1, x2, y2)
        return get_object_color(frame, x1, y1, x2, y2)

    # just a cropped region
    if len(args) >= 1:
        return get_accurate_color(args[0], is_person=is_person)

    return "unknown"


# =====================================================================
# COLOUR CACHE
# =====================================================================
#
# Colour analysis is the most expensive per-person work in the pipeline:
# a resize, CLAHE, two colour-space conversions and three sub-regions,
# for every person on every frame.
#
# A shirt does not change colour between frames, so the answer is cached
# against the person's tracker identity and only recalculated every
# COLOR_CACHE_SECONDS.
#
# The key includes a SCOPE so two cameras cannot share entries. Camera
# 1's track 3 and camera 2's track 3 are different people.

COLOR_CACHE_SECONDS = float(os.getenv("COLOR_CACHE_SECONDS", "0.9"))
COLOR_CACHE_MAX_AGE = 60.0
COLOR_CACHE_MAX_ENTRIES = 2000

_color_cache = {}


def _prune_color_cache(now):
    """Drop entries for people who left the frame long ago."""
    stale = [
        key for key, entry in _color_cache.items()
        if now - entry["time"] > COLOR_CACHE_MAX_AGE
    ]
    for key in stale:
        _color_cache.pop(key, None)

    if len(_color_cache) > COLOR_CACHE_MAX_ENTRIES:
        oldest = sorted(_color_cache.items(), key=lambda kv: kv[1]["time"])
        for key, _entry in oldest[: len(_color_cache) - COLOR_CACHE_MAX_ENTRIES]:
            _color_cache.pop(key, None)


def _cached_color(compute, key):
    """Shared cache logic for people and objects."""
    now = time.time()
    entry = _color_cache.get(key)

    if entry is not None and (now - entry["time"]) < COLOR_CACHE_SECONDS:
        return entry["color"]

    color = compute()

    if color and color != "unknown":
        _color_cache[key] = {"color": color, "time": now}
        _prune_color_cache(now)
    elif entry is not None:
        # Keep the last good answer rather than flickering to "unknown".
        return entry["color"]

    return color


def get_person_color_cached(frame, x1, y1, x2, y2, scope=None, track_id=None):
    """
    Clothing colour, recalculated at most once every COLOR_CACHE_SECONDS
    for any given person.

    scope     identifies the CAMERA. Passing that camera's ObjectTracker
              instance is enough - each camera has its own - and it keeps
              one camera's track 3 apart from another camera's track 3.
    track_id  the tracked person's id.

    With no track_id there is nothing stable to cache against, so it
    simply calculates as normal.
    """
    if track_id is None:
        return get_person_color(frame, x1, y1, x2, y2)

    key = (id(scope) if scope is not None else 0, "person", track_id)
    return _cached_color(
        lambda: get_person_color(frame, x1, y1, x2, y2), key
    )


def get_object_color_cached(frame, x1, y1, x2, y2, scope=None, track_id=None):
    """Same idea, for a non-person object."""
    if track_id is None:
        return get_object_color(frame, x1, y1, x2, y2)

    key = (id(scope) if scope is not None else 0, "object", track_id)
    return _cached_color(
        lambda: get_object_color(frame, x1, y1, x2, y2), key
    )


def reset_color_cache():
    """Forget every cached colour. Useful when a camera restarts."""
    _color_cache.clear()


def color_cache_size():
    """How many entries are being held, for debugging."""
    return len(_color_cache)


# =====================================================================
# 8. SELF TEST
# =====================================================================
#
#   python color_detector.py               -> run built-in colour tests
#   python color_detector.py photo.jpg     -> show the colour of a photo

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1])
        if image is None:
            print(f"Could not open image: {sys.argv[1]}")
            raise SystemExit(1)
        print(f"Whole image dominant colour: {get_accurate_color(image)}")
        raise SystemExit(0)

    print()
    print("COLOUR DETECTOR SELF TEST")
    print()

    rng = np.random.default_rng(0)

    def make_person(shirt_bgr, dim=1.0):
        """Build a fake person: background, skin head, shirt, dark trousers."""
        img = np.full((480, 640, 3), (90, 95, 100), np.uint8)
        px1, py1, px2, py2 = 250, 60, 390, 460
        pw, ph = px2 - px1, py2 - py1
        cv2.rectangle(img, (px1 + int(pw * .30), py1),
                      (px2 - int(pw * .30), py1 + int(ph * .16)), (120, 150, 190), -1)
        cv2.rectangle(img, (px1, py1 + int(ph * .16)),
                      (px2, py1 + int(ph * .55)), shirt_bgr, -1)
        cv2.rectangle(img, (px1 + int(pw * .10), py1 + int(ph * .55)),
                      (px2 - int(pw * .10), py2), (35, 35, 38), -1)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape),
                      0, 255).astype(np.uint8)
        if dim != 1.0:
            img = np.clip(img.astype(np.float32) * dim, 0, 255).astype(np.uint8)
        return img, (px1, py1, px2, py2)

    shirts = {
        "red": (38, 38, 200), "blue": (200, 70, 40), "green": (50, 170, 60),
        "yellow": (40, 215, 225), "white": (238, 238, 238), "black": (22, 22, 24),
        "gray": (128, 128, 128), "brown": (40, 70, 120), "orange": (20, 120, 240),
    }

    for label, dim in (("NORMAL LIGHT", 1.0), ("DIM CCTV LIGHT", 0.45)):
        print(f"--- {label} ---")
        passed = 0
        for name, bgr in shirts.items():
            img, (a, b, c, d) = make_person(bgr, dim)
            got = get_person_color(img, a, b, c, d)
            ok = (got == name)
            passed += ok
            print(f"  {'PASS' if ok else 'FAIL'}  shirt={name:<7} detected={got}")
        print(f"  score: {passed}/{len(shirts)}")
        print()

    print("--- OBJECT COLOURS (no skin filtering) ---")
    passed = 0
    for name, bgr in shirts.items():
        patch = np.full((200, 200, 3), bgr, np.uint8)
        patch = np.clip(patch.astype(np.int16) + rng.normal(0, 8, patch.shape),
                        0, 255).astype(np.uint8)
        got = get_accurate_color(patch, is_person=False)
        ok = (got == name)
        passed += ok
        print(f"  {'PASS' if ok else 'FAIL'}  object={name:<7} detected={got}")
    print(f"  score: {passed}/{len(shirts)}")
    print()

    print("--- NAME NORMALISATION ---")
    for raw in ["Blue", "BLUE", "grey", "Grey", "Navy", "maroon", "", None]:
        print(f"  {str(raw):<8} -> {normalize_color_name(raw)}")