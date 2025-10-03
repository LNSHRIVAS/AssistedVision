from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional

Point = Tuple[int, int]  # (x, y) pixel coordinate


def _point_in_polygon(pt: Point, poly: List[Point]) -> bool:
    """
    Classic "ray casting" algorithm to check if a point is inside a polygon.
    - pt: (x,y)
    - poly: list of (x,y) vertices, in order around the shape.

    Returns True if inside, False if outside.
    """
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # Check if the horizontal ray crosses the edge (x1,y1)->(x2,y2)
        intersects = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1)
        if intersects:
            inside = not inside
    return inside


def bottom_strip_polygon(w: int, h: int, margin_ratio: float = 0.25) -> List[Point]:
    """
    Define a simple risk zone: the bottom 'margin_ratio' of the image.
    Example: margin_ratio = 0.25 means bottom 25% of the frame.

    We return the polygon as 4 points (rectangle).
    """
    y_top = int(h * (1.0 - margin_ratio))
    return [(0, y_top), (w, y_top), (w, h), (0, h)]


def annotate_risk(
    packet: Dict[str, Any],
    frame_size: Tuple[int, int],           # (W, H)
    mode: str = "bottom",                  # "bottom" | "polygon" | "trapezoid" | "none"
    margin_ratio: float = 0.25,            # used when mode=="bottom" (bottom % of image)
    polygon: Optional[List[Point]] = None, # used when mode=="polygon"
    *,
    # --- scoring knobs (tweak if you like) ---
    size_weight: float = 0.55,             # weight for bbox height ratio
    y_weight: float = 0.45,                # weight for centroid y ratio
    inside_bonus: float = 0.15,            # boost if centroid lies inside polygon
    warn_threshold: float = 0.50,
    danger_threshold: float = 0.75,
    interesting_classes: Optional[set] = None  # e.g., {"person","chair","car"} to filter
) -> Dict[str, Any]:
    """
    Adds a continuous 'risk_score' (0..1) and discrete 'risk_zone' label
    ('safe' | 'warn' | 'danger') to each detection. Also attaches the polygon
    we used as 'risk_zone_polygon' for visualization.

    The score is a simple, cheap distance proxy:
      risk_score = size_weight * (box_h / H) + y_weight * (cy / H) + inside_bonus_if_inside

    - Larger boxes (taller) -> nearer -> higher score
    - Lower centroid (closer to feet on ground plane) -> higher score
    - Being inside the polygon adds a small bonus
    """

    if mode == "none":
        return packet

    W, H = frame_size

    # choose polygon
    if mode == "bottom":
        poly = bottom_strip_polygon(W, H, margin_ratio)
    elif mode == "polygon" and polygon:
        poly = polygon
    elif mode == "trapezoid":
        poly = trapezoid_ground_polygon(W, H, near_height=margin_ratio, far_y=0.55, far_half_width=0.25)
    else:
        # invalid config -> leave packet unchanged
        return packet

    payload = packet.get("payload", {})
    dets = payload.get("detections", [])

    # optional class filter to reduce noise
    if interesting_classes:
        dets[:] = [d for d in dets if d.get("cls") in interesting_classes]

    # score + label each detection
    for det in dets:
        x1, y1, x2, y2 = det["box"]
        box_h = max(1, y2 - y1)
        cx = int(0.5 * (x1 + x2))
        cy = int(0.5 * (y1 + y2))

        # ratios in [0..1]
        size_ratio = box_h / H
        y_ratio    = cy / H

        # base score
        score = size_weight * size_ratio + y_weight * y_ratio

        # polygon membership
        inside = _point_in_polygon((cx, cy), poly)
        if inside:
            score += inside_bonus

        # clamp 0..1
        score = float(max(0.0, min(1.0, score)))

        # map to label
        if score >= danger_threshold:
            label = "danger"
        elif score >= warn_threshold:
            label = "warn"
        else:
            label = "safe"

        det["risk_score"] = round(score, 3)
        det["risk_zone"]  = label

    # attach polygon so UI can draw it
    packet["payload"]["risk_zone_polygon"] = poly
    return packet


def trapezoid_ground_polygon(
    w: int,
    h: int,
    near_height: float = 0.22,   # how tall the near band is (bottom part) 0..1
    far_y: float = 0.55,         # y-position (0..1 of height) where trapezoid narrows (≈ horizon below mid)
    far_half_width: float = 0.25 # half width at the far_y (fraction of width)
) -> List[Point]:
    """
    Returns a trapezoid approximating the ground in front of the camera.
    Bottom edge is full width; top edge is narrower around 'far_y'.
    Tweak far_y up/down depending on camera tilt.
    """
    y_near_top = int(h * (1.0 - near_height))
    y_far      = int(h * far_y)
    x_far_left = int(w * (0.5 - far_half_width))
    x_far_right= int(w * (0.5 + far_half_width))
    # polygon in clockwise order
    return [(0, y_near_top), (w, y_near_top), (w, h), (0, h), (0, h), (x_far_right, y_far), (x_far_left, y_far), (0, h)]
    # simpler: 4-point trapezoid (recommended)
    # return [(0, y_near_top), (w, y_near_top), (int(w*(0.5+far_half_width)), y_far), (int(w*(0.5-far_half_width)), y_far)]
