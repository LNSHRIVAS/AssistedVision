# src/tracking/box_tracker.py
# Fast class-aware IoU tracker with velocity EMA (no 3rd-party deps)

import time, math

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

class BoxTracker:
    def __init__(self, iou_thresh=0.3, max_missed=8, ema=0.6):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.ema = ema
        self.tracks = {}   # id -> track dict
        self.next_id = 1
        self.last_time = None

    def _center(self, box):
        x1, y1, x2, y2 = box
        # return float for smoother velocity
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


    def update(self, detections, frame_shape, now=None):
        """
        detections: list of {cls, conf, box}
        returns: list of tracks [{cls, conf, box, track_id, center, velocity [vx,vy], speed}]
        """
        if now is None:
            now = time.time()
        H, W = frame_shape

        # dt
        if self.last_time is None:
            dt = 1 / 30.0
        else:
            dt = max(1e-3, now - self.last_time)
        self.last_time = now

        dets = detections or []
        used = [False] * len(dets)

        # first, try to match existing tracks by class + IoU
        for tid, tr in list(self.tracks.items()):
            tr["missed"] += 1
            best_iou, best_idx = 0.0, -1
            for i, d in enumerate(dets):
                if used[i]: 
                    continue
                if d.get("cls_id", None) != tr.get("cls_id", None):
                    continue
                ov = iou(d["box"], tr["box"])
                if ov > best_iou:
                    best_iou, best_idx = ov, i
            if best_idx >= 0 and best_iou >= self.iou_thresh:
                d = dets[best_idx]
                used[best_idx] = True

                old_cx, old_cy = tr["center"]
                new_cx, new_cy = self._center(d["box"])

                vx = (new_cx - old_cx) / dt
                vy = (new_cy - old_cy) / dt
                # EMA smoothing
                tr["velocity"][0] = self.ema * tr["velocity"][0] + (1 - self.ema) * vx
                tr["velocity"][1] = self.ema * tr["velocity"][1] + (1 - self.ema) * vy
                tr["speed"] = math.hypot(tr["velocity"][0], tr["velocity"][1])

                tr["box"] = d["box"]
                tr["center"] = [new_cx, new_cy]
                tr["conf"] = float(d.get("conf", 0.0))
                tr["missed"] = 0
            else:
                # decay velocity when missed
                tr["velocity"][0] *= 0.9
                tr["velocity"][1] *= 0.9
                tr["speed"] = math.hypot(tr["velocity"][0], tr["velocity"][1])

        # create new tracks for unmatched detections
        for i, d in enumerate(dets):
            if used[i]: 
                continue
            cx, cy = self._center(d["box"])
            #self.tracks[self.next_id] = {
            #    "track_id": self.next_id,
            #    "cls": d["cls"],
            #    "conf": float(d.get("conf", 0.0)),
            #    "box": d["box"],
            #    "center": [cx, cy],
            #    "velocity": [0.0, 0.0],
            #    "speed": 0.0,
            #    "missed": 0
            #}
            self.tracks[self.next_id] = {
                "track_id": self.next_id,
                "cls": d["cls"],          # normalized string
                "cls_raw": d.get("cls_raw", d["cls"]),
                "cls_id": d.get("cls_id", None),
                "conf": float(d.get("conf", 0.0)),
                "box": d["box"],
                "center": [cx, cy],       # floats
                "velocity": [0.0, 0.0],
                "speed": 0.0,
                "missed": 0
            }
            self.next_id += 1

        # drop long-missed
        for tid in [tid for tid, tr in self.tracks.items() if tr["missed"] > self.max_missed]:
            del self.tracks[tid]

        return [dict(tr) for tr in self.tracks.values()]
