# src/main.py
# AssistiveVision v5.0 — code-only
# - YOLOv8 detection (all COCO classes)
# - Lightweight class-aware IoU tracker with velocity EMA
# - Fast risk score per object (proximity + depth proxy + motion) * social weight
# - Optional JSON/JSONL/video outputs (disabled unless flags are given)

import os, time, json, argparse, collections, math
import numpy as np
import cv2
from ultralytics import YOLO
from tracking.box_tracker import BoxTracker

# ---------------- Tunables (fast, simple, explainable) ----------------
W_PROXIMITY = 0.55
W_DEPTH     = 0.30
W_MOTION    = 0.15

# Social weights (extend as needed, normalize globally)
def _norm(s): 
    return s.replace(" ", "").replace("_", "").lower()

SOCIAL_WEIGHTS = collections.defaultdict(lambda: 1.0, {
    _norm("person"): 1.00,
    _norm("bicycle"): 1.10,
    _norm("motorcycle"): 1.20,
    _norm("car"): 1.30,
    _norm("bus"): 1.50,
    _norm("truck"): 1.45,
    _norm("train"): 1.40,
    _norm("boat"): 1.25,
    _norm("bench"): 0.6,
    _norm("chair"): 0.6,
    _norm("sofa"): 0.6,
    _norm("potted plant"): 0.8
})
# ---------------------------------------------------------------------

def norm01(x, lo, hi):
    if hi <= lo: return 0.0
    return float(max(0.0, min(1.0, (x - lo) / (hi - lo))))

def score_risk(det, H):
    """
    det: dict with keys ['cls','box','speed','center',...]
    Returns: risk (0..1), components dict, social_weight
    """
    x1, y1, x2, y2 = det["box"]
    h = max(1, y2 - y1)
    bottom = y2

    # Proximity: how deep bottom lies in lower 30% band
    band_top = int(H * 0.70)
    if bottom <= band_top:
        prox = 0.0
    else:
        prox = norm01(bottom, band_top, H)

    # Depth proxy: bbox height normalized by image height
    depth = norm01(h / H, 0.05, 0.60)

    # Motion: speed normalized by 200 px/s (heuristic)
    motion = norm01(det.get("speed", 0.0), 0.0, 200.0)

    base = W_PROXIMITY * prox + W_DEPTH * depth + W_MOTION * motion
    sw = SOCIAL_WEIGHTS[det["cls"]]
    risk = max(0.0, min(1.0, base * sw))
    return risk, {"proximity": prox, "depth": depth, "motion": motion}, sw

def draw_overlay(frame, tracked, fps=None, scene_risk=None):
    H, W = frame.shape[:2]
    out = frame.copy()

    # Subtle hint of lower-risk band (no bold line)
    band_top = int(H * 0.70)
    cv2.rectangle(out, (0, band_top), (W, H), (40, 40, 40), 1)

    for tr in tracked:
        x1, y1, x2, y2 = tr["box"]
        rid = tr.get("track_id", -1)
        r   = tr.get("risk", 0.0)
        cls = tr["cls"]

        # color from green→red by risk
        color = (0, int((1 - r) * 180 + 50), int(r * 220))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        lbl = f"#{rid} {cls} r={r:.2f}"
        cv2.putText(out, lbl, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)

    # HUD
    y = 18
    if fps is not None:
        cv2.putText(out, f"FPS: {fps:.1f}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2, cv2.LINE_AA)
        y += 22
    if scene_risk is not None:
        cv2.putText(out, f"scene_risk: {scene_risk:.3f}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2, cv2.LINE_AA)
    return out

def ensure_dir(p: str):
    if p: os.makedirs(p, exist_ok=True)

def parse_args():
    ap = argparse.ArgumentParser("AssistiveVision v5.0 — code-only")
    ap.add_argument("--source", type=str, default="data/demo.mp4", help="path or 0 for webcam")
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--det-every", type=int, default=2, help="run detector every N frames")
    ap.add_argument("--device", type=str, default=None, help="cpu|cuda:0|mps (auto if None)")
    ap.add_argument("--show", action="store_true", help="visualize overlay")
    # Optional logging (off unless provided)
    ap.add_argument("--json_dir", type=str, default=None, help="write per-frame JSON to this dir")
    ap.add_argument("--jsonl", type=str, default=None, help="append per-second JSONL aggregates here")
    ap.add_argument("--agg_per_sec", action="store_true", help="enable per-second aggregation")
    ap.add_argument("--write_video", type=str, default=None, help="save rendered MP4 (disabled if not set)")
    return ap.parse_args()

def main():
    args = parse_args()
    source = 0 if str(args.source) == "0" else args.source

    # YOLO (auto-download yolov8n)
    yolo = YOLO("yolov8n.pt")
    names = yolo.model.names  # class id -> name

    # Capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] cannot open source: {source}")
        return
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps_nom = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Tracker
    tracker = BoxTracker(iou_thresh=0.3, max_missed=8, ema=0.6)

    # Optional writers
    vw = None
    if args.write_video:
        ensure_dir(os.path.dirname(args.write_video) or ".")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(args.write_video, fourcc, fps_nom, (W, H))

    if args.json_dir:
        ensure_dir(args.json_dir)

    t0 = time.time()
    last_det = -10**9
    frame_id = 0
    last_sec = None
    sec_accum = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        now_sec = int(now)

        # Detect every N frames
        dets = []
        if (frame_id - last_det) >= args.det_every:
            last_det = frame_id
            r = yolo(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                     device=args.device or None, verbose=False)
            boxes = r[0].boxes
            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    xyxy = b.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                    conf   = float(b.conf[0].item()) if b.conf is not None else 0.0
                    cls_raw = names.get(cls_id, str(cls_id))
                    # normalized class string for consistent matching/keys
                    cls = cls_raw.replace(" ", "").replace("_", "").lower()
                    dets.append({
                        "cls": cls,
                        "cls_id": cls_id,
                        "cls_raw": cls_raw,
                        "conf": conf,
                        "box": [x1, y1, x2, y2]
                    })

        # Track all objects
        tracked = tracker.update(dets, (H, W), now=now)

        # Risk per object
        risks = []
        for tr in tracked:
            r, comps, sw = score_risk(tr, H)
            tr["risk"] = r
            tr["risk_components"] = comps
            tr["social_weight"] = sw
            risks.append(r)
        scene_risk = float(np.mean(risks)) if risks else 0.0

        # Optional per-frame JSON
        if args.json_dir:
            rec = {
                "frame_id": frame_id,
                "timestamp": now,
                "scene_risk": round(scene_risk, 3),
                "objects": [
                    {
                        "cls": tr["cls_raw"] if "cls_raw" in tr else tr["cls"],
                        "conf": round(tr.get("conf", 0.0), 4),
                        "box": tr["box"],
                        "track_id": tr["track_id"],
                        "center": tr["center"],
                        "velocity": [round(v, 2) for v in tr["velocity"]],
                        "speed": round(tr["speed"], 2),
                        "risk": round(tr["risk"], 3),
                        "risk_components": {k: round(v, 3) for k, v in tr["risk_components"].items()},
                        "social_weight": tr["social_weight"]
                    } for tr in tracked
                ]
            }
            with open(os.path.join(args.json_dir, f"frame_{frame_id:06d}.json"), "w") as f:
                json.dump(rec, f)

        # Optional per-second JSONL aggregation
        if args.agg_per_sec and args.jsonl:
            if last_sec is None:
                last_sec = now_sec
            if now_sec == last_sec:
                sec_accum.append(scene_risk)
            else:
                avg = float(np.mean(sec_accum)) if sec_accum else 0.0
                with open(args.jsonl, "a") as f:
                    f.write(json.dumps({"type": "second", "sec": last_sec, "avg_risk": round(avg, 3)}) + "\n")
                sec_accum = [scene_risk]
                last_sec = now_sec

        # Overlay
        dt = max(1e-6, (time.time() - t0))
        fps = (frame_id + 1) / dt
        vis = draw_overlay(frame, tracked, fps=fps, scene_risk=scene_risk)

        if args.show:
            cv2.imshow("AssistiveVision", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if vw is not None:
            vw.write(vis)

        frame_id += 1

    # flush last sec aggregate
    if args.agg_per_sec and args.jsonl and sec_accum:
        avg = float(np.mean(sec_accum)) if sec_accum else 0.0
        with open(args.jsonl, "a") as f:
            f.write(json.dumps({"type": "second", "sec": last_sec, "avg_risk": round(avg, 3)}) + "\n")

    cap.release()
    if vw is not None: vw.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
