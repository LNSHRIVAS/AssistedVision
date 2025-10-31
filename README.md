# AssistiveVision 👁️  
*Empowering accessibility with AI-driven computer vision*  

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen)  
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)  

---

## 📖 Overview  
**AssistiveVision** is an AI-powered system designed to help visually impaired individuals navigate their surroundings safely.  
It combines **computer vision**, **machine learning**, and **assistive audio feedback** to:  
- Detect and track objects in real-time  
- Estimate depth or distance to obstacles  
- Provide clear voice guidance to the user  

The system is designed for **proactive navigation**, and can be adapted for **mobile devices**.  

---

## 🗂️ Project Structure  

```bash
AssistedVision/
├─ data/
│  ├─ demo.mp4                  # sample video (you can change the source)
├─ docs/
│  └─ README_assets/            # (optional) put screenshots here
├─ results/                      # created at runtime
│  ├─ json/                     # per-frame JSON files
│  ├─ stream.jsonl              # per-second aggregates (JSONL)
│  └─ overlay_v50.mp4           # optional rendered video
├─ src/
│  ├─ main.py                   # entrypoint (v5.0 pipeline)
│  ├─ detection/                # (reserved, currently handled in main via YOLO)
│  ├─ tracking/
│  │  └─ box_tracker.py         # class-aware IoU tracker with velocity
│  ├─ utils/                    # (reserved for future helpers)
│  └─ semantic/                 # (reserved; earlier panoptic code removed for speed)
├─ tests/
├─ environment.yml              # conda env spec (Python 3.10 + pip deps)
├─ requirements.txt             # pip fallback (ultralytics, opencv, torch, etc.)
├─ yolov8n.pt                   # tiny YOLO weights (downloaded by ultralytics if missing)
└─ README.md                    # this file

To run the file : python -m src.main --source data/demo.mp4 --show \
  --imgsz 448 --conf 0.45 --det-every 2 \
  --json_dir results/json \
  --jsonl results/stream.jsonl --agg_per_sec \
  --write_video results/overlay_v50.mp4

What the model computes
1) Per-object risk (0–1)
For each tracked object:
Proximity: how far its bottom is into the “walkable band” (lower screen).
Depth proxy: how tall the bbox is relative to the image height (taller ≈ closer).
Motion: magnitude of EMA velocity (px/sec), small weight to avoid jitter.
Social weight: class multiplier (e.g., buses/cars > benches/chairs).
Final risk = (0.55*proximity + 0.30*depth + 0.15*motion) * social_weight, clamped to [0,1].
These weights live in main.py → W_PROXIMITY/W_DEPTH/W_MOTION and SOCIAL_WEIGHTS.
2) Scene heatmap & scene_risk
We add soft “risk splats” around risky objects only in the lower band of the image and apply a mild decay.
scene_risk = mean(heatmap[lower_band]) → single scalar per frame.

Output formats : 
{
  "frame_id": 123,
  "timestamp": 1761805000.12,
  "scene_risk": 0.067,
  "objects": [
    {
      "cls": "person",
      "conf": 0.84,
      "box": [x1, y1, x2, y2],
      "track_id": 7,
      "center": [cx, cy],
      "velocity": [vx_px_s, vy_px_s],
      "speed": 22.5,
      "risk": 0.34,
      "risk_components": { "proximity": 0.42, "depth": 0.28, "motion": 0.19 },
      "social_weight": 1.0,
      "sector": "center"
    }
  ]
}


# AssistedVision — Detection + Risk Zone (Milestone 1)

This milestone implements **YOLOv8 object detection** and a **risk zone** tagger that marks each detection as `safe` / `warn` / `danger` with a `risk_score` (0..1).  
The risk zone currently uses a **bottom-of-frame** heuristic (tunable).

