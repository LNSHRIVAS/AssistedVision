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
AssistiveVision/
├── src/               # Core source code
│   ├── detection/     # Object detection (YOLO)
│   ├── tracking/      # Multi-object tracking
│   ├── depth/         # Depth estimation or LiDAR integration
│   ├── guidance/      # Voice guidance / LLM integration
│   ├── utils/         # Helper functions (logging, configs)
│   └── main.py        # System entry point
│
├── experiments/       # Jupyter notebooks and prototypes
├── data/              # Datasets (raw, processed, samples)
├── docs/              # Reports, slides, and documentation
├── tests/             # Unit tests
├── requirements.txt   # Project dependencies
├── .gitignore
├── LICENSE
└── README.md

# AssistiveVision (v5.0, code-only)

**What it does**
- YOLOv8n detects **all COCO classes (80)**
- Lightweight class-aware IoU tracker gives **stable track_id**, velocity, and speed
- Per-object **risk score** = 0.55·proximity + 0.30·depth + 0.15·motion, then × social weight
- Optional JSON/JSONL/video outputs (disabled by default)
- Designed to be **fast, readable, and extendable** for the team

## Quickstart
```bash
# create env (conda or venv; your choice)
pip install -r requirements.txt

# run (video file)
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python -m src.main --source data/demo.mp4 --show --imgsz 448 --conf 0.45 --det-every 2

# run (webcam)
python -m src.main --source 0 --show



