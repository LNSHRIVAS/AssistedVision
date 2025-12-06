#!/bin/bash
VIDEO=${1:-data/demo.mp4}
YOLO=${2:-yolov8n.pt}
OUT=${3:-output/out.mp4}
python3 src/main.py --video "$VIDEO" --yolo "$YOLO" --output "$OUT" --imgsz 320 --skip_depth 5 --no-show
