from __future__ import annotations
import time
from typing import Dict, Any, List, Optional
import numpy as np
from ultralytics import YOLO


class YoloDetector:
    """
    A simple wrapper around YOLOv8 for our project.

    - You can choose the model weights, confidence threshold, IOU threshold, and device.
    - detect_frame(...) returns a dictionary with:
        {
          "type": "DETECTIONS",
          "ts": <timestamp>,
          "frame_id": <int>,
          "payload": {
             "detections": [
               {
                 "track_id": None,      # tracker will fill later
                 "cls": <label>,        # e.g. "person"
                 "cls_id": <int>,       # YOLO class id
                 "conf": <float 0..1>,  # confidence score
                 "box": [x1,y1,x2,y2]   # absolute pixel coordinates
               },
               ...
             ]
          }
        }
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",   # "n" = nano model (small & fast). You can switch to "yolov8s.pt" etc.
        conf: float = 0.25,            # confidence threshold (lower = more detections, more false positives)
        iou: float = 0.45,             # IOU threshold for NMS (how much boxes can overlap)
        device: Optional[str] = None   # examples: None (auto), "cpu", "cuda:0"
    ):
        # Load the YOLO model
        self.model = YOLO(weights)
        # Fuse layers to run a bit faster (safe to call)
        self.model.fuse()

        # If you specify a device, move the model there
        if device:
            self.model.to(device)

        self.conf = conf
        self.iou = iou

    def detect_frame(self, frame: np.ndarray, frame_id: int, ts: Optional[float] = None) -> Dict[str, Any]:
        """
        Run YOLO on one frame and return your required JSON.

        INPUTS:
        - frame: numpy array (H,W,3), BGR or RGB both okay.
        - frame_id: integer counter for the current frame.
        - ts: optional timestamp (float seconds). If None, we use current time.

        OUTPUT:
        - dictionary with keys: "type", "ts", "frame_id", "payload"
        """
        # If no timestamp is provided, use now()
        if ts is None:
            ts = float(time.time())

        # Get height/width to clamp boxes later
        h, w = frame.shape[:2]

        # Run YOLO prediction on this frame
        # verbose=False to keep console clean
        results = self.model.predict(source=frame, conf=self.conf, iou=self.iou, verbose=False)[0]

        # This dict maps class ids to human-readable labels, e.g. {0:"person", ...}
        names = self.model.names

        # Prepare the list of detections to fill in
        detections: List[Dict[str, Any]] = []

        # If YOLO returned any boxes, process them
        if results.boxes is not None and len(results.boxes) > 0:
            # xyxy: [x1, y1, x2, y2] for each detection
            xyxy = results.boxes.xyxy.cpu().numpy()
            # conf: confidence for each detection
            confs = results.boxes.conf.cpu().numpy()
            # cls: class id for each detection (integers)
            clss = results.boxes.cls.cpu().numpy().astype(int)

            # Loop over detections and convert each to our desired dict format
            for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, clss):
                # Clamp coords to image boundaries and convert to int
                x1 = int(max(0, min(w - 1, x1)))
                y1 = int(max(0, min(h - 1, y1)))
                x2 = int(max(0, min(w - 1, x2)))
                y2 = int(max(0, min(h - 1, y2)))

                detections.append({
                    "track_id": None,                     # tracker will fill later (keep None for now)
                    "cls": names.get(cid, str(cid)),      # human-readable class label
                    "cls_id": int(cid),                   # YOLO class id
                    "conf": float(round(float(c), 4)),    # round to 4 decimals to keep it neat
                    "box": [x1, y1, x2, y2],              # absolute pixel coords [x1,y1,x2,y2]
                })

        # Build the final packet exactly in your required structure
        packet = {
            "type": "DETECTIONS",
            "ts": float(ts),
            "frame_id": int(frame_id),
            "payload": {
                "detections": detections
            }
        }
        return packet
