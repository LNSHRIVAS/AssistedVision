  # src/main.py
# ------------------------------------------------------------
# PURPOSE (in simple words):
#   - This is the "glue" that runs the system.
#   - It opens a video (or webcam), gets frames, runs YOLO detection,
#     then runs risk assessment, and prints (or visualizes) the results.
#
# HOW TO RUN:
#   conda activate AssistedVision
#   python src/main.py
#
# TIPS:
#   - Change "video_path" to your own file.
#   - If you want webcam: set video_path = 0
#   - Press ESC in the window to stop.
# ------------------------------------------------------------

import time
import cv2

# import our two components
from detection.detector import YoloDetector
from guidance.risk_assessor import annotate_risk, bottom_strip_polygon


def main():
    # -----------------------------
    # 1) CONFIGURATION
    # -----------------------------
    # YOLO model weights: small/fast start is "yolov8n.pt"
    weights = "yolov8n.pt"

    # Confidence & IOU thresholds (can tweak later)
    conf = 0.25
    iou = 0.45

    # Device:
    #   - None -> YOLO chooses
    #   - "cpu"
    #   - "cuda:0" if you have an NVIDIA GPU + CUDA PyTorch installed
    device = None

    # Choose RISK MODE:
    #   - "bottom": use bottom strip of the frame as risk area
    #   - "polygon": use your own polygon (set below)
    #   - "none": disable risk tagging
    risk_mode = "bottom"
    risk_margin = 0.25  # bottom 25% of the frame is "danger"
    custom_polygon = None  # e.g., [(100,400),(540,400),(639,479),(0,479)]

    # Video input:
    #   - Path to a video file (e.g., "data/demo.mp4")
    #   - Or 0 for webcam
    video_path = "data/demo.mp4"  # change this to your file; or set 0 for webcam

    # -----------------------------
    # 2) INITIALIZE DETECTOR
    # -----------------------------
    detector = YoloDetector(weights=weights, conf=conf, iou=iou, device=device)

    # -----------------------------
    # 3) OPEN VIDEO / WEBCAM
    # -----------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_path}")
        return

    frame_id = 0

    # -----------------------------
    # 4) MAIN LOOP: READ, DETECT, RISK, SHOW/PRINT
    # -----------------------------
    while True:
        ok, frame = cap.read()
        if not ok:
            # End of video (or webcam error)
            break

        # Current time as float seconds
        ts = time.time()

        # Run detection on this frame → returns your required JSON
        packet = detector.detect_frame(frame, frame_id=frame_id, ts=ts)
        
        # keep only classes we care about (optional)
        INTERESTING = {
        "person","bicycle","car","motorbike","bus","truck",
        "chair","bench","table","sofa" 
        }

        # Add risk info (safe/danger) + polygon
        H, W = frame.shape[:2]
        
        packet = annotate_risk(
            packet,
            frame_size=(W, H),
            mode="bottom",            # "bottom" | "trapezoid" | "polygon" | "none"
            margin_ratio=risk_margin,
            polygon=custom_polygon,
            interesting_classes=INTERESTING  # remove this arg to keep all classes
        )

        # For now, just print packet (you can replace with saving to file or sending to a message bus)
        print(packet)

        # OPTIONAL: Visualization to "see" risk zone polygon
        # - Draw the risk polygon
        if "risk_zone_polygon" in packet["payload"]:
            vis = frame.copy()
            poly = packet["payload"]["risk_zone_polygon"]
            # Draw polygon edges in yellow
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % len(poly)]
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Draw each detection as a box: red if danger, green if safe
            for det in packet["payload"]["detections"]:
                x1, y1, x2, y2 = det["box"]
                color = (0, 255, 0) if det.get("risk_zone") == "safe" else (0, 0, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f'{det["cls"]} {det["conf"]:.2f}'
                cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # Show the result window
            cv2.imshow("AssistiveVision (Risk View)", vis)
            # ESC to quit
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_id += 1

    # -----------------------------
    # 5) CLEANUP
    # -----------------------------
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
