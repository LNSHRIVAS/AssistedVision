# src/main.py
# ----------------------------------------
import time
import cv2
import json
from detection.detector import YoloDetector
from guidance.risk_assessor import annotate_risk, bottom_strip_polygon

from deep_sort_realtime.deepsort_tracker import DeepSort  

def main():
    weights = "yolov8n.pt"
    conf = 0.25
    iou = 0.45
    device = None


    risk_mode = "bottom"
    risk_margin = 0.25
    custom_polygon = None
    video_path = "data/demo.mp4"
    output_file = open("data/output_tr.jsonl", "a")

    detector = YoloDetector(weights=weights, conf=conf, iou=iou, device=device)
    tracker = DeepSort(max_age=30)


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_path}")
        return

    frame_id = 0
    prev_positions = {}  # 记录track_id上一次位置和时间

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts = time.time()
        packet = detector.detect_frame(frame, frame_id=frame_id, ts=ts)

        INTERESTING = {
            "person","bicycle","car","motorbike","bus","truck",
            "chair","bench","table","sofa"
        }

        H, W = frame.shape[:2]
        packet = annotate_risk(
            packet,
            frame_size=(W, H),
            mode="bottom",
            margin_ratio=risk_margin,
            polygon=custom_polygon,
            interesting_classes=INTERESTING
        )

        detections_for_tracking = []
        for det in packet["payload"]["detections"]:
            x1, y1, x2, y2 = det["box"]
            conf_score = det["conf"]
            cls = det["cls"]
            detections_for_tracking.append(([x1, y1, x2 - x1, y2 - y1], conf_score, cls))

        tracks = tracker.update_tracks(detections_for_tracking, frame=frame)

        # 
        for det in packet["payload"]["detections"]:
            det["track_id"] = None

        # 
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            # 
            for det in packet["payload"]["detections"]:
                x1, y1, x2, y2 = det["box"]
                # 
                if abs(x1 - l) < 10 and abs(y1 - t) < 10 and abs(x2 - r) < 10 and abs(y2 - b) < 10:
                    det["track_id"] = track_id

        # 
        for det in packet["payload"]["detections"]:
            track_id = det.get("track_id")
            if track_id is None:
                continue
            x1, y1, x2, y2 = det["box"]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            if track_id in prev_positions:
                x_prev, y_prev, t_prev = prev_positions[track_id]
                dt = ts - t_prev
                if dt > 0:
                    vx = (x_center - x_prev) / dt
                    vy = (y_center - y_prev) / dt
                    speed = (vx ** 2 + vy ** 2) ** 0.5
                    det["speed"] = speed
                    det["direction"] = [vx, vy]
            prev_positions[track_id] = (x_center, y_center, ts)

        # 
        print(packet)
        output_file.write(json.dumps(packet) + "\n")

        # 
        if "risk_zone_polygon" in packet["payload"]:
            vis = frame.copy()
            poly = packet["payload"]["risk_zone_polygon"]
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % len(poly)]
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

            for det in packet["payload"]["detections"]:
                x1, y1, x2, y2 = det["box"]
                color = (0, 255, 0) if det.get("risk_zone") == "safe" else (0, 0, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f'{det["cls"]} {det["conf"]:.2f}'
                if det.get("track_id") is not None:
                    label += f' ID:{det["track_id"]}'
                if det.get("speed") is not None:
                    label += f' V:{det["speed"]:.2f}'
                cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            cv2.imshow("AssistiveVision (Risk View)", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    output_file.close()

if __name__ == "__main__":
    main()
