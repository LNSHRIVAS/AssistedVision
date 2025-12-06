from ultralytics import YOLO
class YoloDetector:
    def __init__(self, weights='yolov8n.pt', imgsz=320, conf=0.25):
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
    def detect(self, frame):
        if frame is None:
            return []
        
        # Use conf=0.15 for YOLO internal threshold, with iou=0.5 for NMS to remove duplicates
        results = self.model(frame, imgsz=self.imgsz, device='cpu', verbose=False, conf=0.15, iou=0.5)
        r = results[0]
        dets = []
        
        if hasattr(r, 'boxes') and len(r.boxes) > 0:
            for box in r.boxes:
                xyxy = box.xyxy.cpu().numpy().astype(float).tolist()[0]
                conf = float(box.conf.cpu().numpy()) if hasattr(box, 'conf') else 1.0
                cls = int(box.cls.cpu().numpy()) if hasattr(box, 'cls') else -1
                if conf < self.conf:
                    continue
                dets.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls])
            
        return dets
