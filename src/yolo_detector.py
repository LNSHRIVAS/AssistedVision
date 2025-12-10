"""
YOLO Object Detector with Caching
==================================
Runs YOLOv8n occasionally to detect obstacles, caches results.
Used to verify/enhance floor-based navigation.
"""

import time
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics not installed. Run: pip install ultralytics")


class CachedYOLODetector:
    """
    YOLO detector that runs infrequently and caches results.
    
    Detects obstacles and people to enhance navigation accuracy.
    """
    
    # Classes we care about for navigation
    OBSTACLE_CLASSES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        13: 'bench',
        16: 'dog',
        17: 'cat',
        56: 'chair',
        57: 'couch',
        58: 'potted plant',
        59: 'bed',
        60: 'dining table',
        62: 'tv',
        63: 'laptop',
        67: 'cell phone',
        73: 'suitcase',
    }
    
    def __init__(self, 
                 model_path: str = "yolov8n.pt",
                 cache_duration: float = 1.5,  # Run every 1.5 seconds
                 confidence_threshold: float = 0.4):
        """
        Args:
            model_path: Path to YOLO model
            cache_duration: Seconds to cache results
            confidence_threshold: Minimum detection confidence
        """
        self.cache_duration = cache_duration
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.enabled = False
        
        # Cache
        self.cached_detections: List[Dict] = []
        self.last_detection_time = 0
        
        # Load model
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                self.enabled = True
                print(f"✅ YOLO loaded | Model: {model_path} | Cache: {cache_duration}s")
            except Exception as e:
                print(f"⚠️ YOLO load failed: {e}")
        else:
            print("⚠️ YOLO disabled - ultralytics not installed")
    
    def detect(self, frame_bgr: np.ndarray, force: bool = False) -> List[Dict]:
        """
        Detect obstacles in frame with caching.
        
        Args:
            frame_bgr: BGR frame
            force: Force detection even if cache is valid
            
        Returns:
            List of detections: [{class_name, confidence, bbox, center_x, center_y}]
        """
        if not self.enabled or frame_bgr is None:
            return self.cached_detections
        
        # Check cache
        time_since = time.time() - self.last_detection_time
        if not force and time_since < self.cache_duration:
            return self.cached_detections
        
        # Run YOLO
        try:
            results = self.model(frame_bgr, verbose=False, conf=self.confidence_threshold)
            
            detections = []
            h, w = frame_bgr.shape[:2]
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Only keep relevant classes
                    if cls_id not in self.OBSTACLE_CLASSES:
                        continue
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2 / w  # Normalized 0-1
                    center_y = (y1 + y2) / 2 / h
                    box_width = (x2 - x1) / w
                    box_height = (y2 - y1) / h
                    
                    detections.append({
                        'class_id': cls_id,
                        'class_name': self.OBSTACLE_CLASSES[cls_id],
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': box_width,
                        'height': box_height,
                    })
            
            # Update cache
            self.cached_detections = detections
            self.last_detection_time = time.time()
            
            return detections
            
        except Exception as e:
            print(f"YOLO error: {e}")
            return self.cached_detections
    
    def get_obstacles_in_path(self, min_y: float = 0.3) -> List[Dict]:
        """
        Get obstacles that are in the walking path (lower part of image).
        
        Args:
            min_y: Minimum y position (0=top, 1=bottom). Default 0.3 = bottom 70%
            
        Returns:
            Obstacles in walking path
        """
        return [d for d in self.cached_detections if d['center_y'] > min_y]
    
    def get_clock_sector(self, detection: Dict) -> int:
        """
        Convert detection center_x to clock sector (9-3).
        
        Args:
            detection: Detection dict with center_x
            
        Returns:
            Clock hour (9, 10, 11, 12, 1, 2, 3)
        """
        x = detection['center_x']  # 0-1
        
        # Map to 7 sectors
        if x < 0.143:
            return 9
        elif x < 0.286:
            return 10
        elif x < 0.429:
            return 11
        elif x < 0.571:
            return 12
        elif x < 0.714:
            return 1
        elif x < 0.857:
            return 2
        else:
            return 3
    
    def get_blocked_sectors(self, size_threshold: float = 0.03) -> List[int]:
        """
        Get clock sectors blocked by obstacles.
        Now more aggressive - lower threshold and blocks adjacent sectors.
        
        Args:
            size_threshold: Minimum size (width*height) to consider blocking
            
        Returns:
            List of blocked clock hours
        """
        adjacent = {
            9: [10], 10: [9, 11], 11: [10, 12], 12: [11, 1], 
            1: [12, 2], 2: [1, 3], 3: [2]
        }
        blocked = set()
        for det in self.get_obstacles_in_path():
            size = det['width'] * det['height']
            if size > size_threshold:
                sector = self.get_clock_sector(det)
                blocked.add(sector)
                # Also block adjacent sectors for large obstacles
                if size > 0.08:  # 8% of screen = large obstacle
                    for adj in adjacent.get(sector, []):
                        blocked.add(adj)
        return list(blocked)


# Test
if __name__ == "__main__":
    detector = CachedYOLODetector()
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        blocked = detector.get_blocked_sectors()
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = [int(x) for x in det['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        # Show blocked sectors
        cv2.putText(frame, f"Blocked: {blocked}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        cv2.imshow("YOLO", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
