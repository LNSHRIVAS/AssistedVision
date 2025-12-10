"""
Fast Navigation Server with ASYNC Processing
=============================================
Uses threading to eliminate lag:
- Main thread: receives frames, sends responses
- Worker thread: runs ML models in background
- Always responds immediately with latest result
"""

import asyncio
import websockets
import json
import struct
import numpy as np
import cv2
import logging
import sys
import os
import argparse
import threading
import queue
from datetime import datetime
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

from depth_clean import DepthEstimator
from floor_segmentation import FloorSegmentationModel
from simple_navigator import SimpleDepthNavigator
from clock_direction import ClockDirectionConverter
from yolo_detector import CachedYOLODetector

# Use all CPU cores for PyTorch
import torch
torch.set_num_threads(8)


class FastNavigationServer:
    """Navigation server with async model processing - NO LAG."""
    
    def __init__(self, host='0.0.0.0', port=8768, outdoor_mode=False):
        self.host = host
        self.port = port
        self.outdoor_mode = outdoor_mode
        
        mode_str = "OUTDOOR (sidewalk only)" if outdoor_mode else "INDOOR (all floor)"
        print(f"⏳ Loading models ({mode_str})...")
        
        # Floor model - HIGHER resolution for accuracy (runs less often)
        self.seg_model = FloorSegmentationModel(
            device='cpu',
            target_size=(192, 144)  # Higher res for better floor detection
        )
        # Enable outdoor mode if specified (only detect sidewalk/road, not grass)
        if outdoor_mode:
            self.seg_model.set_outdoor_mode(True)
        
        # Depth model - higher res for stability
        self.depth_model = DepthEstimator(
            device='cpu',
            model_name='depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',
            max_depth=10.0,
            target_size=(128, 96)
        )
        
        # Simple depth-based navigator (reliable, not floor-obsessed)
        self.navigator = SimpleDepthNavigator()
        
        # YOLO object detector with caching (runs every 1.5s)
        self.yolo_detector = CachedYOLODetector(
            model_path="yolov8n.pt",
            cache_duration=1.5,  # Run every 1.5 seconds
            confidence_threshold=0.4
        )
        self.yolo_blocked_sectors = []  # Sectors blocked by YOLO detections
        
        # Clock direction converter (tracks user heading)
        self.clock_converter = ClockDirectionConverter(smoothing_window=5)
        self.user_heading = 0.0  # Current user heading from IMU
        self.user_clock = 12     # What clock direction user is facing (relative to calibrated 12)
        
        # === MOVEMENT TRACKING ===
        self.last_recommended_clock = 12   # Direction we're guiding toward
        self.user_aligned = False          # Has user turned to recommended direction?
        self.target_heading = None         # Real-world heading of target direction
        
        # === OBSTACLE WARNING CACHE ===
        # Don't repeat warnings for same obstacles
        self.warned_obstacles = set()      # Set of (obstacle_type, sector) we've warned about
        self.last_obstacle_clear_time = datetime.now()
        
        # Frame queue for async processing
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_lock = threading.Lock()
        
        # Latest results
        self.latest_depth = None
        self.latest_floor = None
        self.latest_frame = None
        self.center_depth = 5.0
        self.floor_coverage = 1.0
        self.recommended_clock = 12
        self.sector_depths = [0.5] * 5  # 5 quadrants: 10, 11, 12, 1, 2
        self.corridor_overlay = None    # Corridor visualization overlay
        
        # === TEMPORAL SMOOTHING with HYSTERESIS ===
        self.depth_history = deque(maxlen=10)
        self.floor_history = deque(maxlen=8)
        self.floor_low_count = 0
        self.floor_low_threshold = 3
        
        # === INSTRUCTION COOLDOWN ===
        self.guidance_text = "Starting..."
        self.last_spoken = ""
        self.last_speak_time = datetime.now()
        self.speak_cooldown_sec = 3.0
        
        # Stats
        self.fps = 0
        self.process_times = deque(maxlen=10)
        self.worker_cycle = 0
        
        # Visualization
        self.vis_running = True
        self.vis_thread = None
        
        # Start worker thread
        self.worker_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        print("✅ FAST MODE with smoothing and cooldown")
    
    def _worker_loop(self):
        """Background thread that processes frames with SMOOTHING."""
        while self.worker_running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
                
                start = datetime.now()
                self.worker_cycle += 1
                
                # FLOOR: Run every cycle
                floor_mask = self.seg_model.predict(frame)
                
                # DEPTH: Run every cycle now (caching handles speed)
                depth_map = self.depth_model.estimate_full_map(frame)
                
                if depth_map is not None:
                    dh, dw = depth_map.shape
                    center = depth_map[dh//3:2*dh//3, dw//3:2*dw//3]
                    valid = center[(center > 0.3) & (center < 10)]
                    raw_depth = float(np.percentile(valid, 20)) if len(valid) > 0 else 5.0
                    
                    # === TEMPORAL SMOOTHING for depth ===
                    self.depth_history.append(raw_depth)
                    smoothed_depth = float(np.median(self.depth_history))
                    
                    sector_width = dw // 5
                    sectors = []
                    for i in range(5):
                        s = depth_map[dh//3:2*dh//3, i*sector_width:(i+1)*sector_width]
                        v = s[(s > 0.3) & (s < 10)]
                        sectors.append(float(np.percentile(v, 30)) if len(v) > 0 else 5.0)
                    
                    self.latest_depth = depth_map
                    self.center_depth = smoothed_depth  # Use smoothed!
                    self.sector_depths = sectors
                
                # Floor coverage with smoothing
                if floor_mask is not None:
                    fh = floor_mask.shape[0]
                    raw_floor = float(np.mean(floor_mask[fh//2:, :]))
                    self.floor_history.append(raw_floor)
                    floor_cov = float(np.median(self.floor_history))
                    
                    # === HYSTERESIS for floor detection ===
                    if raw_floor < 0.15:
                        self.floor_low_count += 1
                    else:
                        self.floor_low_count = 0
                else:
                    floor_cov = 1.0
                    floor_mask = None
                
                # === YOLO OBSTACLE DETECTION (runs occasionally, cached) ===
                yolo_detections = self.yolo_detector.detect(frame)
                self.yolo_blocked_sectors = self.yolo_detector.get_blocked_sectors()
                
                # === NAVIGATION with target locking ===
                # Pass user heading for turn tracking
                rec_clock, guidance, sector_clearances = self.navigator.navigate(
                    floor_mask, self.center_depth, self.yolo_blocked_sectors,
                    user_heading=self.user_heading
                )
                
                # === OBSTACLE WARNING (only warn once per obstacle type) ===
                if yolo_detections and rec_clock != 12:
                    for det in yolo_detections:
                        obs_key = det['class_name']
                        if obs_key not in self.warned_obstacles:
                            # First time seeing this obstacle
                            guidance = f"{det['class_name'].title()} - {guidance}"
                            self.warned_obstacles.add(obs_key)
                            break
                
                # Clear obstacle cache when direction changes
                if rec_clock != self.last_recommended_clock:
                    self.warned_obstacles.clear()
                self.last_recommended_clock = rec_clock
                
                # Override if no floor detected
                floor_is_missing = self.floor_low_count >= self.floor_low_threshold
                if floor_is_missing:
                    guidance = "Stop. No floor"
                    rec_clock = 12
                
                with self.result_lock:
                    self.latest_floor = floor_mask
                    self.latest_frame = frame
                    self.floor_coverage = floor_cov
                    self.recommended_clock = rec_clock
                    self.guidance_text = guidance
                
                elapsed = (datetime.now() - start).total_seconds()
                self.process_times.append(elapsed)
                self.fps = 1.0 / np.mean(self.process_times)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"Worker error: {e}")
    
    def _create_visualization(self):
        """Create visualization from latest results."""
        with self.result_lock:
            frame = self.latest_frame
            depth = self.latest_depth
            floor = self.latest_floor
            guidance = self.guidance_text
        
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        h, w = frame.shape[:2]
        canvas = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        
        # Top-left: Camera
        canvas[0:h, 0:w] = frame
        cv2.putText(canvas, "Camera", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # Top-right: Depth
        if depth is not None:
            d_vis = np.clip(depth, 0, 5) / 5.0 * 255
            d_color = cv2.applyColorMap(d_vis.astype(np.uint8), cv2.COLORMAP_JET)
            d_color = cv2.resize(d_color, (w, h))
            canvas[0:h, w:2*w] = d_color
        cv2.putText(canvas, f"Depth: {self.center_depth:.1f}m", (w+10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # Bottom-left: Floor overlay with CENTER LINE and CLOCK SECTORS
        if floor is not None:
            overlay = frame.copy()
            floor_resized = cv2.resize((floor * 255).astype(np.uint8), (w, h))
            overlay[:,:,1] = np.where(floor_resized > 128, 
                                       np.minimum(255, frame[:,:,1] + 80), 
                                       frame[:,:,1])
            
            # Draw FIXED CLOCK using IMU heading offset
            center_x = w // 2
            center_y = h
            clock_labels = ['9', '10', '11', '12', '1', '2', '3']
            clock_hours = [9, 10, 11, 12, 1, 2, 3]
            
            import math
            total_spread = 120
            sector_angle = total_spread / 7
            
            # NO IMU OFFSET - 12 is always center (camera forward)
            start_angle = -60  # Fixed: 9 o'clock at left, 3 o'clock at right
            
            
            for i in range(7):
                angle_start = start_angle + i * sector_angle
                angle_end = start_angle + (i + 1) * sector_angle
                angle_mid = (angle_start + angle_end) / 2
                
                rad_start = math.radians(90 - angle_start)
                rad_end = math.radians(90 - angle_end)
                rad_mid = math.radians(90 - angle_mid)
                
                line_len = h
                x_start = int(center_x + line_len * math.cos(rad_start))
                x_end = int(center_x + line_len * math.cos(rad_end))
                y_start = int(center_y - line_len * math.sin(rad_start))
                y_end = int(center_y - line_len * math.sin(rad_end))
                
                # Only draw sectors that are visible in frame
                if -w < x_start < 2*w or -w < x_end < 2*w:
                    cv2.line(overlay, (center_x, center_y), (x_start, max(0, y_start)), (200, 200, 200), 1)
                    
                    # Highlight recommended sector
                    if clock_hours[i] == self.recommended_clock:
                        pts = np.array([
                            [center_x, center_y],
                            [x_start, max(0, y_start)],
                            [x_end, max(0, y_end)]
                        ], np.int32)
                        cv2.fillPoly(overlay, [pts], (0, 150, 0))
                    
                    # Draw label
                    label_x = int(center_x + (h//3) * math.cos(rad_mid))
                    label_y = int(center_y - (h//3) * math.sin(rad_mid))
                    if 0 < label_x < w and 0 < label_y < h:
                        cv2.putText(overlay, clock_labels[i], (label_x-10, label_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw USER'S CURRENT DIRECTION (always straight up = center of camera)
            # This is the "hour hand" that stays fixed at 12 position in frame
            cv2.arrowedLine(overlay, (center_x, center_y-10), (center_x, 30), 
                           (255, 255, 0), 4, tipLength=0.15)
            cv2.putText(overlay, "YOU", (center_x - 18, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Show what clock direction user is facing
            cv2.putText(overlay, f"Facing: {self.user_clock}", (5, h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Draw depth crosshair
            cv2.circle(overlay, (center_x, h//3), 10, (0, 255, 255), 2)
            cv2.line(overlay, (center_x-15, h//3), (center_x+15, h//3), (0, 255, 255), 2)
            cv2.line(overlay, (center_x, h//3-15), (center_x, h//3+15), (0, 255, 255), 2)
            
            canvas[h:2*h, 0:w] = overlay
        cv2.putText(canvas, f"Floor: {self.floor_coverage:.0%}", (10, h+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # Bottom-right: Guidance + IMU DEBUG
        bg = (0, 100, 0) if "clear" in guidance.lower() else (0, 0, 150)
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        panel[:] = bg
        cv2.putText(panel, guidance, (10, h//3), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        
        # IMU DEBUG INFO
        cv2.putText(panel, f"IMU Raw: {self.user_heading:.0f}deg", (10, h//2+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.putText(panel, f"Calib 12: {self.clock_converter.reference_heading or 0:.0f}deg", (10, h//2+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.putText(panel, f"Facing: {self.user_clock}", (10, h//2+50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        
        cv2.putText(panel, f"FPS: {self.fps:.1f} | Rec: {self.recommended_clock}", 
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
        canvas[h:2*h, w:2*w] = panel
        
        return canvas
    
    def _visualization_thread(self):
        """Display visualization."""
        cv2.namedWindow("Navigation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Navigation", 800, 600)
        
        while self.vis_running:
            canvas = self._create_visualization()
            cv2.imshow("Navigation", canvas)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    async def handle_client(self, websocket):
        """Handle client - FAST response, async processing."""
        logger.info(f"✅ Client connected")
        
        try:
            async for message in websocket:
                if isinstance(message, str):
                    continue
                
                try:
                    # Parse frame
                    header_len = struct.unpack('>I', message[:4])[0]
                    header = json.loads(message[4:4+header_len].decode('utf-8'))
                    w, h = header.get('w', 320), header.get('h', 240)
                    
                    # Extract IMU data and track user heading
                    heading = header.get('heading', 0.0)
                    pitch = header.get('pitch', 0.0)
                    roll = header.get('roll', 0.0)
                    
                    # Auto-calibrate on first heading (sets 12 o'clock reference)
                    if not self.clock_converter.heading_calibrated:
                        self.clock_converter.calibrate(heading)
                        logger.info(f"📍 Calibrated 12 o'clock = {heading:.1f}°")
                    
                    # Get user's current clock direction
                    clock_result = self.clock_converter.get_clock_direction(pitch, roll, heading)
                    self.user_heading = heading
                    self.user_clock = clock_result['clock_hour']
                    
                    image_data = message[4+header_len:]
                    if len(image_data) >= w * h:
                        frame_gray = np.frombuffer(image_data[:w*h], dtype=np.uint8).reshape((h, w))
                        frame_gray = cv2.rotate(frame_gray, cv2.ROTATE_90_CLOCKWISE)
                        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
                        
                        # Put frame in queue (non-blocking, drops old frames)
                        try:
                            self.frame_queue.put_nowait(frame_bgr)
                        except queue.Full:
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put_nowait(frame_bgr)
                            except:
                                pass
                        
                        # Check if should speak (cooldown logic)
                        with self.result_lock:
                            current_guidance = self.guidance_text
                            current_depth = self.center_depth
                        
                        now = datetime.now()
                        time_since_speak = (now - self.last_speak_time).total_seconds()
                        
                        should_speak = False
                        if current_guidance != self.last_spoken:
                            # Guidance changed - speak immediately
                            should_speak = True
                        elif "Stop" in current_guidance or "Turn" in current_guidance:
                            # Critical warnings - can repeat every 2 seconds
                            should_speak = time_since_speak >= 2.0
                        elif time_since_speak >= self.speak_cooldown_sec:
                            # Normal guidance - repeat after cooldown
                            should_speak = True
                        
                        response = {
                            'text': current_guidance,
                            'should_speak': should_speak,
                            'center_depth': current_depth
                        }
                        
                        if should_speak:
                            self.last_spoken = current_guidance
                            self.last_speak_time = now
                        
                        await websocket.send(json.dumps(response))
                
                except Exception as e:
                    logger.warning(f"Frame error: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("👋 Client disconnected")
    
    async def run(self):
        """Start server."""
        logger.info("\n" + "="*50)
        logger.info("🚀 FAST NAVIGATION SERVER (ASYNC)")
        logger.info("="*50)
        logger.info(f"Server: ws://{self.host}:{self.port}")
        logger.info("="*50 + "\n")
        
        self.vis_thread = threading.Thread(target=self._visualization_thread, daemon=True)
        self.vis_thread.start()
        
        async with websockets.serve(self.handle_client, self.host, self.port, max_size=10*1024*1024):
            logger.info("✅ Ready!")
            await asyncio.Future()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--outdoor', action='store_true', 
                        help='Enable outdoor mode (sidewalk only, no grass)')
    args = parser.parse_args()
    
    server = FastNavigationServer(port=args.port, outdoor_mode=args.outdoor)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
