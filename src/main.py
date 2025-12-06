"""
Usage:
    python main.py --video ../data/demo.mp4 --yolo yolov8n.pt --output output/out.mp4
    python main.py --camera 0 --yolo yolov8n.pt --output output/out.mp4
    python main.py --camera http://192.168.1.100:8080/video --yolo yolov8n.pt --mobile
demo video at ../data/demo.mp4 (or pass --video).
Use --camera for webcam (0) or IP camera URL.
Use --mobile flag to enable mobile companion app for audio/gyro.
"""
import os, sys, argparse, time, json
sys.path.append(os.path.dirname(__file__))
from detection import YoloDetector
from tracker import TrackerManager
from depth import DepthEstimator
from path_finder import PathFinder
from viz import draw_overlay
from tts import TTSWorker
from mobile_server import MobileServer
import cv2
import numpy as np
import math

def calculate_gap_width(obj1, obj2, frame_width, camera_fov=60):
    """
    Calculate the real-world width between two objects considering their depths and positions.
    
    Args:
        obj1, obj2: Objects with 'bbox' [x1,y1,x2,y2] and 'depth' in meters
        frame_width: Width of the frame in pixels
        camera_fov: Camera field of view in degrees (default 60°)
    
    Returns:
        gap_width: Width in meters, or None if gap can't be calculated
        gap_angle: Angle between objects in degrees
        is_perpendicular: True if objects are roughly perpendicular to user
    """
    # Get object centers and depths
    obj1_center_x = (obj1['bbox'][0] + obj1['bbox'][2]) / 2
    obj2_center_x = (obj2['bbox'][0] + obj2['bbox'][2]) / 2
    obj1_depth = obj1.get('depth', 5.0)
    obj2_depth = obj2.get('depth', 5.0)
    
    # Calculate horizontal angle between objects
    # Each pixel represents (FOV / frame_width) degrees
    degrees_per_pixel = camera_fov / frame_width
    pixel_separation = abs(obj2_center_x - obj1_center_x)
    angle_separation = pixel_separation * degrees_per_pixel
    
    # Convert to radians for calculation
    angle_rad = math.radians(angle_separation)
    
    # Case 1: Objects at similar depth (perpendicular - easy case)
    depth_diff = abs(obj1_depth - obj2_depth)
    avg_depth = (obj1_depth + obj2_depth) / 2
    
    is_perpendicular = depth_diff < (avg_depth * 0.2)  # Within 20% depth difference
    
    if is_perpendicular:
        # Simple calculation: gap = depth * tan(angle)
        # Using average depth since they're roughly at same distance
        gap_width = 2 * avg_depth * math.tan(angle_rad / 2)
    else:
        # Case 2: Objects at different depths (angled gap)
        # Use law of cosines to find actual gap distance
        # Gap forms a triangle with user: d1, d2, gap
        # gap² = d1² + d2² - 2*d1*d2*cos(angle)
        gap_width_squared = obj1_depth**2 + obj2_depth**2 - 2*obj1_depth*obj2_depth*math.cos(angle_rad)
        gap_width = math.sqrt(max(0, gap_width_squared))
        
        # Adjust for projection angle - gap might be angled toward/away from user
        # If one object is much closer, the passable width is reduced
        projection_factor = min(obj1_depth, obj2_depth) / max(obj1_depth, obj2_depth)
        gap_width *= projection_factor
    
    return gap_width, angle_separation, is_perpendicular

def find_passable_gaps(objs, frame_width, min_gap_width=0.6):
    """
    Find gaps between objects that are wide enough for a person to pass through.
    
    Args:
        objs: List of detected objects with bbox, depth, and position
        frame_width: Width of the frame in pixels
        min_gap_width: Minimum gap width in meters (default 0.6m for person width)
    
    Returns:
        List of passable gaps with properties: {left_obj, right_obj, width, angle, direction}
    """
    if len(objs) < 2:
        return []
    
    # Sort objects by horizontal position (left to right)
    sorted_objs = sorted(objs, key=lambda o: (o['bbox'][0] + o['bbox'][2]) / 2)
    
    passable_gaps = []
    
    # Check consecutive objects for gaps
    for i in range(len(sorted_objs) - 1):
        left_obj = sorted_objs[i]
        right_obj = sorted_objs[i + 1]
        
        # Only consider gaps where both objects are within reasonable distance (< 4m)
        if left_obj.get('depth', 5.0) > 4.0 or right_obj.get('depth', 5.0) > 4.0:
            continue
        
        # Calculate gap width
        gap_width, gap_angle, is_perpendicular = calculate_gap_width(
            left_obj, right_obj, frame_width
        )
        
        # Check if gap is passable
        if gap_width >= min_gap_width:
            # Calculate gap center position (clock position)
            left_center_x = (left_obj['bbox'][0] + left_obj['bbox'][2]) / 2
            right_center_x = (right_obj['bbox'][0] + right_obj['bbox'][2]) / 2
            gap_center_x = (left_center_x + right_center_x) / 2
            position_ratio = gap_center_x / frame_width
            
            # Convert to clock position
            if position_ratio < 0.08:
                clock_pos = "9 o'clock"
            elif position_ratio < 0.25:
                clock_pos = "10 o'clock"
            elif position_ratio < 0.42:
                clock_pos = "11 o'clock"
            elif position_ratio < 0.58:
                clock_pos = "12 o'clock"
            elif position_ratio < 0.75:
                clock_pos = "1 o'clock"
            elif position_ratio < 0.92:
                clock_pos = "2 o'clock"
            else:
                clock_pos = "3 o'clock"
            
            # Calculate confidence based on gap width margin
            confidence = min(1.0, (gap_width - min_gap_width) / min_gap_width)
            
            passable_gaps.append({
                'left_obj': left_obj,
                'right_obj': right_obj,
                'width': gap_width,
                'angle': gap_angle,
                'is_perpendicular': is_perpendicular,
                'direction': clock_pos,
                'confidence': confidence,
                'avg_depth': (left_obj.get('depth', 5.0) + right_obj.get('depth', 5.0)) / 2
            })
    
    # Sort by confidence (wider gaps first) and closeness
    passable_gaps.sort(key=lambda g: (g['confidence'], -g['avg_depth']), reverse=True)
    
    return passable_gaps

def get_relative_direction(current_heading, target_direction):
    """Calculate relative direction from current heading to target.
    Returns 'straight', 'slight left', 'left', 'hard left', 'slight right', 'right', 'hard right'"""
    # Convert clock positions to degrees (12=0, 3=90, 6=180, 9=270)
    clock_to_deg = {12: 0, 1: 30, 2: 60, 3: 90, 4: 120, 5: 150, 
                    6: 180, 7: 210, 8: 240, 9: 270, 10: 300, 11: 330}
    
    current_deg = clock_to_deg.get(current_heading, 0)
    target_deg = clock_to_deg.get(target_direction, 0)
    
    # Calculate shortest angular difference
    diff = (target_deg - current_deg) % 360
    if diff > 180:
        diff -= 360
    
    # Categorize the turn
    if abs(diff) < 15:
        return "straight"
    elif -45 < diff < -15:
        return "slight left"
    elif -90 < diff <= -45:
        return "left"
    elif diff <= -90:
        return "hard left"
    elif 15 < diff < 45:
        return "slight right"
    elif 45 <= diff < 90:
        return "right"
    else:
        return "hard right"

def process(video_source, yolo_weights, output_path, imgsz=320, skip_depth=10, no_show=False, use_mobile=False, process_every=1):
    det = YoloDetector(weights=yolo_weights, imgsz=imgsz, conf=0.20)  # Balanced confidence - not too low to avoid false positives
    tracker = TrackerManager(iou_threshold=0.25, max_age=20)  # Improved tracking parameters
    depth = DepthEstimator(device='cpu')  # Depth for better risk calculation
    path_finder = PathFinder()  # Floor segmentation and path finding
    tts = TTSWorker() if not use_mobile else None
    last_objs = []  # Store last processed objects for skipped frames
    last_depth_map = None  # Cache depth map
    last_audio_time = 0  # Track last audio warning time
    audio_cooldown = 8.0  # Seconds between audio warnings (increased to account for lag)
    last_audio_message = None  # Track last message to avoid repetition
    message_variety = 0  # Counter for message variation
    no_objects_count = 0  # Track consecutive frames with no objects
    last_boundary_state = False  # Track if we were at boundary last frame
    boundary_warning_count = 0  # Count warnings given while at boundary
    last_safe_direction = None  # Track last safe direction to avoid flip-flopping
    current_heading = 12  # Track user's current heading (assume starting at 12 o'clock)
    
    # Turn detection variables - track if user is completing instructed turn
    instructed_direction = None  # The clock position we told user to turn to
    waiting_for_turn = False  # True if we're waiting for user to complete turn
    obstacle_position_history = []  # Track obstacle clock positions over time
    turn_detection_frames = 0  # Count frames since instruction
    turn_confirmed = False  # True when we detect user has turned
    
    # Start mobile server if requested
    mobile_server = None
    if use_mobile:
        mobile_server = MobileServer(host='0.0.0.0', port=8765)
        mobile_server.start()
        import socket
        local_ip = socket.gethostbyname(socket.gethostname())
        print(f"\n*** Mobile Server Started ***")
        print(f"Connect your phone to: http://{local_ip}:8000/mobile.html")
        print(f"WebSocket: ws://{local_ip}:8765")
        print(f"Waiting for mobile connection...\n")
        time.sleep(2)
    
    # Open video source (file, camera index, or IP camera URL)
    if isinstance(video_source, int) or video_source.isdigit():
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_source}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    frame_idx = 0
    logf = open(os.path.join(os.path.dirname(output_path) or '.', 'output_log.jsonl'), 'w')
    ego = (w/2, h)
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1; t0 = time.time()
            
            # Process every Nth frame for speed
            if frame_idx % process_every == 0:  # Fixed: should be == 0, not == 1
                dets = det.detect(frame)
                bboxes = [d[:4] for d in dets]
                
                # Compute depth map every skip_depth frames for performance
                if frame_idx % skip_depth == 0 or last_depth_map is None:
                    # Get full depth map
                    last_depth_map = depth.estimate(frame, [])
                
                tracks = tracker.update(dets, frame_idx)
                objs = []
                for i, tr in enumerate(tracks):
                    tb = tr['bbox']; vx, vy = tr['velocity']
                    
                    # Extract depth for this bbox from depth map
                    x1, y1, x2, y2 = int(tb[0]), int(tb[1]), int(tb[2]), int(tb[3])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if last_depth_map is not None and x2 > x1 and y2 > y1:
                        bbox_depth = last_depth_map[y1:y2, x1:x2]
                        if bbox_depth.size > 0:
                            # MiDaS outputs relative disparity - normalize to frame
                            # Higher values = closer, use percentile-based normalization
                            disparity = float(np.median(bbox_depth))
                            frame_max = float(np.percentile(last_depth_map, 95))  # Top 5% as "very close"
                            frame_min = float(np.percentile(last_depth_map, 5))   # Bottom 5% as "far"
                            
                            if frame_max > frame_min:
                                # Normalize to 0-1 (1=close, 0=far)
                                norm_depth = (disparity - frame_min) / (frame_max - frame_min)
                                norm_depth = np.clip(norm_depth, 0, 1)
                                # Convert to meters: 1.0 (very close) -> 0.5m, 0.0 (far) -> 5m
                                depth_val = 5.0 - (norm_depth * 4.5)  # Range: 0.5m to 5m
                            else:
                                depth_val = 2.5
                        else:
                            depth_val = 2.5
                    else:
                        depth_val = 2.5
                    
                    # Simplified risk calculation focusing on proximity
                    area = (tb[2]-tb[0]) * (tb[3]-tb[1]) / (w*h)
                    bottom_y = tb[3] / h
                    
                    # Depth-based risk: exponential inverse for close objects
                    if depth_val < 1.0:
                        depth_risk = 1.0
                    elif depth_val < 2.0:
                        depth_risk = 0.9
                    elif depth_val < 3.0:
                        depth_risk = 0.6
                    else:
                        depth_risk = 0.3
                    
                    # Size risk: larger objects in frame = closer/more important
                    size_risk = min(1.0, area * 10.0)
                    
                    # Position risk: lower in frame = closer
                    position_risk = bottom_y
                    
                    # Combined risk: heavily weight depth and size
                    risk = 0.65 * depth_risk + 0.25 * size_risk + 0.10 * position_risk
                    
                    objs.append({
                        'id': int(tr['id']),
                        'bbox': [float(x) for x in tb],
                        'risk': float(risk),
                        'depth': float(depth_val),
                        'class': -1,
                        'polygon': None,
                        'ttc': None
                    })
                last_objs = objs  # Store for skipped frames
            else:
                # Use last processed objects for skipped frames
                objs = last_objs
            
            # Path guidance using floor segmentation and path finding
            safe_direction = None
            boundary_warning = False
            floor_mask = None
            path_scores = (0, 0, 0)
            
            if last_depth_map is not None:
                # Get intelligent path guidance from PathFinder
                safe_direction, boundary_warning, floor_mask, path_msg, path_scores = \
                    path_finder.get_path_guidance(frame, last_depth_map, objs, clock_format=False)
                
                # Track consecutive frames with no objects detected
                if not objs:
                    no_objects_count += 1
                    # Only trigger wall warning if PathFinder also detects boundary
                    # Don't force warning just because no objects detected
                    # (empty path is good, not bad!)
                    
                    # Additional check: if no objects for extended period AND depth shows close surface
                    if no_objects_count >= 60 and not boundary_warning:  # 2 seconds of no objects
                        # Check if we're facing a wall using depth
                        if last_depth_map is not None:
                            # Sample center region of depth map
                            h_depth, w_depth = last_depth_map.shape
                            center_region = last_depth_map[h_depth//3:2*h_depth//3, w_depth//3:2*w_depth//3]
                            
                            if center_region.size > 0:
                                # Check if depth indicates close surface (high disparity values)
                                center_disparity = float(np.median(center_region))
                                max_disparity = float(np.percentile(last_depth_map, 95))
                                
                                # If center has high disparity (close surface), likely a wall
                                if center_disparity > (max_disparity * 0.7):
                                    boundary_warning = True
                                    path_msg = "STOP! Wall ahead. Turn around."
                                    print(f"[WALL DETECTED] No objects + high center disparity = wall")
                else:
                    no_objects_count = 0  # Reset counter when objects detected
                    boundary_warning_count = 0  # Reset boundary warning count when clear
                
                # Debug: show wall detection when no objects detected
                if not objs and frame_idx % 30 == 0:
                    is_wall, wall_dist = path_finder.detect_wall_ahead(frame, last_depth_map)
                    print(f"[DEBUG] Frame {frame_idx}: objs=0, wall_detected={is_wall}, wall_dist={wall_dist:.2f}m, boundary={boundary_warning}, no_obj_count={no_objects_count}")
                
                # If at boundary/wall (critical warning), announce on state change or periodically
                if boundary_warning:
                    current_time = time.time()
                    # Only warn on first detection or every 90 frames (~6 seconds) after that
                    just_entered_boundary = not last_boundary_state
                    periodic_reminder = (boundary_warning_count > 0 and no_objects_count % 90 == 0)
                    
                    if (just_entered_boundary or periodic_reminder) and current_time - last_audio_time >= audio_cooldown:
                        boundary_warning_count += 1
                        print(f"[AUDIO] {path_msg} (wall/boundary #{boundary_warning_count})")
                        if use_mobile and mobile_server:
                            mobile_server.send_audio_command(path_msg, urgency=1.0)
                        elif tts:
                            tts.speak(path_msg, urgency=1.0, clear_queue=True)  # Clear queue for critical boundary warning
                        last_audio_time = current_time
                        last_audio_message = path_msg
                    # Skip normal object warnings when at wall/boundary
                    continue
                
                # Update boundary state for next frame
                last_boundary_state = boundary_warning
            
            # Find passable gaps between obstacles
            passable_gaps = find_passable_gaps(objs, w, min_gap_width=0.6)
            
            # TTS for highest risk obstacles (only if not at boundary)
            if objs and not boundary_warning:
                top = max(objs, key=lambda o: o['risk'])
                # Only warn for significant risks
                if top['risk'] > 0.35:
                    obj_center_x = (top['bbox'][0]+top['bbox'][2])/2
                    depth_val = top.get('depth', 5.0)
                    
                    # Convert position to clock hours (12 = ahead, 3 = right, 9 = left)
                    # Divide frame width into 12 positions
                    position_ratio = obj_center_x / w
                    
                    if position_ratio < 0.08:  # Far left
                        clock_pos = "9 o'clock"
                    elif position_ratio < 0.25:  # Left
                        clock_pos = "10 o'clock"
                    elif position_ratio < 0.42:  # Center-left
                        clock_pos = "11 o'clock"
                    elif position_ratio < 0.58:  # Center
                        clock_pos = "12 o'clock"
                    elif position_ratio < 0.75:  # Center-right
                        clock_pos = "1 o'clock"
                    elif position_ratio < 0.92:  # Right
                        clock_pos = "2 o'clock"
                    else:  # Far right
                        clock_pos = "3 o'clock"
                    
                    # Track obstacle position history for turn detection
                    current_clock_num = int(clock_pos.split()[0])
                    obstacle_position_history.append(current_clock_num)
                    if len(obstacle_position_history) > 30:  # Keep last 30 frames (~1 second)
                        obstacle_position_history.pop(0)
                    
                    # Detect if user has turned (obstacle moved significantly in frame)
                    if waiting_for_turn and instructed_direction and len(obstacle_position_history) >= 15 and turn_detection_frames >= 20:
                        # Require minimum 20 frames before detecting (prevents false positives)
                        # Require more frames for stable detection
                        recent_positions = obstacle_position_history[-15:]
                        avg_recent_pos = sum(recent_positions) / len(recent_positions)
                        
                        instructed_clock_num = int(instructed_direction.split()[0])
                        
                        # Calculate if obstacle actually moved in expected direction
                        # If told to turn to 3 (right), obstacle should shift left (toward 9-12)
                        # If told to turn to 9 (left), obstacle should shift right (toward 12-3)
                        initial_pos = obstacle_position_history[0] if obstacle_position_history else current_clock_num
                        position_change = current_clock_num - initial_pos
                        
                        # Detect turn completion with stricter criteria:
                        # 1. Obstacle clearly moved away (position changed by 2+ clock positions)
                        # 2. AND obstacle is now in safer zone (11-1 o'clock OR risk significantly reduced)
                        # 3. AND position stable for last 10 frames
                        recent_stable = max(recent_positions[-10:]) - min(recent_positions[-10:]) <= 1
                        obstacle_moved_away = abs(position_change) >= 2
                        obstacle_now_centered = (11 <= current_clock_num <= 1 or current_clock_num == 12)
                        risk_significantly_reduced = top['risk'] < 0.4  # More strict threshold
                        
                        if recent_stable and (obstacle_moved_away or (obstacle_now_centered and risk_significantly_reduced)):
                            turn_confirmed = True
                            waiting_for_turn = False
                            turn_detection_frames = 0
                            print(f"[TURN DETECTED] User turned to {instructed_direction}, obstacle moved from {initial_pos} to {clock_pos} (risk={top['risk']:.2f})")
                    
                    # Increment turn detection counter
                    if waiting_for_turn:
                        turn_detection_frames += 1
                        # Don't accept turns too quickly (give user time to actually turn)
                        # Minimum 20 frames (~0.7 seconds) before detecting turn
                        # Auto-reset after 150 frames (~5 seconds) if no turn detected
                        if turn_detection_frames > 150:
                            print(f"[TURN TIMEOUT] Resetting turn detection after {turn_detection_frames} frames")
                            waiting_for_turn = False
                            turn_detection_frames = 0
                            obstacle_position_history.clear()  # Clear history for fresh start
                    
                    # Determine safe path using clock positions (from PathFinder)
                    safe_clock = None
                    gap_guidance = None
                    
                    # First, check if there's a passable gap we can guide user through
                    if passable_gaps:
                        best_gap = passable_gaps[0]  # Highest confidence gap
                        # Only use gap if it's not where the obstacle is
                        gap_clock_num = int(best_gap['direction'].split()[0])
                        if gap_clock_num != current_clock_num:
                            safe_clock = best_gap['direction']
                            # Create informative gap guidance message
                            gap_width_cm = int(best_gap['width'] * 100)
                            if best_gap['is_perpendicular']:
                                gap_guidance = f"Gap {gap_width_cm} cm at {safe_clock}"
                            else:
                                gap_guidance = f"Angled gap at {safe_clock}"
                    
                    # If no gap found, use PathFinder's safe direction
                    if not safe_clock and safe_direction:
                        if safe_direction == 'left':
                            safe_clock = "9 o'clock"
                        elif safe_direction == 'right':
                            safe_clock = "3 o'clock"
                        elif safe_direction == 'ahead':
                            safe_clock = "12 o'clock"
                        # else: turn_around - will be handled by boundary warning
                    
                    # Check if obstacle is dangerously close (depth < 1m or risk > 0.9)
                    is_critical = ('depth' in top and top['depth'] < 1.0) or top['risk'] > 0.9
                    
                    if is_critical:
                        # STOP command for critical proximity
                        steps = max(1, int(depth_val * 1.3))  # Approximate: 1 meter ≈ 1.3 steps
                        step_word = "step" if steps == 1 else "steps"
                        
                        if safe_clock and safe_clock != clock_pos:
                            if gap_guidance:
                                msg = f"STOP! {steps} {step_word} to obstacle at {clock_pos}. {gap_guidance}."
                            else:
                                msg = f"STOP! {steps} {step_word} to obstacle at {clock_pos}. Go {safe_clock}."
                        else:
                            msg = f"STOP! {steps} {step_word} ahead. Obstacle at {clock_pos}."
                        urgency = 1.0
                    else:
                        # For non-critical obstacles, give simple directional guidance
                        # Only warn for obstacles within 3 meters
                        if 'depth' in top and top['depth'] >= 3.0:
                            continue  # Skip distant obstacles
                        
                        steps = max(1, int(top['depth'] * 1.3))
                        step_word = "step" if steps == 1 else "steps"
                        
                        # Use clock positions only - simple and consistent
                        if safe_clock and safe_clock != clock_pos:
                            # Safe path exists - guide to it
                            if gap_guidance:
                                msg = f"Obstacle at {clock_pos}. {gap_guidance}."
                            else:
                                msg = f"Obstacle at {clock_pos}. Turn to {safe_clock}."
                            urgency = 0.7
                        else:
                            # No safe direction found - choose escape based on obstacle position
                            clock_num = int(clock_pos.split()[0])
                            
                            # Check if there's a gap we should prefer
                            if passable_gaps and not gap_guidance:
                                # Find best gap that's not blocked by current obstacle
                                for gap in passable_gaps:
                                    gap_clock_num = int(gap['direction'].split()[0])
                                    if gap_clock_num != clock_num:
                                        safe_clock = gap['direction']
                                        gap_width_cm = int(gap['width'] * 100)
                                        gap_guidance = f"Gap {gap_width_cm} cm at {safe_clock}"
                                        break
                            
                            # If still no gap, use direction consistency logic
                            if not safe_clock:
                                # Determine escape direction with strong consistency
                                # Stay consistent with previous direction to avoid flip-flopping
                                if current_heading != 12:
                                    # User was already told to turn somewhere - STRONGLY prefer keeping that direction
                                    # Only change if obstacle is DIRECTLY blocking that path
                                    heading_blocked = (current_heading == clock_num)
                                    
                                    if not heading_blocked:
                                        # Previous direction still valid - maintain it
                                        safe_clock = f"{current_heading} o'clock"
                                    else:
                                        # Obstacle directly blocking - choose most similar direction
                                        # If was going to 3, try 2 or 1 before switching to 9
                                        # If was going to 9, try 10 or 11 before switching to 3
                                        if current_heading >= 9:
                                            safe_clock = "10 o'clock"  # Slight adjustment
                                        elif current_heading <= 3:
                                            safe_clock = "2 o'clock"  # Slight adjustment
                                        else:
                                            # Middle positions - choose closest cardinal
                                            safe_clock = "9 o'clock" if current_heading > 6 else "3 o'clock"
                                else:
                                    # First time guidance - choose based on obstacle position
                                    if clock_num <= 6:
                                        # Obstacle on right side, suggest left (9)
                                        safe_clock = "9 o'clock"
                                    else:
                                        # Obstacle on left side, suggest right (3)
                                        safe_clock = "3 o'clock"
                            
                            # Build message with gap info if available
                            if gap_guidance:
                                msg = f"Obstacle {steps} {step_word} at {clock_pos}. {gap_guidance}."
                            else:
                                msg = f"Obstacle {steps} {step_word} at {clock_pos}. Turn to {safe_clock}."
                            urgency = 0.8
                    
                    # Provide positive feedback if turn was confirmed (but not too frequently)
                    current_time = time.time()
                    if turn_confirmed and (current_time - last_audio_time >= 3.0):
                        # Only give positive feedback if it's been at least 3 seconds since last audio
                        confirmation_msg = "Good, continue."
                        print(f"[AUDIO] {confirmation_msg}")
                        if use_mobile and mobile_server:
                            mobile_server.send_audio_command(confirmation_msg, urgency=0.3)
                        elif tts:
                            tts.speak(confirmation_msg, urgency=0.3)
                        turn_confirmed = False
                        last_audio_time = current_time
                    elif turn_confirmed:
                        # Turn detected but too soon after last audio - just clear flag
                        turn_confirmed = False
                        print(f"[TURN DETECTED] Confirmed but skipping audio (too soon)")
                    
                    # Send audio guidance (with lag-aware timing and direction stability)
                    current_time = time.time()
                    # Check if direction has changed significantly
                    direction_changed = (last_safe_direction != safe_clock) if safe_clock else False
                    time_ok = current_time - last_audio_time >= audio_cooldown
                    long_silence = current_time - last_audio_time >= 15.0  # Force update every 15 sec
                    
                    # Skip new guidance if waiting for user to complete turn (unless critical)
                    can_give_new_direction = not waiting_for_turn or is_critical
                    
                    if time_ok and (direction_changed or is_critical or long_silence) and can_give_new_direction:
                        print(f"[AUDIO] {msg} (urgency={urgency:.2f}, dir_changed={direction_changed}, critical={is_critical})")
                        if use_mobile and mobile_server:
                            mobile_server.send_audio_command(msg, urgency=urgency)
                        elif tts:
                            # Clear queue for critical messages to ensure immediate delivery
                            tts.speak(msg, urgency=urgency, clear_queue=is_critical)
                        last_audio_time = current_time
                        last_audio_message = msg
                        # Update heading and last safe direction after speaking
                        if safe_clock:
                            # Extract just the clock number to track heading
                            current_heading = int(safe_clock.split()[0])
                            last_safe_direction = safe_clock
                            # Mark that we're waiting for user to turn
                            if safe_clock != "12 o'clock":  # Only wait if actual turn needed
                                instructed_direction = safe_clock
                                waiting_for_turn = True
                                turn_detection_frames = 0
                                print(f"[TURN INSTRUCTION] Waiting for user to turn to {instructed_direction}")
                    elif not time_ok:
                        # Debug: show why audio was skipped
                        time_remaining = audio_cooldown - (current_time - last_audio_time)
                        if frame_idx % 60 == 0:  # Reduced debug frequency
                            print(f"[AUDIO SKIP] Cooldown active ({time_remaining:.1f}s remaining)")
                    elif waiting_for_turn and not is_critical:
                        # Debug: show we're waiting for turn
                        if frame_idx % 60 == 0:
                            print(f"[WAITING] User turning to {instructed_direction} (frame {turn_detection_frames})")
            
            # Visualization with floor mask overlay and gap indicators
            vis = draw_overlay(frame, objs, passable_gaps)
            
            # Overlay floor mask if available (show walkable path in green tint)
            if floor_mask is not None:
                # Create green overlay for walkable floor
                floor_overlay = np.zeros_like(vis)
                floor_overlay[:, :, 1] = floor_mask  # Green channel
                vis = cv2.addWeighted(vis, 1.0, floor_overlay, 0.2, 0)
                
                # Draw path scores as text
                left_s, center_s, right_s = path_scores
                cv2.putText(vis, f"L:{left_s:.2f} C:{center_s:.2f} R:{right_s:.2f}", 
                           (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if boundary_warning:
                    cv2.putText(vis, "BOUNDARY WARNING!", 
                               (w//2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            writer.write(vis)
            if not no_show:
                cv2.imshow('AssistedVision', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            # Log gyro data if available
            log_entry = {'frame': frame_idx, 'objects': objs}
            if use_mobile and mobile_server and mobile_server.is_connected():
                gyro = mobile_server.get_gyro_average()
                log_entry['gyro'] = gyro
            logf.write(json.dumps(log_entry) + '\n')
            
            if frame_idx % 30 == 0:
                status = f"Frame {frame_idx} processed in {time.time()-t0:.3f}s, objs={len(objs)}"
                if use_mobile and mobile_server:
                    status += f", mobile={'connected' if mobile_server.is_connected() else 'disconnected'}"
                print(status)
    finally:
        cap.release(); writer.release(); logf.close()
        if tts:
            tts.stop()
        if mobile_server:
            mobile_server.stop()
        cv2.destroyAllWindows()
        print('Finished. Output at', output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', help='Path to video file')
    group.add_argument('--camera', help='Camera index (0) or IP camera URL (http://...)')
    parser.add_argument('--yolo', default='yolov8n.pt')
    parser.add_argument('--output', default='output/out.mp4')
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--skip_depth', type=int, default=3)
    parser.add_argument('--process_every', type=int, default=1, help='Process every Nth frame (1=all, 2=every 2nd, 3=every 3rd, etc.)')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--mobile', action='store_true', help='Enable mobile companion app for audio/gyro')
    args = parser.parse_args()
    
    video_source = args.video if args.video else args.camera
    process(video_source, args.yolo, args.output, imgsz=args.imgsz, skip_depth=args.skip_depth, no_show=args.no_show, use_mobile=args.mobile, process_every=args.process_every)
