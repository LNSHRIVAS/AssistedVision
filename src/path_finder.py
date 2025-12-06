import cv2
import numpy as np

class PathFinder:
    """
    Detects walkable floor/ground areas and finds safe paths for navigation.
    Uses color-based segmentation and depth information to identify walkable surfaces.
    """
    def __init__(self):
        self.last_floor_mask = None
        self.floor_color_range = None
        
    def detect_floor(self, frame, depth_map=None):
        """
        Detect floor/walkable surface using color segmentation and depth.
        Focus on the bottom half of the frame where the floor typically appears.
        """
        h, w = frame.shape[:2]
        
        # Focus on bottom 60% of frame (where floor is visible)
        floor_region_start = int(h * 0.4)
        floor_region = frame[floor_region_start:, :]
        
        # Convert to HSV for color-based segmentation
        hsv = cv2.cvtColor(floor_region, cv2.COLOR_BGR2HSV)
        
        # Sample the bottom 10% center region as likely floor
        sample_region = frame[int(h * 0.9):, int(w * 0.3):int(w * 0.7)]
        
        if sample_region.size > 100:  # Ensure we have enough pixels
            # Get dominant color in sample region (likely floor)
            sample_hsv = cv2.cvtColor(sample_region, cv2.COLOR_BGR2HSV)
            h_mean = float(np.mean(sample_hsv[:, :, 0]))
            s_mean = float(np.mean(sample_hsv[:, :, 1]))
            v_mean = float(np.mean(sample_hsv[:, :, 2]))
            
            # Create adaptive range around dominant color (wider ranges for robustness)
            h_range = 40  # Wider hue range
            s_range = 100  # Wider saturation range
            v_range = 100  # Wider value range
            
            lower = np.array([max(0, int(h_mean - h_range)), 
                            max(0, int(s_mean - s_range)), 
                            max(20, int(v_mean - v_range))], dtype=np.uint8)
            upper = np.array([min(179, int(h_mean + h_range)), 
                            min(255, int(s_mean + s_range)), 
                            min(255, int(v_mean + v_range))], dtype=np.uint8)
            
            # Create floor mask
            floor_mask = cv2.inRange(hsv, lower, upper)
        else:
            # Fallback: detect common indoor floor colors
            # Wood/tile/carpet typically have moderate V and low-moderate S
            mask1 = cv2.inRange(hsv, np.array([0, 0, 30], dtype=np.uint8), 
                                     np.array([179, 100, 200], dtype=np.uint8))
            floor_mask = mask1
        
        # Morphological operations to clean up mask
        kernel = np.ones((7, 7), np.uint8)
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill holes in floor mask
        kernel_large = np.ones((15, 15), np.uint8)
        floor_mask = cv2.dilate(floor_mask, kernel_large, iterations=2)
        
        # Use depth to refine floor detection (floor should be at consistent depth in bottom region)
        if depth_map is not None and depth_map.size > 0:
            try:
                depth_region = depth_map[floor_region_start:, :]
                # Get depth statistics for bottom region
                valid_depths = depth_region[depth_region > 0]
                if valid_depths.size > 100:
                    # Floor typically has higher disparity values (closer in MiDaS output)
                    depth_thresh = np.percentile(valid_depths, 60)
                    # Keep regions with depth similar to floor depth
                    depth_mask = ((depth_region > depth_thresh * 0.7) & 
                                (depth_region < depth_thresh * 1.5)).astype(np.uint8) * 255
                    floor_mask = cv2.bitwise_and(floor_mask, depth_mask)
            except:
                pass  # Skip depth refinement if it fails
        
        # Create full-frame mask
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[floor_region_start:, :] = floor_mask
        
        self.last_floor_mask = full_mask
        return full_mask
    
    def find_walkable_zones(self, frame, floor_mask, obstacles):
        """
        Divide the walkable floor into zones and find the safest path.
        Returns left, center, right zone scores and recommended direction.
        """
        h, w = frame.shape[:2]
        
        # Create obstacle mask
        obstacle_mask = np.zeros((h, w), dtype=np.uint8)
        for obj in obstacles:
            bbox = obj['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            obstacle_mask[y1:y2, x1:x2] = 255
        
        # Erode obstacle regions to create safety margin
        kernel = np.ones((15, 15), np.uint8)
        obstacle_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)
        
        # Subtract obstacles from floor to get clear walkable area
        clear_floor = cv2.bitwise_and(floor_mask, cv2.bitwise_not(obstacle_mask))
        
        # Divide into 3 vertical zones
        zone_width = w // 3
        left_zone = clear_floor[:, :zone_width]
        center_zone = clear_floor[:, zone_width:2*zone_width]
        right_zone = clear_floor[:, 2*zone_width:]
        
        # Score each zone based on walkable area (focus on bottom half where walking happens)
        bottom_start = int(h * 0.5)
        left_score = np.sum(left_zone[bottom_start:]) / (zone_width * (h - bottom_start) * 255)
        center_score = np.sum(center_zone[bottom_start:]) / (zone_width * (h - bottom_start) * 255)
        right_score = np.sum(right_zone[bottom_start:]) / (zone_width * (h - bottom_start) * 255)
        
        # Prefer center if scores are similar (within 15%)
        if center_score > 0.3:  # Minimum threshold for center
            if center_score >= max(left_score, right_score) * 0.85:
                return left_score, center_score, right_score, 'ahead', clear_floor
        
        # Choose best zone
        if left_score > center_score and left_score > right_score:
            direction = 'left'
        elif right_score > center_score and right_score > left_score:
            direction = 'right'
        else:
            direction = 'ahead'
        
        return left_score, center_score, right_score, direction, clear_floor
    
    def detect_wall_ahead(self, frame, depth_map):
        """
        Detect if there's a wall or large obstacle directly ahead using depth analysis.
        Uses multiple strategies for robust wall detection.
        Returns (is_wall_ahead, distance_estimate)
        """
        if depth_map is None or depth_map.size == 0:
            return False, 5.0
        
        h, w = depth_map.shape[:2]
        
        # Strategy 1: Analyze CENTER region ahead (aggressive, larger area)
        center_h_start = int(h*0.15)  # More vertical coverage
        center_h_end = int(h*0.7)
        center_w_start = int(w*0.25)  # Wider horizontal coverage
        center_w_end = int(w*0.75)
        center_region = depth_map[center_h_start:center_h_end, center_w_start:center_w_end]
        
        if center_region.size < 100:
            return False, 5.0
        
        # Get depth statistics
        valid_depths = center_region[center_region > 0]
        if valid_depths.size < 50:
            return False, 5.0
        
        depth_mean = float(np.mean(valid_depths))
        depth_std = float(np.std(valid_depths))
        depth_median = float(np.median(valid_depths))
        depth_90th = float(np.percentile(valid_depths, 90))
        
        # Frame statistics for comparison
        frame_75th = float(np.percentile(depth_map.flatten(), 75))
        frame_median = float(np.median(depth_map.flatten()))
        
        # Strategy 1: High consistent depth = flat wall
        flat_wall = (depth_std < depth_mean * 0.4 and depth_90th > frame_75th)
        
        # Strategy 2: Center region significantly higher than frame average = obstacle/wall blocking
        center_high = (depth_median > frame_median * 1.3)
        
        # Strategy 3: Very high depth concentration in center = close wall
        very_close = (depth_mean > frame_75th and depth_90th > np.percentile(depth_map.flatten(), 85))
        
        # Detect wall if ANY strategy triggers
        is_wall = flat_wall or center_high or very_close
        
        # Estimate distance using frame-relative scaling
        frame_max = float(np.percentile(depth_map, 95))
        frame_min = float(np.percentile(depth_map, 5))
        
        if frame_max > frame_min:
            # Use 90th percentile for more conservative distance estimate
            norm_depth = (depth_90th - frame_min) / (frame_max - frame_min)
            distance = 5.0 - (norm_depth * 4.5)  # Convert to meters (0.5-5m)
            distance = max(0.5, min(5.0, distance))  # Clamp to valid range
        else:
            distance = 2.5
        
        return is_wall, distance
    
    def detect_boundary_crossing(self, frame, floor_mask):
        """
        Detect if user is approaching indoor/outdoor boundary or edge of walkable area.
        Returns True if approaching boundary (e.g., doorway to outside).
        """
        h, w = frame.shape[:2]
        
        # Check bottom center region (where user is walking)
        bottom_region = floor_mask[int(h*0.7):, int(w*0.3):int(w*0.7)]
        
        if bottom_region.size == 0:
            return False  # Can't determine, assume safe
        
        floor_coverage = np.sum(bottom_region) / (bottom_region.size * 255)
        
        # If floor coverage drops below 25%, likely approaching boundary
        # Made less sensitive (was 40%)
        if floor_coverage < 0.25:
            return True
        
        # Check for sudden changes in floor appearance (indoor vs outdoor)
        # Outdoor areas typically have more varied/bright colors
        bottom_frame = frame[int(h*0.7):, int(w*0.3):int(w*0.7)]
        if bottom_frame.size > 100:
            try:
                hsv = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2HSV)
                # High saturation + high value = likely outdoor (grass, bright concrete, etc.)
                s_mean = float(np.mean(hsv[:, :, 1]))
                v_mean = float(np.mean(hsv[:, :, 2]))
                
                # More conservative outdoor detection heuristic
                if s_mean > 120 and v_mean > 180:
                    return True  # Likely bright outdoor area
            except:
                pass
        
        return False
    
    def get_path_guidance(self, frame, depth_map, obstacles, clock_format=True):
        """
        Main function to get path guidance.
        Returns: (safe_direction, boundary_warning, floor_mask, message)
        """
        # Detect floor
        floor_mask = self.detect_floor(frame, depth_map)
        
        # Check for wall ahead (even if no objects detected)
        is_wall_ahead, wall_distance = self.detect_wall_ahead(frame, depth_map)
        
        # Check for boundary crossing
        at_boundary = self.detect_boundary_crossing(frame, floor_mask)
        
        # Find walkable zones
        left_score, center_score, right_score, direction, clear_floor = \
            self.find_walkable_zones(frame, floor_mask, obstacles)
        
        # Generate message
        # Priority: wall ahead > boundary > normal navigation
        if is_wall_ahead and wall_distance < 4.0:  # Increased from 3.0 to warn even earlier
            # Wall detected close ahead - critical warning
            steps = max(1, int(wall_distance * 1.3))
            step_word = "step" if steps == 1 else "steps"
            message = f"STOP! Wall {steps} {step_word} ahead. Turn around."
            direction = 'turn_around'
            at_boundary = True  # Treat as boundary for priority handling
        elif at_boundary:
            message = "STOP! End of safe path. Turn around."
            direction = 'turn_around'
        else:
            # Convert to clock format if requested
            if clock_format:
                if direction == 'left':
                    clock_dir = '9 o\'clock'
                elif direction == 'right':
                    clock_dir = '3 o\'clock'
                else:
                    clock_dir = '12 o\'clock'
                
                # Add confidence info
                scores = [left_score, center_score, right_score]
                max_score = max(scores)
                if max_score < 0.3:
                    message = f"Limited path ahead. Walk carefully {clock_dir}."
                elif max_score > 0.6:
                    message = f"Clear path {clock_dir}."
                else:
                    message = f"Walk {clock_dir}."
            else:
                message = f"Walk {direction}."
        
        return direction, at_boundary, floor_mask, message, (left_score, center_score, right_score)
