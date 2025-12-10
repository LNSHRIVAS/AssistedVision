"""
Look-Ahead Navigator
====================
Scans for obstacles BEFORE reaching them and pre-computes escape routes.

Strategy:
1. SAFE ZONE (depth > 2m): 
   - "Clear ahead"
   - But continuously scan for obstacles
   - If obstacle detected far away, PRE-SCAN alternatives
   
2. WARNING ZONE (1.5m < depth < 2m):
   - Obstacle detected ahead
   - Scan left/right for best alternative
   - STORE the best escape direction
   - Still say "Clear ahead" (not dangerous yet)
   
3. DANGER ZONE (depth < 1.5m):
   - "Turn to X" (use pre-stored direction)
   - Track user's IMU heading
   - Wait for user to align
   
4. ALIGNED:
   - User turned to escape direction
   - Re-calibrate 12 o'clock
   - Back to SAFE ZONE scanning
"""

import numpy as np
import math
from typing import Tuple, List, Optional
from collections import deque


class LookAheadNavigator:
    """
    Look-ahead navigation with pre-scanning and stored escape routes.
    Only guides user when actually needed (danger zone).
    """
    
    CLOCKS = [9, 10, 11, 12, 1, 2, 3]
    
    # IMAGE-SPACE ray casting angles
    IMAGE_ANGLES = {
        9: 90, 10: 60, 11: 30, 12: 0, 1: -30, 2: -60, 3: -90
    }
    
    # REAL-WORLD heading offsets (IMU convention)
    HEADING_OFFSETS = {
        9: -90, 10: -60, 11: -30, 12: 0, 1: 30, 2: 60, 3: 90
    }
    
    # Thresholds
    SAFE_DEPTH = 2.0       # > 2m = safe, just walk
    WARNING_DEPTH = 1.5    # 1.5-2m = pre-scan alternatives
    DANGER_DEPTH = 1.0     # < 1m = must turn NOW
    MIN_CLEARANCE = 0.1    # Minimum floor clearance ratio
    ALIGN_THRESHOLD = 25   # Degrees to consider aligned
    
    def __init__(self):
        self.state = "SAFE"  # SAFE, WARNING, DANGER, TURNING
        
        # Current heading
        self.current_heading = 0
        self.calibrated_12 = 0
        
        # Pre-scanned escape route
        self.escape_clock = None     # Best alternative direction
        self.escape_heading = None   # Real-world heading for escape
        self.escape_clearance = 0    # How clear is the escape route
        
        # Smoothing
        self.clearance_history = deque(maxlen=5)
        self.depth_history = deque(maxlen=5)
    
    def _normalize_angle(self, angle: float) -> float:
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def _clock_to_heading(self, clock: int, base_heading: float) -> float:
        offset = self.HEADING_OFFSETS.get(clock, 0)
        return self._normalize_angle(base_heading + offset)
    
    def _heading_diff(self, h1: float, h2: float) -> float:
        return self._normalize_angle(h1 - h2)
    
    def _cast_ray(self, floor_mask: np.ndarray, angle_deg: float) -> float:
        h, w = floor_mask.shape
        start_x, start_y = w // 2, h - 1
        
        angle_rad = math.radians(angle_deg)
        dx = -math.sin(angle_rad)
        dy = -math.cos(angle_rad)
        
        step_size = 2
        max_steps = int(h / step_size)
        clearance_pixels = 0
        
        for i in range(max_steps):
            x = int(start_x + dx * i * step_size)
            y = int(start_y + dy * i * step_size)
            
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            
            if floor_mask[y, x]:
                clearance_pixels = i * step_size
            else:
                break
        
        return clearance_pixels / h
    
    def _get_sector_clearances(self, floor_mask: np.ndarray, blocked: set) -> List[float]:
        clearances = []
        for clock in self.CLOCKS:
            if clock in blocked:
                clearances.append(0.0)
            else:
                angles = [self.IMAGE_ANGLES[clock] - 8, 
                         self.IMAGE_ANGLES[clock], 
                         self.IMAGE_ANGLES[clock] + 8]
                rays = [self._cast_ray(floor_mask, a) for a in angles]
                clearances.append(min(rays))
        return clearances
    
    def _find_best_escape(self, clearances: List[float], blocked: set) -> Tuple[int, float]:
        """Find best escape direction (excluding center)."""
        best_clock = None
        best_clearance = 0
        
        for i, clock in enumerate(self.CLOCKS):
            if clock == 12 or clock in blocked:
                continue
            if clearances[i] > best_clearance:
                best_clearance = clearances[i]
                best_clock = clock
        
        return best_clock, best_clearance
    
    def calibrate(self, heading: float):
        self.calibrated_12 = heading
        self.current_heading = heading
        self.state = "SAFE"
        print(f"[LookAhead] Calibrated: 12 o'clock = {heading:.0f}°")
    
    def navigate(self, floor_mask: np.ndarray, 
                 center_depth: float = 5.0,
                 blocked_sectors: List[int] = None,
                 user_heading: float = None) -> Tuple[int, str, List[float]]:
        
        if floor_mask is None or floor_mask.size == 0:
            return 12, "Scanning...", [0.5] * 7
        
        if user_heading is not None:
            self.current_heading = user_heading
        
        blocked = set(blocked_sectors or [])
        
        # Smooth depth
        self.depth_history.append(center_depth)
        avg_depth = np.mean(self.depth_history)
        
        # Get clearances
        clearances = self._get_sector_clearances(floor_mask, blocked)
        self.clearance_history.append(clearances)
        if len(self.clearance_history) >= 3:
            avg_clearances = np.mean(list(self.clearance_history), axis=0).tolist()
        else:
            avg_clearances = clearances
        
        center_clearance = avg_clearances[3]
        
        # === LOOK-AHEAD STATE MACHINE ===
        
        if self.state == "SAFE":
            # Check if we're approaching an obstacle
            if avg_depth < self.WARNING_DEPTH:
                # Entering warning zone - start pre-scanning
                self.state = "WARNING"
                print(f"[LookAhead] Warning zone entered at {avg_depth:.1f}m")
            
            # Always scan for escape routes even in safe zone
            if avg_depth < self.SAFE_DEPTH:
                escape_clock, escape_clear = self._find_best_escape(avg_clearances, blocked)
                if escape_clock and escape_clear >= self.MIN_CLEARANCE:
                    self.escape_clock = escape_clock
                    self.escape_heading = self._clock_to_heading(escape_clock, self.current_heading)
                    self.escape_clearance = escape_clear
            
            return 12, "Clear ahead", avg_clearances
        
        elif self.state == "WARNING":
            # Pre-scan and store best escape route
            escape_clock, escape_clear = self._find_best_escape(avg_clearances, blocked)
            if escape_clock and escape_clear >= self.MIN_CLEARANCE:
                self.escape_clock = escape_clock
                self.escape_heading = self._clock_to_heading(escape_clock, self.current_heading)
                self.escape_clearance = escape_clear
            
            # Check if obstacle cleared (depth increased)
            if avg_depth > self.SAFE_DEPTH:
                self.state = "SAFE"
                return 12, "Clear ahead", avg_clearances
            
            # Check if entering danger zone
            if avg_depth < self.DANGER_DEPTH:
                self.state = "DANGER"
                print(f"[LookAhead] DANGER! Depth={avg_depth:.1f}m, escape={self.escape_clock}")
            
            # Still in warning - user can keep going but we're watching
            return 12, "Clear ahead", avg_clearances
        
        elif self.state == "DANGER":
            # Must turn NOW - use pre-stored escape route
            if self.escape_clock is None:
                # No escape found - try to find one now
                escape_clock, escape_clear = self._find_best_escape(avg_clearances, blocked)
                if escape_clock and escape_clear >= self.MIN_CLEARANCE:
                    self.escape_clock = escape_clock
                    self.escape_heading = self._clock_to_heading(escape_clock, self.current_heading)
                else:
                    return 12, "Stop - blocked", avg_clearances
            
            self.state = "TURNING"
            return self.escape_clock, f"Turn to {self.escape_clock}", avg_clearances
        
        elif self.state == "TURNING":
            # Check if user has aligned
            if self.escape_heading is not None:
                heading_diff = abs(self._heading_diff(self.current_heading, self.escape_heading))
                
                if heading_diff <= self.ALIGN_THRESHOLD:
                    # User aligned!
                    self.calibrated_12 = self.current_heading
                    self.escape_clock = None
                    self.escape_heading = None
                    self.state = "SAFE"
                    print(f"[LookAhead] Aligned! Re-calibrated to {self.current_heading:.0f}°")
                    return 12, "Good - keep going", avg_clearances
            
            # Check if depth cleared (user went around)
            if avg_depth > self.SAFE_DEPTH:
                self.state = "SAFE"
                self.escape_clock = None
                self.escape_heading = None
                return 12, "Clear ahead", avg_clearances
            
            # Still need to turn
            return self.escape_clock, f"Turn to {self.escape_clock}", avg_clearances
        
        return 12, "Clear ahead", avg_clearances


# Test
if __name__ == "__main__":
    nav = LookAheadNavigator()
    nav.calibrate(0)
    
    print("\n=== Look-Ahead Navigator Test ===\n")
    
    # Create full floor
    floor = np.ones((200, 200), dtype=bool)
    
    # Simulate approaching obstacle
    print("Simulating approach to obstacle:")
    depths = [5.0, 3.0, 2.5, 2.0, 1.5, 1.2, 0.8, 0.5]
    for depth in depths:
        clock, msg, _ = nav.navigate(floor, depth, [], user_heading=0)
        print(f"  Depth {depth:.1f}m: {msg} (state={nav.state})")
