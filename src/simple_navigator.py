"""
Simple Depth Navigator
======================
Simple, reliable navigation based primarily on DEPTH.

Philosophy:
- Floor detection is NOISY and unreliable for direction
- Depth is MORE RELIABLE for obstacle detection
- Only give specific directions when YOLO confirms obstacle

Guidance levels:
1. DEPTH > 2m     → "Clear ahead" (safe, just walk)
2. DEPTH 1-2m    → "Clear ahead" (still safe)
3. DEPTH < 1m    → "Slow down" (getting close to something)
4. DEPTH < 0.6m  → "Stop" (too close)
5. YOLO + blocked→ "Turn to X" (only when confirmed obstacle)
"""

import numpy as np
import math
from typing import Tuple, List
from collections import deque


class SimpleDepthNavigator:
    """
    Simple navigation based on depth readings.
    Only gives specific turn commands when YOLO confirms obstacle.
    """
    
    CLOCKS = [9, 10, 11, 12, 1, 2, 3]
    
    # IMAGE-SPACE ray casting angles
    IMAGE_ANGLES = {
        9: 90, 10: 60, 11: 30, 12: 0, 1: -30, 2: -60, 3: -90
    }
    
    # IMU heading offsets
    HEADING_OFFSETS = {
        9: -90, 10: -60, 11: -30, 12: 0, 1: 30, 2: 60, 3: 90
    }
    
    # Thresholds
    SAFE_DEPTH = 2.0        # > 2m = completely safe
    SLOW_DEPTH = 1.0        # < 1m = slow down
    STOP_DEPTH = 0.6        # < 0.6m = stop
    ALIGN_THRESHOLD = 25    # Degrees to consider aligned
    
    def __init__(self):
        self.state = "CLEAR"  # CLEAR, SLOW, STOP, TURNING
        
        # Heading tracking
        self.current_heading = 0
        self.calibrated_12 = 0
        
        # Turn tracking (only used when YOLO detects obstacle)
        self.target_clock = None
        self.target_heading = None
        
        # Smoothing
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
    
    def calibrate(self, heading: float):
        self.calibrated_12 = heading
        self.current_heading = heading
        print(f"[SimpleNav] Calibrated: 12 o'clock = {heading:.0f}°")
    
    def _find_clear_direction(self, blocked_sectors: set) -> int:
        """Find first clear direction, preferring slight turns."""
        # Priority: 11, 1, 10, 2, 9, 3 (prefer slight turns)
        priority_order = [11, 1, 10, 2, 9, 3]
        for clock in priority_order:
            if clock not in blocked_sectors:
                return clock
        return None
    
    def navigate(self, floor_mask: np.ndarray, 
                 center_depth: float = 5.0,
                 blocked_sectors: List[int] = None,
                 user_heading: float = None) -> Tuple[int, str, List[float]]:
        
        if user_heading is not None:
            self.current_heading = user_heading
        
        blocked = set(blocked_sectors or [])
        
        # Smooth depth
        self.depth_history.append(center_depth)
        avg_depth = np.mean(self.depth_history)
        
        # Dummy clearances (not used for decisions anymore)
        clearances = [0.5] * 7
        
        # === SIMPLE DEPTH-BASED LOGIC ===
        
        # If we were turning, check if aligned
        if self.state == "TURNING" and self.target_heading is not None:
            heading_diff = abs(self._heading_diff(self.current_heading, self.target_heading))
            if heading_diff <= self.ALIGN_THRESHOLD:
                # Aligned!
                self.calibrated_12 = self.current_heading
                self.target_clock = None
                self.target_heading = None
                self.state = "CLEAR"
                print(f"[SimpleNav] Aligned! Re-calibrated to {self.current_heading:.0f}°")
                return 12, "Good - keep going", clearances
            else:
                # Still turning
                return self.target_clock, f"Turn to {self.target_clock}", clearances
        
        # Check if YOLO blocked center (12 in blocked)
        center_blocked_by_yolo = 12 in blocked
        
        # Depth-based decisions
        if avg_depth > self.SAFE_DEPTH:
            # Completely safe
            self.state = "CLEAR"
            return 12, "Clear ahead", clearances
        
        elif avg_depth > self.SLOW_DEPTH:
            # Getting closer but still ok
            self.state = "CLEAR"
            return 12, "Clear ahead", clearances
        
        elif avg_depth > self.STOP_DEPTH:
            # Close - warn user
            if center_blocked_by_yolo:
                # YOLO confirms obstacle - give specific direction
                clear_dir = self._find_clear_direction(blocked)
                if clear_dir:
                    self.state = "TURNING"
                    self.target_clock = clear_dir
                    self.target_heading = self._clock_to_heading(clear_dir, self.current_heading)
                    return clear_dir, f"Turn to {clear_dir}", clearances
                else:
                    self.state = "STOP"
                    return 12, "Stop - blocked", clearances
            else:
                # No YOLO confirmation - just warn
                self.state = "SLOW"
                return 12, "Slow down", clearances
        
        else:
            # Very close - stop
            if center_blocked_by_yolo:
                clear_dir = self._find_clear_direction(blocked)
                if clear_dir:
                    self.state = "TURNING"
                    self.target_clock = clear_dir
                    self.target_heading = self._clock_to_heading(clear_dir, self.current_heading)
                    return clear_dir, f"Turn to {clear_dir}", clearances
            
            self.state = "STOP"
            return 12, "Stop", clearances


# Test
if __name__ == "__main__":
    nav = SimpleDepthNavigator()
    nav.calibrate(0)
    
    print("\n=== Simple Depth Navigator Test ===\n")
    
    # Simulate approaching obstacle
    print("Approaching obstacle (no YOLO):")
    depths = [5.0, 2.5, 1.5, 0.8, 0.5]
    for depth in depths:
        clock, msg, _ = nav.navigate(None, depth, [], user_heading=0)
        print(f"  Depth {depth:.1f}m: {msg}")
    
    print("\nApproaching with YOLO blocking center:")
    nav2 = SimpleDepthNavigator()
    nav2.calibrate(0)
    depths = [2.0, 0.8]
    for depth in depths:
        clock, msg, _ = nav2.navigate(None, depth, [12], user_heading=0)  # 12 blocked
        print(f"  Depth {depth:.1f}m: {msg}")
