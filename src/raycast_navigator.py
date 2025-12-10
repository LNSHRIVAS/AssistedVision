"""
Ray Casting Navigator
=====================
Uses ray casting to find clearance distance in each clock direction.
More accurate than simple floor percentage - measures actual path length.

Casts 3 rays per sector for robustness against narrow obstacles.
"""

import numpy as np
import math
from typing import Tuple, List, Optional


class RayCastNavigator:
    """
    Navigation using ray casting for clearance detection.
    
    For each clock sector (9-3), casts multiple rays and finds
    the minimum clearance distance. Best direction = longest clearance.
    """
    
    CLOCKS = [9, 10, 11, 12, 1, 2, 3]
    
    # Angles for each clock hour (from bottom center, 0 = up)
    # Clock 12 = 0°, Clock 3 = -90°, Clock 9 = +90°
    CLOCK_ANGLES = {
        9:  90,   # Far left
        10: 60,   # Left
        11: 30,   # Slight left
        12: 0,    # Straight ahead
        1:  -30,  # Slight right
        2:  -60,  # Right
        3:  -90,  # Far right
    }
    
    # Configuration
    RAYS_PER_SECTOR = 3      # Cast 3 rays per sector for robustness
    RAY_SPREAD = 8           # Degrees spread for sub-rays (±8°)
    MIN_CLEARANCE = 0.1      # Minimum clearance ratio (10% of image height)
    OBSTACLE_DIST = 1.0      # meters for depth check
    
    def __init__(self):
        self.locked_clock = 12
        self.locked_clearance = 0.5
    
    def _cast_ray(self, floor_mask: np.ndarray, angle_deg: float) -> float:
        """
        Cast a single ray from bottom-center at given angle.
        
        Args:
            floor_mask: Boolean mask where True = floor
            angle_deg: Angle in degrees (0 = straight up, positive = left)
            
        Returns:
            Clearance as ratio of image height (0.0 to 1.0)
        """
        h, w = floor_mask.shape
        
        # Start point: bottom center
        start_x = w // 2
        start_y = h - 1
        
        # Convert angle to radians and calculate direction
        angle_rad = math.radians(angle_deg)
        dx = -math.sin(angle_rad)  # Negative because image y increases downward
        dy = -math.cos(angle_rad)  # Negative because we go up
        
        # Step along the ray
        step_size = 2  # pixels per step (balance speed vs accuracy)
        max_steps = int(h / step_size)
        
        clearance_pixels = 0
        
        for i in range(max_steps):
            x = int(start_x + dx * i * step_size)
            y = int(start_y + dy * i * step_size)
            
            # Check bounds
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            
            # Check if floor
            if floor_mask[y, x]:
                clearance_pixels = i * step_size
            else:
                # Hit non-floor - stop
                break
        
        # Return as ratio of image height
        return clearance_pixels / h
    
    def _cast_sector_rays(self, floor_mask: np.ndarray, clock: int) -> float:
        """
        Cast multiple rays for a clock sector and return minimum clearance.
        
        Args:
            floor_mask: Boolean floor mask
            clock: Clock hour (9-3)
            
        Returns:
            Minimum clearance ratio for the sector
        """
        base_angle = self.CLOCK_ANGLES[clock]
        
        # Cast rays at base angle and ± spread
        angles = [
            base_angle - self.RAY_SPREAD,
            base_angle,
            base_angle + self.RAY_SPREAD,
        ]
        
        clearances = [self._cast_ray(floor_mask, angle) for angle in angles]
        
        # Return minimum (most conservative)
        return min(clearances)
    
    def navigate(self, floor_mask: np.ndarray, center_depth: float = 5.0,
                 blocked_sectors: List[int] = None) -> Tuple[int, str, List[float]]:
        """
        Find best direction using ray casting.
        
        Returns: (clock_hour, guidance_text, sector_clearances)
        """
        if floor_mask is None or floor_mask.size == 0:
            return self.locked_clock, "Scanning...", [0.5] * 7
        
        blocked = set(blocked_sectors or [])
        
        # Cast rays for each sector
        sector_clearances = []
        for clock in self.CLOCKS:
            if clock in blocked:
                clearance = 0.0  # Blocked by YOLO
            else:
                clearance = self._cast_sector_rays(floor_mask, clock)
            sector_clearances.append(clearance)
        
        # Check center depth
        center_blocked = center_depth < self.OBSTACLE_DIST
        if center_blocked:
            sector_clearances[3] = 0.0  # Set clock 12 clearance to 0
        
        # Find best direction (maximum clearance)
        best_idx = int(np.argmax(sector_clearances))
        best_clock = self.CLOCKS[best_idx]
        best_clearance = sector_clearances[best_idx]
        
        # CENTER PRIORITY: If center is nearly as good (within 20%), prefer 12
        center_clearance = sector_clearances[3]
        if not center_blocked and center_clearance >= best_clearance * 0.8:
            best_clock = 12
            best_clearance = center_clearance
        
        # SMART LOCKING: Only change if new is significantly better or current is blocked
        current_idx = self.CLOCKS.index(self.locked_clock) if self.locked_clock in self.CLOCKS else 3
        current_clearance = sector_clearances[current_idx]
        
        should_change = False
        if current_clearance < self.MIN_CLEARANCE:
            should_change = True  # Current is blocked
        elif best_clearance > current_clearance * 1.3:
            should_change = True  # New is 30% better
        
        if should_change and best_clearance >= self.MIN_CLEARANCE:
            self.locked_clock = best_clock
            self.locked_clearance = best_clearance
        
        output_clock = self.locked_clock
        
        # Generate guidance
        if output_clock == 12:
            if center_blocked or center_clearance < self.MIN_CLEARANCE:
                guidance = "Stop - blocked"
            else:
                guidance = "Clear ahead"
        else:
            guidance = f"Go to {output_clock}"
        
        return output_clock, guidance, sector_clearances


# Test
if __name__ == "__main__":
    nav = RayCastNavigator()
    
    # Create test floor masks
    print("=== Ray Cast Navigator Test ===\n")
    
    # Test 1: Full floor
    full_floor = np.ones((200, 200), dtype=bool)
    clock, msg, clearances = nav.navigate(full_floor, 5.0)
    print(f"Full floor: {msg}")
    print(f"  Clearances: {[f'{c:.2f}' for c in clearances]}\n")
    
    # Test 2: Floor only on left
    left_floor = np.zeros((200, 200), dtype=bool)
    left_floor[:, :100] = True  # Left half is floor
    clock, msg, clearances = nav.navigate(left_floor, 5.0)
    print(f"Left floor only: {msg}")
    print(f"  Clearances: {[f'{c:.2f}' for c in clearances]}\n")
    
    # Test 3: Floor only on right
    right_floor = np.zeros((200, 200), dtype=bool)
    right_floor[:, 100:] = True  # Right half is floor
    clock, msg, clearances = nav.navigate(right_floor, 5.0)
    print(f"Right floor only: {msg}")
    print(f"  Clearances: {[f'{c:.2f}' for c in clearances]}\n")
    
    # Test 4: Narrow corridor in center
    corridor = np.zeros((200, 200), dtype=bool)
    corridor[:, 80:120] = True  # Center corridor
    clock, msg, clearances = nav.navigate(corridor, 5.0)
    print(f"Center corridor: {msg}")
    print(f"  Clearances: {[f'{c:.2f}' for c in clearances]}")
