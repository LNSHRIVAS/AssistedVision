"""
Target Lock Navigator
=====================
Locks a REAL-WORLD target heading when recommending a direction.
Tracks user's actual turn via IMU heading to detect alignment.

Flow:
1. Recommend "Go to 1" → Calculate real-world target heading
2. Lock target heading (fixed in real world)
3. Track user's IMU heading as they turn
4. When heading matches target → "Good, keep going"
5. Re-calibrate 12 o'clock to new facing
"""

import numpy as np
import math
from typing import Tuple, List, Optional
from collections import deque


class TargetLockNavigator:
    """
    Navigator with real-world target locking.
    
    Uses IMU heading to track when user has turned toward recommended direction.
    This solves the camera-relative problem.
    """
    
    CLOCKS = [9, 10, 11, 12, 1, 2, 3]
    
    # IMAGE-SPACE ray casting angles (for floor detection)
    # Left in image = positive angle from vertical
    IMAGE_ANGLES = {
        9: 90, 10: 60, 11: 30, 12: 0, 1: -30, 2: -60, 3: -90
    }
    
    # REAL-WORLD heading offsets (for IMU turn tracking)
    # IMU heading DECREASES when turning LEFT
    HEADING_OFFSETS = {
        9: -90, 10: -60, 11: -30, 12: 0, 1: 30, 2: 60, 3: 90
    }
    
    # Configuration
    RAYS_PER_SECTOR = 3
    RAY_SPREAD = 8
    MIN_CLEARANCE = 0.15
    ALIGN_THRESHOLD = 20      # Degrees - user within ±20° of target = aligned
    CHANGE_THRESHOLD = 0.25   # New path must be 25% better to switch
    STABLE_FRAMES = 5         # Frames needed to confirm new direction
    
    def __init__(self):
        # State
        self.state = "SCANNING"  # SCANNING, TARGETING, ALIGNED
        
        # Target lock
        self.target_clock = 12          # Current target clock hour
        self.target_heading = None      # Real-world heading of target (degrees)
        self.current_heading = 0        # User's current IMU heading
        self.calibrated_12 = 0          # What heading = 12 o'clock
        
        # Stability
        self.candidate_clock = 12
        self.candidate_count = 0
        self.clearance_history = deque(maxlen=5)  # Smooth clearances
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to -180 to +180."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def _clock_to_heading(self, clock: int, base_heading: float) -> float:
        """Convert clock hour to real-world heading using IMU convention."""
        offset = self.HEADING_OFFSETS.get(clock, 0)
        return self._normalize_angle(base_heading + offset)
    
    def _heading_diff(self, h1: float, h2: float) -> float:
        """Get smallest angle difference between two headings."""
        diff = h1 - h2
        return self._normalize_angle(diff)
    
    def _cast_ray(self, floor_mask: np.ndarray, angle_deg: float) -> float:
        """Cast ray from bottom-center, return clearance ratio."""
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
    
    def _get_sector_clearances(self, floor_mask: np.ndarray, 
                                blocked: set) -> List[float]:
        """Get clearance for each clock sector."""
        clearances = []
        
        for clock in self.CLOCKS:
            if clock in blocked:
                clearances.append(0.0)
                continue
            
            base_angle = self.IMAGE_ANGLES[clock]  # Use IMAGE_ANGLES for ray casting
            angles = [base_angle - self.RAY_SPREAD, base_angle, base_angle + self.RAY_SPREAD]
            rays = [self._cast_ray(floor_mask, a) for a in angles]
            clearances.append(min(rays))
        
        return clearances
    
    def update_heading(self, imu_heading: float):
        """Update user's current heading from IMU."""
        self.current_heading = imu_heading
    
    def calibrate(self, heading: float):
        """Set current heading as 12 o'clock."""
        self.calibrated_12 = heading
        self.target_heading = heading
        self.target_clock = 12
        self.state = "ALIGNED"
        print(f"[TargetLock] Calibrated: 12 o'clock = {heading:.0f}°")
    
    def navigate(self, floor_mask: np.ndarray, 
                 center_depth: float = 5.0,
                 blocked_sectors: List[int] = None,
                 user_heading: float = None) -> Tuple[int, str, List[float]]:
        """
        Main navigation with target locking.
        
        Args:
            floor_mask: Boolean floor mask
            center_depth: Depth at center in meters
            blocked_sectors: Sectors blocked by YOLO
            user_heading: Current user heading from IMU (degrees)
            
        Returns:
            (clock_hour, guidance_text, sector_clearances)
        """
        if floor_mask is None or floor_mask.size == 0:
            return self.target_clock, "Scanning...", [0.5] * 7
        
        # Update heading if provided
        if user_heading is not None:
            self.current_heading = user_heading
        
        blocked = set(blocked_sectors or [])
        
        # Get clearances
        clearances = self._get_sector_clearances(floor_mask, blocked)
        
        # Apply depth check for center
        if center_depth < 1.0:
            clearances[3] = 0.0  # Block center
        
        # Smooth clearances
        self.clearance_history.append(clearances)
        if len(self.clearance_history) >= 3:
            avg_clearances = np.mean(list(self.clearance_history), axis=0).tolist()
        else:
            avg_clearances = clearances
        
        # Get center clearance
        center_clearance = avg_clearances[3]  # 12 o'clock
        center_is_safe = center_clearance >= self.MIN_CLEARANCE and center_depth >= 0.8
        
        # Find best alternative (only used if center is blocked)
        best_idx = int(np.argmax(avg_clearances))
        best_clock = self.CLOCKS[best_idx]
        best_clearance = avg_clearances[best_idx]
        
        # === CENTER-FIRST PHILOSOPHY ===
        # Rule 1: If center is safe, GO STRAIGHT (don't suggest turns)
        # Rule 2: Only suggest turns when center is actually BLOCKED
        
        if center_is_safe:
            # CENTER IS SAFE - user can keep going straight
            if self.state == "TARGETING":
                # Was turning, but now center is clear - check if aligned first
                heading_diff = abs(self._heading_diff(self.current_heading, self.target_heading))
                if heading_diff <= self.ALIGN_THRESHOLD:
                    # User aligned with turn, re-calibrate
                    self.calibrated_12 = self.current_heading
                    print(f"[TargetLock] Aligned! Re-calibrated to {self.current_heading:.0f}°")
            
            self.state = "ALIGNED"
            self.target_clock = 12
            return 12, "Clear ahead", avg_clearances
        
        # CENTER IS BLOCKED - need to turn
        if self.state == "ALIGNED" or self.state == "SCANNING":
            # Start turning to best alternative
            if best_clearance >= self.MIN_CLEARANCE and best_clock != 12:
                self.target_clock = best_clock
                self.target_heading = self._clock_to_heading(best_clock, self.current_heading)
                self.state = "TARGETING"
                print(f"[TargetLock] Center blocked, turning to {best_clock} at {self.target_heading:.0f}°")
                return best_clock, f"Turn to {best_clock}", avg_clearances
            else:
                # No good alternative
                return 12, "Stop - blocked", avg_clearances
        
        elif self.state == "TARGETING":
            # Already turning - check if aligned
            heading_diff = abs(self._heading_diff(self.current_heading, self.target_heading))
            
            if heading_diff <= self.ALIGN_THRESHOLD:
                # User aligned!
                self.state = "ALIGNED"
                self.calibrated_12 = self.current_heading
                self.target_clock = 12
                print(f"[TargetLock] User aligned! Re-calibrated to {self.current_heading:.0f}°")
                return 12, "Good - keep going", avg_clearances
            
            # Check if target is still valid
            target_idx = self.CLOCKS.index(self.target_clock)
            target_clearance = avg_clearances[target_idx]
            
            if target_clearance < self.MIN_CLEARANCE or self.target_clock in blocked:
                # Target became blocked - find new direction
                if best_clearance >= self.MIN_CLEARANCE and best_clock != 12:
                    self.target_clock = best_clock
                    self.target_heading = self._clock_to_heading(best_clock, self.current_heading)
                    return best_clock, f"Turn to {best_clock}", avg_clearances
                else:
                    self.state = "ALIGNED"
                    return 12, "Stop - blocked", avg_clearances
            
            # Keep guiding to target
            return self.target_clock, f"Turn to {self.target_clock}", avg_clearances
        
        return 12, "Clear ahead", avg_clearances


# Test
if __name__ == "__main__":
    nav = TargetLockNavigator()
    nav.calibrate(0)  # North = 12 o'clock
    
    print("\n=== Target Lock Navigator Test ===\n")
    
    # Simulate: Clear floor on right (clock 2)
    floor = np.zeros((200, 200), dtype=bool)
    floor[:, 120:] = True  # Floor on right side
    
    print("1. Floor only on right side:")
    for i in range(7):
        clock, msg, _ = nav.navigate(floor, 5.0, [], user_heading=0)
        print(f"   Frame {i}: {msg} (state={nav.state})")
    
    print("\n2. User turns right (heading changes 0 → -50°):")
    for heading in [0, -15, -30, -45, -60]:
        clock, msg, _ = nav.navigate(floor, 5.0, [], user_heading=heading)
        print(f"   Heading {heading}°: {msg} (state={nav.state})")
