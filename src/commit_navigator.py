"""
Commit-Based Navigation Algorithm
=================================
State machine that commits to directions and only changes when certain.

States:
  SCANNING  - Finding best path (needs 5 frames)
  COMMITTED - Locked to direction
  BLOCKED   - Current direction blocked, finding new
"""

import numpy as np
from collections import deque
from typing import Tuple, List, Optional


class CommitNavigator:
    """
    State machine navigator that commits to directions.
    Prevents flip-flopping by using multi-frame confirmation.
    """
    
    CLOCKS = [9, 10, 11, 12, 1, 2, 3]
    SECTOR_MAP = {9:0, 10:1, 11:2, 12:3, 1:4, 2:5, 3:6}
    
    MIN_FLOOR = 0.12       # Minimum floor to be walkable
    COMMIT_FRAMES = 5      # Frames needed to commit to new direction
    BLOCK_FRAMES = 3       # Frames needed to confirm blocked
    OBSTACLE_DIST = 1.0    # Depth threshold for obstacle
    
    def __init__(self):
        self.state = "SCANNING"
        self.committed_clock = 12       # Currently committed direction
        self.scan_history = deque(maxlen=self.COMMIT_FRAMES)  # Recent best directions
        self.block_count = 0            # Consecutive blocked frames
        
    def navigate(self, floor_mask, center_depth: float = 5.0, 
                 blocked_sectors: List[int] = None) -> Tuple[int, str, List[float]]:
        """
        Main navigation with state machine.
        
        Returns: (clock_hour, guidance_text, sector_floors)
        """
        # Calculate sector floors
        if floor_mask is None or floor_mask.size == 0:
            return self.committed_clock, "Scanning...", [0.5] * 7
        
        h, w = floor_mask.shape
        bottom = floor_mask[h//2:, :]
        bw = bottom.shape[1]
        sector_w = bw // 7
        
        sector_floors = []
        for i in range(7):
            sx = i * sector_w
            ex = (i + 1) * sector_w if i < 6 else bw
            sector = bottom[:, sx:ex]
            floor_pct = float(np.mean(sector))
            sector_floors.append(floor_pct)
        
        # Zero out YOLO-blocked sectors
        blocked = set(blocked_sectors or [])
        for clk in blocked:
            if clk in self.SECTOR_MAP:
                sector_floors[self.SECTOR_MAP[clk]] = 0.0
        
        # Check center
        center_floor = sector_floors[3]
        center_blocked = center_depth < self.OBSTACLE_DIST or center_floor < self.MIN_FLOOR or 12 in blocked
        
        # Find best direction this frame
        best_clock, best_floor = self._find_best(sector_floors, center_blocked)
        
        # Check if current committed direction is valid
        committed_idx = self.SECTOR_MAP.get(self.committed_clock, 3)
        committed_floor = sector_floors[committed_idx]
        committed_valid = (
            self.committed_clock not in blocked and
            committed_floor >= self.MIN_FLOOR and
            not (self.committed_clock == 12 and center_blocked)
        )
        
        # === STATE MACHINE ===
        if self.state == "SCANNING":
            return self._state_scanning(best_clock, best_floor, sector_floors)
        
        elif self.state == "COMMITTED":
            return self._state_committed(committed_valid, best_clock, best_floor, sector_floors)
        
        else:  # BLOCKED - finding new direction
            return self._state_blocked(best_clock, best_floor, sector_floors)
    
    def _find_best(self, sector_floors: List[float], center_blocked: bool) -> Tuple[int, float]:
        """Find direction with maximum floor."""
        best_clock = 12
        best_floor = 0
        
        for clk in self.CLOCKS:
            if clk == 12 and center_blocked:
                continue
            
            idx = self.SECTOR_MAP[clk]
            floor = sector_floors[idx]
            
            if floor > best_floor:
                best_floor = floor
                best_clock = clk
        
        return best_clock, best_floor
    
    def _state_scanning(self, best_clock: int, best_floor: float, 
                        sector_floors: List[float]) -> Tuple[int, str, List[float]]:
        """SCANNING: Looking for consistent best direction."""
        self.scan_history.append(best_clock)
        
        # Need COMMIT_FRAMES of history
        if len(self.scan_history) < self.COMMIT_FRAMES:
            return 12, "Scanning...", sector_floors
        
        # Check if all recent frames agree
        if len(set(self.scan_history)) == 1:
            # All frames agree - COMMIT!
            self.committed_clock = best_clock
            self.state = "COMMITTED"
            self.block_count = 0
            
            if best_clock == 12:
                return 12, "Clear ahead", sector_floors
            else:
                return best_clock, f"Go to {best_clock}", sector_floors
        
        # Not consistent yet - keep scanning
        # Use most common direction as tentative
        from collections import Counter
        most_common = Counter(self.scan_history).most_common(1)[0][0]
        return most_common, "Scanning...", sector_floors
    
    def _state_committed(self, committed_valid: bool, best_clock: int, 
                         best_floor: float, sector_floors: List[float]) -> Tuple[int, str, List[float]]:
        """COMMITTED: Locked to direction, monitoring for blocks."""
        
        # CENTER PRIORITY: If center is now clear and good, switch to 12
        center_floor = sector_floors[3]
        if self.committed_clock != 12 and center_floor >= 0.40:
            # Center is now good - switch to it (prefer straight)
            self.committed_clock = 12
            self.block_count = 0
            return 12, "Clear ahead", sector_floors
        
        if committed_valid:
            # Direction still valid - keep it
            self.block_count = 0
            
            if self.committed_clock == 12:
                return 12, "Clear ahead", sector_floors
            else:
                return self.committed_clock, f"Go to {self.committed_clock}", sector_floors
        else:
            # Direction blocked - count frames
            self.block_count += 1
            
            if self.block_count >= self.BLOCK_FRAMES:
                # Confirmed blocked - switch to finding new
                self.state = "BLOCKED"
                self.scan_history.clear()
                return self.committed_clock, "Finding new path...", sector_floors
            
            # Not confirmed yet - keep current
            if self.committed_clock == 12:
                return 12, "Obstacle ahead", sector_floors
            else:
                return self.committed_clock, f"Go to {self.committed_clock}", sector_floors
    
    def _state_blocked(self, best_clock: int, best_floor: float,
                       sector_floors: List[float]) -> Tuple[int, str, List[float]]:
        """BLOCKED: Current direction blocked, quickly find new."""
        self.scan_history.append(best_clock)
        
        # Need 3 frames to confirm new direction (faster than initial scan)
        if len(self.scan_history) < 3:
            # While scanning, output warning
            if best_floor >= self.MIN_FLOOR:
                return best_clock, f"Switching to {best_clock}...", sector_floors
            else:
                return 12, "Stop - blocked", sector_floors
        
        # Check if 3 frames agree
        recent = list(self.scan_history)[-3:]
        if len(set(recent)) == 1:
            # Found new direction
            self.committed_clock = best_clock
            self.state = "COMMITTED"
            self.block_count = 0
            
            if best_clock == 12:
                return 12, "Clear ahead", sector_floors
            else:
                return best_clock, f"Go to {best_clock}", sector_floors
        
        # Keep searching
        return 12, "Finding path...", sector_floors


# Test
if __name__ == "__main__":
    nav = CommitNavigator()
    
    # Simulate frames
    import random
    for i in range(20):
        # Random floor with some consistency
        base = random.choice([9, 10, 12, 1, 2])
        floors = [0.1] * 7
        floors[nav.SECTOR_MAP[base]] = 0.8
        
        clock, msg, _ = nav.navigate(np.ones((100, 100)), 2.0, [])
        print(f"Frame {i}: {msg} (state={nav.state})")
