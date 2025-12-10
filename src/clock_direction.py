"""
Clock Direction Module
======================
Converts IMU orientation data to clock-hour directions for blind users.

This module can be integrated into the streaming server to provide
directional guidance based on where the user is pointing their phone.

Example Usage:
    from clock_direction import ClockDirectionConverter
    
    converter = ClockDirectionConverter()
    converter.calibrate(current_heading)  # Set reference
    
    result = converter.get_clock_direction(pitch, roll, heading)
    print(result['message'])  # "3 o'clock, sharp right"
"""

import math
from typing import Dict, Tuple, Optional
from collections import deque
import numpy as np


class ClockDirectionConverter:
    """
    Converts phone IMU data to clock-hour directions.
    
    Key concepts:
    - 12 o'clock = straight ahead (reference direction)
    - Hours go clockwise: 1-11 represent rotations from straight ahead
    - Uses compass heading to determine absolute direction
    - Uses pitch to validate phone angle
    """
    
    def __init__(self, smoothing_window: int = 5):
        """
        Initialize converter.
        
        Args:
            smoothing_window: Number of frames to smooth over
        """
        # Calibration
        self.reference_heading: Optional[float] = None
        self.heading_calibrated: bool = False
        
        # Smoothing buffers
        self.heading_buffer = deque(maxlen=smoothing_window)
        self.pitch_buffer = deque(maxlen=smoothing_window)
        
        # State tracking
        self.last_clock_hour: Optional[int] = None
        self.consecutive_stable_frames: int = 0
        
        # Configuration
        self.stability_threshold: int = 3  # Frames before announcing change
        self.pitch_min: float = -80  # Minimum acceptable pitch
        self.pitch_max: float = 20   # Maximum acceptable pitch
    
    def calibrate(self, heading: float) -> None:
        """
        Set the reference heading (12 o'clock direction).
        
        Args:
            heading: Compass heading in degrees (0-360)
        """
        self.reference_heading = heading
        self.heading_calibrated = True
        self.heading_buffer.clear()
        self.pitch_buffer.clear()
        print(f"[ClockDirection] Calibrated: 12 o'clock = {heading:.1f}°")
    
    def reset_calibration(self) -> None:
        """Reset calibration (e.g., when user changes direction)"""
        self.reference_heading = None
        self.heading_calibrated = False
        self.heading_buffer.clear()
        self.pitch_buffer.clear()
        self.last_clock_hour = None
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to 0-360 range."""
        angle = angle % 360
        if angle < 0:
            angle += 360
        return angle
    
    def get_relative_angle(self, heading: float) -> float:
        """
        Calculate relative angle from reference heading.
        
        Args:
            heading: Current compass heading (0-360)
            
        Returns:
            Relative angle in degrees (-180 to +180)
            Positive = clockwise, Negative = counter-clockwise
        """
        if not self.heading_calibrated:
            return 0.0
        
        diff = heading - self.reference_heading
        
        # Normalize to -180 to +180
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        
        return diff
    
    def relative_angle_to_clock_hour(self, relative_angle: float) -> Tuple[int, str, str]:
        """
        Convert relative angle to clock hour.
        
        Args:
            relative_angle: Angle relative to 12 o'clock (-180 to +180)
            
        Returns:
            Tuple of (hour, short_desc, long_desc)
            - hour: 1-12
            - short_desc: Brief description (e.g., "right")
            - long_desc: Detailed description (e.g., "to your right")
        """
        # Normalize to 0-360 for clock math
        angle = self.normalize_angle(relative_angle)
        
        # 360° / 12 hours = 30° per hour
        hour_float = angle / 30.0
        hour = int(round(hour_float))
        if hour == 0:
            hour = 12
        
        # Map to descriptions
        hour_info = {
            12: ("ahead", "straight ahead", "Continue straight"),
            1: ("1 o'clock", "slightly right", "Bear slightly to the right"),
            2: ("2 o'clock", "right", "Turn to your right"),
            3: ("3 o'clock", "sharp right", "Sharp turn right"),
            4: ("4 o'clock", "back right", "Behind you to the right"),
            5: ("5 o'clock", "back right", "Behind you to the right"),
            6: ("6 o'clock", "behind", "Turn around, obstacle behind"),
            7: ("7 o'clock", "back left", "Behind you to the left"),
            8: ("8 o'clock", "back left", "Behind you to the left"),
            9: ("9 o'clock", "sharp left", "Sharp turn left"),
            10: ("10 o'clock", "left", "Turn to your left"),
            11: ("11 o'clock", "slightly left", "Bear slightly to the left"),
        }
        
        clock_name, short_desc, long_desc = hour_info.get(hour, ("unknown", "unknown", "Unknown direction"))
        
        return hour, short_desc, long_desc
    
    def check_pitch_valid(self, pitch: float) -> Tuple[bool, str]:
        """
        Check if phone pitch is in acceptable range.
        
        Args:
            pitch: Phone pitch in degrees
            
        Returns:
            Tuple of (is_valid, message)
        """
        if pitch < self.pitch_min:
            return False, f"Tilt phone forward more (currently {pitch:.0f}°)"
        elif pitch > self.pitch_max:
            return False, f"Tilt phone down (currently {pitch:.0f}°)"
        else:
            return True, "Pitch OK"
    
    def get_clock_direction(
        self,
        pitch: float,
        roll: float,
        heading: float,
        use_smoothing: bool = True
    ) -> Dict:
        """
        Main method: Convert IMU data to clock direction.
        
        Args:
            pitch: Phone pitch in degrees (-90 to +90)
            roll: Phone roll in degrees (-180 to +180)
            heading: Compass heading in degrees (0-360)
            use_smoothing: Whether to apply temporal smoothing
            
        Returns:
            Dictionary with:
            {
                'calibrated': bool,
                'clock_hour': int (1-12),
                'clock_name': str,
                'short_desc': str,
                'long_desc': str,
                'relative_angle': float,
                'pitch_valid': bool,
                'pitch_message': str,
                'raw_pitch': float,
                'raw_heading': float,
                'smooth_pitch': float,
                'smooth_heading': float,
                'message': str (what to speak),
                'should_announce': bool
            }
        """
        # Add to smoothing buffers
        if use_smoothing:
            self.heading_buffer.append(heading)
            self.pitch_buffer.append(pitch)
            smooth_heading = float(np.mean(self.heading_buffer))
            smooth_pitch = float(np.mean(self.pitch_buffer))
        else:
            smooth_heading = heading
            smooth_pitch = pitch
        
        # Check if calibrated
        if not self.heading_calibrated:
            return {
                'calibrated': False,
                'clock_hour': 12,
                'clock_name': 'not calibrated',
                'short_desc': 'not calibrated',
                'long_desc': 'Point phone straight ahead to calibrate',
                'relative_angle': 0.0,
                'pitch_valid': False,
                'pitch_message': 'Not calibrated',
                'raw_pitch': pitch,
                'raw_heading': heading,
                'smooth_pitch': smooth_pitch,
                'smooth_heading': smooth_heading,
                'message': 'Point phone straight ahead and wait for calibration',
                'should_announce': False
            }
        
        # Validate pitch
        pitch_valid, pitch_message = self.check_pitch_valid(smooth_pitch)
        
        # Calculate clock direction
        relative_angle = self.get_relative_angle(smooth_heading)
        clock_hour, short_desc, long_desc = self.relative_angle_to_clock_hour(relative_angle)
        clock_name = f"{clock_hour} o'clock"
        
        # Determine if we should announce
        should_announce = False
        if not pitch_valid:
            message = pitch_message
            should_announce = True  # Always announce pitch issues
        else:
            # Check if direction changed significantly
            if self.last_clock_hour is None or clock_hour != self.last_clock_hour:
                self.consecutive_stable_frames = 0
            else:
                self.consecutive_stable_frames += 1
            
            # Announce if stable for threshold frames
            if self.consecutive_stable_frames == self.stability_threshold:
                should_announce = True
                self.last_clock_hour = clock_hour
            
            message = f"{clock_name}, {short_desc}"
        
        return {
            'calibrated': True,
            'clock_hour': clock_hour,
            'clock_name': clock_name,
            'short_desc': short_desc,
            'long_desc': long_desc,
            'relative_angle': relative_angle,
            'pitch_valid': pitch_valid,
            'pitch_message': pitch_message,
            'raw_pitch': pitch,
            'raw_heading': heading,
            'smooth_pitch': smooth_pitch,
            'smooth_heading': smooth_heading,
            'message': message,
            'should_announce': should_announce
        }
    
    def get_direction_to_target(
        self,
        current_heading: float,
        target_heading: float,
        pitch: float
    ) -> Dict:
        """
        Get clock direction to a specific target heading.
        Useful for guiding user toward a detected safe path.
        
        Args:
            current_heading: User's current heading (0-360)
            target_heading: Heading of target/safe path (0-360)
            pitch: Current phone pitch
            
        Returns:
            Dictionary with clock direction to target
        """
        # Temporarily override reference
        old_ref = self.reference_heading
        self.reference_heading = current_heading
        
        result = self.get_clock_direction(pitch, 0, target_heading, use_smoothing=False)
        
        # Restore reference
        self.reference_heading = old_ref
        
        return result


# Convenience functions for quick usage
def heading_to_clock(heading: float, reference: float = 0.0) -> str:
    """
    Quick conversion: heading to clock direction string.
    
    Args:
        heading: Current heading (0-360)
        reference: Reference heading for 12 o'clock (default: 0=North)
        
    Returns:
        String like "3 o'clock, sharp right"
    """
    converter = ClockDirectionConverter(smoothing_window=1)
    converter.calibrate(reference)
    result = converter.get_clock_direction(0, 0, heading, use_smoothing=False)
    return result['message']


def angle_to_clock(angle: float) -> str:
    """
    Quick conversion: relative angle to clock direction.
    
    Args:
        angle: Relative angle in degrees (-180 to +180)
        
    Returns:
        String like "2 o'clock, right"
    """
    converter = ClockDirectionConverter(smoothing_window=1)
    converter.calibrate(0)
    heading = (angle + 360) % 360
    result = converter.get_clock_direction(0, 0, heading, use_smoothing=False)
    return result['message']


# Test function
def test_clock_directions():
    """Test the clock direction converter"""
    print("\n" + "="*70)
    print("Clock Direction Converter Test")
    print("="*70 + "\n")
    
    converter = ClockDirectionConverter()
    converter.calibrate(0)  # North = 12 o'clock
    
    test_cases = [
        (0, "Straight ahead (North)"),
        (30, "1 o'clock (slightly right)"),
        (60, "2 o'clock (right)"),
        (90, "3 o'clock (sharp right)"),
        (180, "6 o'clock (behind)"),
        (270, "9 o'clock (sharp left)"),
        (300, "10 o'clock (left)"),
        (330, "11 o'clock (slightly left)"),
    ]
    
    for heading, expected in test_cases:
        result = converter.get_clock_direction(-30, 0, heading, use_smoothing=False)
        print(f"Heading {heading:3d}° → {result['clock_name']:12s} - {result['short_desc']:15s} | Expected: {expected}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    test_clock_directions()
