#!/usr/bin/env python3
"""
Main entry point for the DirectGuide mobile application.

This script is designed to be run on an Android device using PyDroid3 or a similar
Python environment. It initializes the DirectGuideNavigator, captures video from the
device's camera, and provides real-time auditory feedback to the user.
"""

import os
import sys
import argparse
import cv2
import time

# Add the parent directory to the Python path to allow for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from directguide.navigator import DirectGuideNavigator
from directguide.guidance_templates import GuidanceMessage

def main(args):
    """
    Initializes and runs the DirectGuide navigation system.
    """
    
    # Initialize the DirectGuideNavigator
    try:
        navigator = DirectGuideNavigator(
            yolo_weights=args.yolo_weights,
            yolo_imgsz=args.imgsz,
            depth_skip_frames=args.depth_skip,
            device='cpu',
            enable_tts=True,
            debug=args.debug
        )
    except Exception as e:
        print(f"Error initializing navigator: {e}")
        return

    # Open the camera
    try:
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {args.camera_id}")
            return
    except Exception as e:
        print(f"Error opening camera: {e}")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    print("Starting navigation... Press Ctrl+C to exit.")

    try:
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera.")
                break

            # Process the frame
            try:
                result = navigator.process_frame(frame)
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

            # Optional: Display the debug view
            if args.show_debug:
                try:
                    debug_frame = navigator.draw_debug_overlay(frame, result)
                    cv2.imshow('DirectGuide Debug', debug_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"Error displaying debug view: {e}")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        navigator.cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DirectGuide Mobile Navigation')
    parser.add_argument('--yolo_weights', type=str, default='yolov8n.onnx', help='Path to YOLO ONNX weights')
    parser.add_argument('--imgsz', type=int, default=320, help='YOLO input image size')
    parser.add_argument('--depth_skip', type=int, default=3, help='Number of frames to skip for depth estimation')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera ID to use')
    parser.add_argument('--width', type=int, default=640, help='Camera frame width')
    parser.add_argument('--height', type=int, default=480, help='Camera frame height')
    parser.add_argument('--fps', type=int, default=30, help='Desired camera FPS')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--show_debug', action='store_true', help='Show debug video overlay')
    
    args = parser.parse_args()
    main(args)
