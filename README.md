# Assisted Vision - Navigation for Blind Users

A real-time navigation system for visually impaired users using AI-powered depth sensing, floor detection, and obstacle avoidance.

## Quick Start

### 1. Download the Mobile App
- Download `BlindNavigator.apk` from the Releases section
- Install on your Android phone

### 2. Start the Server (on your computer)
```bash
# Clone the repository
git clone https://github.com/LNSHRIVAS/AssistedVision.git
cd AssistedVision

# Install dependencies
pip install -r requirements.txt

# Start the navigation server
python src/fast_nav_server.py --port 8768
```

### 3. Connect the App
- Open the BlindNavigator app on your phone
- Enter your computer's IP address and port (default: 8768)
- Point your phone camera ahead and walk!

## Server Options

```bash
# Indoor mode (default)
python src/fast_nav_server.py --port 8768

# Outdoor mode (sidewalk only, excludes grass)
python src/fast_nav_server.py --port 8768 --outdoor
```

## How It Works

1. **Phone sends camera frames** → Server via WebSocket
2. **Server processes frames**:
   - Depth estimation (how far things are)
   - Floor segmentation (where can you walk)
   - YOLO object detection (people, chairs, etc.)
3. **Server sends guidance** → "Clear ahead", "Slow down", "Turn to 11"
4. **Phone speaks guidance** → Text-to-speech

## System Requirements

- **Server**: Python 3.8+, ~4GB RAM
- **Phone**: Android 8.0+, camera

## Files Structure

```
src/
├── fast_nav_server.py      # Main server (run this)
├── simple_navigator.py     # Navigation logic
├── depth_clean.py          # Depth estimation
├── floor_segmentation.py   # Floor detection
├── yolo_detector.py        # Object detection
├── clock_direction.py      # IMU clock direction
└── tts.py                  # Text-to-speech
```

## License
MIT
