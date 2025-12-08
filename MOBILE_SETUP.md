# AssistedVision - Hybrid Mobile Setup Guide

## Overview
The hybrid system allows your PC to process video while your mobile phone provides:
- Video streaming via camera
- Gyroscope orientation data
- Audio guidance output

## Setup Instructions

### Step 1: Install IP Webcam App on Phone
1. Install "IP Webcam" app from:
   - Android: Google Play Store
   - iOS: App Store (or use "IP Camera Lite")
2. Open the app and scroll down to "Start server"
3. Note the IP address shown (e.g., `http://192.168.1.100:8080`)

### Step 2: Install Python Dependencies on PC
```bash
pip install websockets
```

### Step 3: Start the PC Server
Run the main program with mobile flag:
```bash
python src/main.py --camera http://192.168.1.100:8080/video --yolo yolov8n.pt --output output/out.mp4 --mobile
```

Replace `192.168.1.100:8080` with your phone's IP from Step 1.

### Step 4: Connect Mobile Companion App
1. Start a simple HTTP server in the project directory:
   ```bash
   python -m http.server 8000
   ```
2. On your phone's browser, navigate to:
   ```
   http://YOUR_PC_IP:8000/mobile.html
   ```
3. Allow permissions for:
   - Gyroscope/motion sensors
   - Audio/speaker
4. You should see "Connected to PC" status

## System Architecture

```
┌─────────────┐         ┌─────────────┐
│   MOBILE    │         │     PC      │
│             │         │             │
│  Camera ────┼────────►│ Video       │
│             │  HTTP   │ Processing  │
│  Gyroscope ─┼────────►│ (YOLO +     │
│             │  WS     │  Depth)     │
│  Speaker ◄──┼─────────┤             │
│             │  WS     │ Risk        │
└─────────────┘         │ Analysis    │
                        └─────────────┘
```

## Usage Examples

### Example 1: Using phone camera with mobile audio
```bash
python src/main.py --camera http://192.168.1.100:8080/video --yolo yolov8n.pt --mobile
```

### Example 2: Using USB webcam with mobile audio (for testing)
```bash
python src/main.py --camera 0 --yolo yolov8n.pt --mobile
```

### Example 3: Using video file with PC audio (no mobile)
```bash
python src/main.py --video data/demo.mp4 --yolo yolov8n.pt
```

## Features

### Mobile App Features:
- Real-time gyroscope display (roll, pitch, yaw)
- WebSocket connection status
- Audio guidance with urgency levels
- Haptic feedback for warnings
- Screen wake lock (keeps screen on)
- Test audio button
- Calibration support

### PC Processing:
- YOLO object detection
- Depth estimation
- Risk assessment
- Collision prediction
- Gyroscope data logging
- Video output with visualizations

## Troubleshooting

### Phone can't connect to PC
- Ensure both devices are on same WiFi network
- Check firewall settings on PC (allow port 8765 and 8000)
- Verify PC's IP address: `ipconfig` (Windows) or `ifconfig` (Linux/Mac)

### No gyroscope data
- Allow motion sensor permissions in browser
- Try pressing "Calibrate" button in mobile app
- For iOS: Must use HTTPS or localhost

### Audio not playing on phone
- Check phone volume
- Allow audio permissions in browser
- Test with "Test Audio" button

### Video stream not working
- Verify IP Webcam app is running
- Check the video URL format: `http://IP:PORT/video`
- Some cameras use `/video` or `/videofeed` endpoint

## Network Configuration

### Find your PC's IP address:
**Windows:**
```powershell
ipconfig
```
Look for "IPv4 Address" under your WiFi adapter

**Linux/Mac:**
```bash
ifconfig
```
Look for `inet` address

### Port Requirements:
- Port 8765: WebSocket (PC ← → Mobile)
- Port 8000: HTTP Server (Mobile App)
- Port 8080: IP Webcam (Phone Camera)

## Performance Tips

1. **Reduce image size** for faster processing:
   ```bash
   --imgsz 320
   ```

2. **Skip depth estimation frames** to improve FPS:
   ```bash
   --skip_depth 5
   ```

3. **Disable video output** for performance:
   ```bash
   --no-show
   ```

4. **Keep phone charging** during extended use

5. **Use 5GHz WiFi** for better bandwidth

## Safety Notes

 **This is an assistive technology. Always use with caution:**
- Do not rely solely on the system
- Use in familiar environments first
- Have a companion when testing
- Ensure phone is securely mounted
- Keep phone battery charged

## Log Files

The system generates:
- `output/out.mp4` - Processed video with visualizations
- `output/output_log.jsonl` - Frame-by-frame data including gyroscope readings

## Advanced Configuration

### Adjust risk thresholds in `src/main.py`:
```python
if top['risk'] > 0.6:  # Change threshold here
```

### Modify audio urgency in `src/mobile_server.py`:
```python
urgency = min(1.0, (top['risk']-0.6)/0.4)
```

### Change WebSocket port:
```python
mobile_server = MobileServer(host='0.0.0.0', port=8765)
```

## Support

For issues, check:
1. Console output on PC
2. Browser console on mobile (F12 or inspect)
3. Ensure all dependencies installed
4. Verify network connectivity
