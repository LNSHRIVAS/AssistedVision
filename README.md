# 👁️ AssistedVision  
![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-green)
![MiDaS](https://img.shields.io/badge/MiDaS-Monocular%20Depth-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📘 Project Summary  
**AssistedVision** is an end-to-end computer vision system that processes video input in real time, detects objects, estimates their distance, evaluates risk, identifies free walking gaps, predicts safe turning directions, and generates spoken audio feedback.  

It integrates several CV components—YOLOv8 detection, MiDaS depth estimation, Kalman tracking, risk scoring, passable-gap detection, turn prediction, and optional mobile audio streaming—into one complete assistive perception pipeline.

---

# 🚀 Features

## 🔍 Object Detection (YOLOv8)  
- Runs via Ultralytics YOLOv8  
- Performs detection every frame  
- Filters predictions by confidence threshold  
- Produces bounding boxes + class labels  

*(See `detection.py` for implementation.)*

---

## 📏 Depth Estimation (MiDaS)  
- Uses MiDaS (DPT-based) model  
- Produces full-frame depth map  
- Extracts median depth per object bounding box  
- Provides relative distance estimation  

*(See `depth.py`.)*

---

## 🎯 Multi-Object Tracking  
- Kalman filter based tracker  
- IoU association + ID persistence  
- Smooth motion trajectories  
- Handles temporary occlusions  

*(See `tracker.py`.)*

---

## ⚠️ Risk Evaluation  
Risk is computed from:

- object distance (MiDaS depth)  
- bbox area (proximity indication)  
- object vertical position relative to horizon  
- class-based rules (e.g., people, vehicles)  

The engine generates:

- **high-risk audio warnings**  
- **distance-aware messages**  
- **object-position descriptions (“ahead”, “left”, “right”)**

*(Logic in `risk.py` and `prob_risk.py`.)*

---

## 🧭 Navigation & Path Guidance  

### ✔ Passable Gap Detection  
From `path_finder.py`:  
- analyzes object bounding boxes  
- identifies **free vertical gaps**  
- chooses **widest traversable path**  
- used to determine safe forward direction

### ✔ Turn Detection  
- if no safe forward gap is found → turn prediction  
- chooses **left** or **right** based on depth + empty space  
- audio output:  
  - “Turn slightly left”  
  - “Turn right”  
  - “Clear ahead”  

### ✔ Clock-Position Mapping  
Objects are mapped to **relative angular zones**:  
- “12 o’clock”  
- “2 o’clock”  
- “9 o’clock”  
for intuitive spoken feedback.

---

## 🔊 Audio Output (TTS)

Two modes:

### **1) Desktop Mode (default)**
- Uses Windows-TTS (PowerShell SAPI)
- Plays prioritized spoken messages

### **2) Mobile Mode (optional)**  
- Browser-based audio via WebSocket  
- Phone receives all spoken guidance  
- Phone can send gyro data back (future use)

*(See `tts.py` and `mobile_server.py`.)*

---

## 📱 Mobile Companion

To enable mobile mode:

```bash
python src/main.py --camera 0 --mobile
```
Open mobile.html on your phone (must be on the same WiFi).

## 🛠 System Architecture

               ┌──────────────────────────────┐
               │        Camera / Video        │
               └───────────────┬──────────────┘
                               ▼
               ┌──────────────────────────────┐
               │       YOLOv8 Detection       │
               └───────────────┬──────────────┘
                               ▼
               ┌──────────────────────────────┐
               │       Kalman Tracking        │
               └───────────────┬──────────────┘
                               ▼
               ┌──────────────────────────────┐
               │       MiDaS Depth Model      │
               └───────────────┬──────────────┘
                               ▼
      ┌─────────────────────────────────────────────────────┐
      │     Risk Engine + Gap Detection + Turning           │
      └───────────────┬─────────────────────────────────────┘
                      ▼
               ┌──────────────────────────────┐
               │        Audio Output (TTS)    │
               └──────────────────────────────┘

# ⚙ Installation

To set up the environment and run the project, follow these steps:

## 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/AssistedVision.git
cd AssistedVision
```

## 2️⃣ Create a Conda environment
```bash
conda create -n assistedvision python=3.10 -y
conda activate assistedvision
```
## 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

## 📁 Project Structure
```text
AssistedVision/
├── data/
│   ├── raw/                     # Raw video samples (optional)
│   ├── processed/               # Saved processed outputs
│   ├── samples/                 # Demo input videos
│   └── myvideo.mp4              # Example user-added video
│
├── src/
│   ├── main.py                  # Main pipeline: detection + depth + risk + audio + UI
│   ├── detection.py             # YOLOv8 detection wrapper
│   ├── depth.py                 # MiDaS depth estimation
│   ├── tracker.py               # Kalman filter + object tracking
│   ├── risk.py                  # Rule-based risk engine
│   ├── prob_risk.py             # Probabilistic risk scoring (experimental)
│   ├── path_finder.py           # Gap detection + navigation decision module
│   ├── instaYOLO_seg.py         # YOLO segmentation (optional module)
│   ├── tts.py                   # Text-to-speech (Windows SAPI)
│   ├── viz.py                   # Visualization utilities and drawing
│   ├── utils.py                 # Helper functions
│   ├── mobile_server.py         # WebSocket server for phone audio + gyro
│   └── __init__.py              # Package initializer
│
├── mobile.html                  # Mobile companion interface (audio + gyroscope)
│
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment file (optional)
│
├── run.sh                       # Linux run script (optional)
├── start_mobile.bat             # Windows helper script (mobile mode)
├── test_webcam.ps1              # Webcam test script for Windows
│
├── README.md                    # Project documentation
├── LICENSE                      # MIT license
└── COMPREHENSIVE_PROJECT_SUMMARY.md  # Full technical write-up
```

# ▶️ How to Run the System

AssistedVision can be run in two main modes:

1. **Real-time mode using your webcam**
2. **Offline mode using a video file**
3. **Android mobile mode (phone camera + phone audio + gyro)**

---

## 📷 Option 1 — Run in Real-Time with Your Webcam

### Steps:
1. Connect your webcam.
2. Open a terminal inside the project folder.
3. Run the command below:

```bash
python src/main.py --camera 0 --yolo yolov8n.pt --output webcam_output.mp4
```

## 🎞️ Option 2 — Run the System on a Video File

Step 1 — Put your video inside the data/ folder:
```bash
data/myvideo.mp4
```

Step 2 — Run this command:
```bash
python src/main.py --video data/myvideo.mp4 --yolo yolov8n.pt --output result.mp4
```

## 📱 Option 3 — Android Mobile Mode (Phone Camera + Phone Audio + Gyroscope)

> **Note:** Full mobile mode (phone camera + audio + gyro) is currently **tested and supported on Android**.

In this mode, your **Android phone acts as the camera and audio device**, while your laptop runs all computer vision and navigation logic.

---

### 1️⃣ Set up the Android phone camera (IP Webcam)

1. On your Android phone, install **“IP Webcam”** from the Google Play Store.
2. Open the app and scroll down to **“Start server”**.
3. Tap **Start server**.
4. At the bottom of the screen, you will see a URL like:

```text
http://192.168.0.15:8080/video
```
This is your phone camera stream URL.
You will use this in the --camera argument.

Make sure the phone and laptop are on the same WiFi network.

### 2️⃣ Start AssistedVision on the laptop (using phone camera)
In a terminal on your laptop, inside the project folder, run:
```bash
python src/main.py \
  --camera http://PHONE_IP:8080/video \
  --yolo yolov8n.pt \
  --mobile
```
Example:
```bash
python src/main.py --camera http://192.168.0.15:8080/video --yolo yolov8n.pt --mobile
```
This will:

Use the Android phone camera as the video source

Run YOLOv8, depth estimation, tracking, risk analysis, and navigation on the laptop

Start the mobile communication server for audio + gyroscope

### 3️⃣ Serve the mobile companion page from the laptop

From the project root on the laptop, start a simple HTTP server:
```bash
python -m http.server 8000
```
This makes mobile.html available over the network.

### 4️⃣ Connect the Android phone to the mobile companion

1. On the same Android phone, open Chrome.
2. In the address bar, go to:
```text
http://YOUR_PC_IP:8000/mobile.html
```
For example:
```text
http://192.168.0.20:8000/mobile.html
```
3. When the page loads, allow:
   3.1 Motion / gyroscope access
   3.2 Audio permissions if prompted

4. The status on the page should indicate that the phone is connected to the PC.

### ⚡ Optional performance settings (recommended for mobile mode)

If the stream is slow, you can add:
```bash
--imgsz 320 --skip-depth 5 --process-every 2
```
Example:
```bash
python src/main.py \
  --camera http://192.168.0.15:8080/video \
  --yolo yolov8n.pt \
  --mobile \
  --imgsz 320 \
  --skip-depth 5 \
  --process-every 2
```
