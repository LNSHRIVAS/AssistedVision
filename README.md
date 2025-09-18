# AssistiveVision 👁️  
*Empowering accessibility with AI-driven computer vision*  

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen)  
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)  

---

## 📖 Overview  
**AssistiveVision** is an AI-powered system designed to help visually impaired individuals navigate their surroundings safely.  
It combines **computer vision**, **machine learning**, and **assistive audio feedback** to:  
- Detect and track objects in real-time  
- Estimate depth or distance to obstacles  
- Provide clear voice guidance to the user  

The system is designed for **proactive navigation**, and can be adapted for **mobile devices**.  

---

## 🗂️ Project Structure  

```bash
AssistiveVision/
├── src/               # Core source code
│   ├── detection/     # Object detection (YOLO)
│   ├── tracking/      # Multi-object tracking
│   ├── depth/         # Depth estimation or LiDAR integration
│   ├── guidance/      # Voice guidance / LLM integration
│   ├── utils/         # Helper functions (logging, configs)
│   └── main.py        # System entry point
│
├── experiments/       # Jupyter notebooks and prototypes
├── data/              # Datasets (raw, processed, samples)
├── docs/              # Reports, slides, and documentation
├── tests/             # Unit tests
├── requirements.txt   # Project dependencies
├── .gitignore
├── LICENSE
└── README.md
