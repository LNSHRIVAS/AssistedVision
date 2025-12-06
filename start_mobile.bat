@echo off
REM Quick Start Script for AssistedVision Hybrid System

echo ================================================
echo   AssistedVision - Hybrid Mobile System
echo ================================================
echo.

REM Get local IP address
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do set LOCAL_IP=%%a
set LOCAL_IP=%LOCAL_IP:~1%

echo Your PC IP Address: %LOCAL_IP%
echo.
echo SETUP INSTRUCTIONS:
echo -------------------
echo 1. Install "IP Webcam" app on your phone
echo 2. Start the IP Webcam server on your phone
echo 3. Note the IP address shown (e.g., 192.168.1.100:8080)
echo 4. Open mobile companion: http://%LOCAL_IP%:8000/mobile.html
echo.
echo Starting HTTP server for mobile app...
echo.

REM Start HTTP server in background
start "HTTP Server" cmd /k "cd /d %~dp0 && python -m http.server 8000"

timeout /t 2 /nobreak >nul

echo.
echo HTTP Server started on port 8000
echo.
echo ================================================
echo NOW READY TO RUN ASSISTED VISION
echo ================================================
echo.
set /p CAMERA_URL="Enter your phone's IP Webcam URL (e.g., http://192.168.1.100:8080/video): "

echo.
echo Starting AssistedVision with mobile support...
echo.

REM Run the main program
.venv\Scripts\python.exe src\main.py --camera %CAMERA_URL% --yolo yolov8n.pt --output output\out.mp4 --mobile

echo.
echo Program finished.
pause
