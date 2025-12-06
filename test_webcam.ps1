# Quick Test Script - Test with webcam before using phone

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   AssistedVision - Quick Webcam Test" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "This will test the system using your computer's webcam (camera 0)" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the test" -ForegroundColor Yellow
Write-Host ""

Start-Sleep -Seconds 2

# Run with webcam
& "C:/Users/sdsha/Downloads/AssistedVision-Assisted_vision_Modified/.venv/Scripts/python.exe" src/main.py --camera 0 --yolo yolov8n.pt --output output/test_webcam.mp4

Write-Host ""
Write-Host "Test completed! Check output/test_webcam.mp4 for results" -ForegroundColor Green
Write-Host ""
pause
