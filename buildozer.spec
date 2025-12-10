[app]
title = Assisted Vision
package.name = org.example.assistedvision
package.domain = org.example
source.dir = src
source.include_exts = py,png,jpg,kv,atlas,onnx,json,pt

version = 0.1
requirements = python3,kivy,opencv,numpy,onnxruntime-lite

orientation = portrait

# Android specific permissions
android.permissions = CAMERA,INTERNET,VIBRATE

android.api = 31
android.minapi = 28
android.sdk = 24
android.ndk = 23c

# Fullscreen mode
fullscreen = 1

[buildozer]
# PPA for Kivy
android.p4a.source_dir = D:/AssistedVision/.buildozer/android/p4a-develop
android.p4a.branch = develop
