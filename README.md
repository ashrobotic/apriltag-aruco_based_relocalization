# ROS2 Tag-Based Relocalization using ArUco Markers (Humble)

## 📌 Overview

This project implements an automatic and service-based relocalization system for ROS2 (Humble) using ArUco markers.

It detects visual markers using OpenCV, estimates robot pose in the map frame, and publishes a corrected `/initialpose` to recover from AMCL drift.

The system supports:
- Automatic drift detection
- Manual relocalization service
- Multi-marker detection
- YAML-based marker configuration
- Side-based pose offsets (left/right/front/back)

---

## 🚀 Features

✅ ArUco marker detection using OpenCV  
✅ Automatic AMCL drift monitoring  
✅ Relocalization cooldown logic  
✅ Closest marker selection  
✅ Map-frame pose computation  
✅ `/initialpose` publishing for Nav2/AMCL  
✅ TF broadcast of relocalized frame  
✅ Service call for manual relocalization  

---

## 🧠 How It Works

1. Detect ArUco markers from camera image
2. Estimate marker pose relative to camera
3. Convert OpenCV coordinates to ROS coordinate system
4. Use predefined marker map poses (YAML)
5. Compute robot pose in `map` frame
6. Publish corrected `/initialpose`
7. Broadcast TF for visualization
8. Monitor AMCL covariance and auto-trigger relocalization if drift detected

---

## 📂 Marker Configuration (YAML)

Example:

```yaml
markers:
  1:
    pose: [2.0, 1.5, 1.57]
    side: left
  2:
    pose: [4.0, 3.0, 0.0]
    side: right
