# GestureFlow

GestureFlow is a real‐time hand and fingertip tracking system using OpenCV, MediaPipe, and classical image‐processing techniques. It supports two modes of operation:
- **MediaPipe mode**: uses Google's MediaPipe Hands.
- **Skin‐segmentation mode**: uses color‐space thresholding, contour analysis, curvature and convexity defects.

## Features

- Real‐time hand detection and tracking (up to two hands)
- Fingertip detection, labeling and smoothing
- Hand orientation estimation and finger naming (Thumb, Index, Middle, Ring, Pinky)
- Motion trails for tracked fingers
- Adaptive lighting conditions (auto, bright, normal, dark)
- Live FPS and processing‐time panel
- Toggleable display and light modes
- Threaded, parallel‐ready architecture

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy
- SciPy
- scikit‐learn
- MediaPipe
- TensorFlow (optional if you extend the model)
- Windows, macOS or Linux

## Installation
Make sure you have **uv** installed using [this guide](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Clone the repository
git clone https://github.com/krystofkubica/GestureFlow
cd GestureFlow

# Run program
uv run main.py
```