# 🐟 Automated Real-Time Video Monitoring of Fish

Fish monitoring using YOLO11 Object Detection model + BotSort Tracking technique + Customized Counting algorithm

---

## 📌 Overview

- This Project focuses on river herring detection, all other type of fishes are considered as non-herring class.
- The main contribution includes:
    - The curated high-quality balanced training dataset (12,000 images)
    - The trained YOLO11 Object Detection Model weights on our dataset
    - The customized counting algorithm leveraging accumulated fish detections of 70% of the frames.

---

## 🎬 Research Demo
<div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
  <video src="output/tracking_nemasket0_113.mp4" controls autoplay loop muted style="width: 49%; border-radius: 8px;"></video>
  <video src="output/tracking_saco5_113.mp4" controls autoplay loop muted style="width: 49%; border-radius: 8px;"></video>
</div>


## ⚙️ Installation

### Clone the repository
```bash
git clone https://github.com/Sea-queue/Fish-detection-and-counting.git.git 
```

### Run the model
```bash
from ultralytics import YOLO

# Load a model with our trained weights
model = YOLO("path_to_the_weights")  # load a custom model
```


