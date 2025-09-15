from ultralytics import YOLO
import os

# Load a model
model = YOLO("/projects/CompVision/seaqueue/yolov11/runs/train/train113/weights/best.pt")  # load a custom model

# # Predict with the model
# results = []
# files = os.listdir('/work/CompVision/data/all_by_quality/high_visibility/train_16/images')
# for file in files:
#     result = model(f'/work/CompVision/data/all_by_quality/high_visibility/train_16/images/{file}')  # predict on an image
#     results.append(result)
    
# print(results)

# Define path to video file
# src = "/work/CompVision/data/raw_data/videos/nemasket0.mp4"
# src = "/work/CompVision/data/raw_data/videos/teton1.mp4"
# src = "/work/CompVision/data/raw_data/videos/saco5_denoised.mp4"
# src = "seaqueue/yolov11/commands/saco5_denoised_gamma_0.6.mp4"
src = "/projects/CompVision/data/raw_data/videos/saco5.mp4"

# Run inference on the source
model.predict(
    source=src,
    conf=0.25,
    iou=0.7,
    save=True,
    project='/projects/CompVision/seaqueue/yolov11/runs/predict',
    name='saco5-113',
    save_frames=True,
    save_txt=True,
    save_conf=True,
)