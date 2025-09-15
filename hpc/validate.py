from ultralytics import YOLO

# load the model
model = YOLO('/projects/CompVision/seaqueue/yolov11/runs/train/train110/weights/best.pt')

# Validate the model
metrics = model.val(data='/projects/CompVision/seaqueue/util/fish.yaml', plots=True, save_json=True, project='/projects/CompVision/seaqueue/yolov11/runs/validation', name='val4', save_txt=True, save_conf=True)
print('map50-95', metrics.box.map)  # map50-95
print('map50', metrics.box.map50)  # map50
print('map75', metrics.box.map75)  # map75
print('map50-95', metrics.box.maps)  # a list contains map50-95 of each category