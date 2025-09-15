from ultralytics import YOLO

# training round count
round = 113

# build a new model from YAML
model = YOLO("yolo11l.yaml").load("yolo11l.pt")
# model = YOLO("yolo11l-cls.yaml").load("yolo11l-cls.pt")

# load a pretrained model
# model = YOLO(f'/projects/CompVision/seaqueue/yolov11/runs/train/train{round-1}/weights/best.pt')  

# Train the model
results = model.train(
    name=f'train{round}',
    data='/projects/CompVision/seaqueue/util/fish.yaml', 
    # classification
    # data='/projects/CompVision/data/all_by_quality/high_visibility/classification', 
    epochs=130,
    batch=32,
    weight_decay=0.00001,
    conf=0.7,
    project='/projects/CompVision/seaqueue/yolov11/runs/train',
    imgsz=640,

    # Data augmentation
    # Color & lighting augmentation
    hsv_h=0.02,       # smaller shift; fish color shouldn’t drift wildly
    hsv_s=0.5,        
    hsv_v=0.4,        

    # Geometric transforms – moderate to preserve fish shape
    degrees=5.0,      # small rotations are usually sufficient
    translate=0.05,   # small shifts to simulate off-center framing
    scale=0.3,        
    shear=1.0,        
    perspective=0.0003, # very slight to be safe

    # Flipping – horizontal good, vertical only if justified
    fliplr=0.5,       
    flipud=0.0,       # unless fish are often upside-down

    # Color noise – simulate lighting/camera change
    bgr=0.05,         

    # Advanced mixing
    mosaic=1.0,       # keep this high if fish appear in clusters
    mixup=0.1,        # lower it; mixup can confuse models on objects with shape
)

