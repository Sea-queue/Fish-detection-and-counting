import cv2
from ultralytics import solutions
from ultralytics.utils.plotting import Annotator

'''
In solutions.ObjectCounter.count():
call self.extract_tracks()

In solutions.solutions.extract_tracks():
call self.model.track()

In ultralytics.engine.model.track():
Returns: (List[ultralytics.engine.results.Results]): A list of tracking results, each a Results object.

In Results: has boxes attribute which is a Boxes class

In Boxes: has conf attribute which is the confidence score of each box

.cpu():
    This is a PyTorch method to move a tensor from GPU memory to CPU memory. This is useful for:
    - Interoperability with libraries or tools that don’t support GPU tensors.
    - Debugging, where you need to view or print the tensor data on the CPU.
'''

# Define your custom counter method
def custom_count(self, im0):
    if not self.region_initialized:
        self.initialize_region()
        self.region_initialized = True
    
    self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
    self.extract_tracks(im0)  # Extract tracks
    print("Custom Counter Method Called")
    for box in self.tracks[0].boxes:
        print(f'box: {box.xyxy}, conf score: {box.conf}, track ids: {box.id}, class: {box.cls}')
    

# Overwrite the original counter method
# solutions.ObjectCounter.count = custom_count


# cap = cv2.VideoCapture("/projects/CompVision/data/raw_data/videos/nemasket0.mp4")
# cap = cv2.VideoCapture("/projects/CompVision/data/raw_data/videos/nemasket_normal.mp4")
# cap = cv2.VideoCapture("/projects/CompVision/data/raw_data/videos/teton1.mp4")
cap = cv2.VideoCapture("/projects/CompVision/data/raw_data/videos/saco3.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = [(800, 10), (800, 1000)]  # For line counting
# (20, 400), (1080, 400), (1080, 360), (20, 360)] 
# region_points = [(80, 10), (120, 10), (120, 230), (80, 230)]  # For rectangle region counting
# region_points = [(800, 10), (1000, 10), (1000, 1000), (800, 1000)]  # For rectangle region counting
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]  # For polygon region counting

# Video writer
video_writer = cv2.VideoWriter("/projects/CompVision/seaqueue/yolov11/runs/count/saco3-110-line.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps-10, (2000, 1000))

# Init ObjectCounter
counter = solutions.ObjectCounter(
    show=True,  # Display the output
    conf=0.25,
    iou=0.7,
    region=region_points,  # Pass region points
    model="/projects/CompVision/seaqueue/yolov11/runs/train/train110/weights/best.pt",  # model="yolo11n-obb.pt" for object counting using YOLO11 OBB model.
    # classes=[0, 2],  # If you want to count specific classes i.e person and car with COCO pretrained model.
    # show_in=False,  # Display in counts
    # show_out=True,  # Display out counts
    # line_width=2,  # Adjust the line width for bounding boxes and text display
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    
    # Resize the frame
    resized_frame = cv2.resize(im0, (2000, 1000), interpolation=cv2.INTER_AREA)
    im0 = counter(resized_frame)
    video_writer.write(im0.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()