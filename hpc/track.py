from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import csv
import os
from collections import defaultdict, Counter

'''
In most YOLO + tracking implementations (like BoT-SORT or ByteTrack), track_id is primarily associated with object motion and appearance features, not the class label.
•	It does not switch track_id just because class prediction changes frame to frame.
•	However, it might switch IDs due to occlusion, poor detection, or re-entry of the object, especially if the tracker confidence drops.
'''

model = YOLO("/projects/CompVision/seaqueue/yolov11/runs/train/train113/weights/best.pt")
# cap = cv2.VideoCapture("/projects/CompVision/data/raw_data/videos/nemasket0.mp4")
# cap = cv2.VideoCapture("/projects/CompVision/data/raw_data/videos/nemasket_normal.mp4")
cap = cv2.VideoCapture("/projects/CompVision/data/raw_data/videos/nemasket_huge.mp4")
# cap = cv2.VideoCapture("/projects/CompVision/data/raw_data/videos/saco5.mp4")
# cap = cv2.VideoCapture("/projects/CompVision/data/raw_data/videos/teton2.mp4")
track_history = defaultdict(lambda: [])

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
# out = cv2.VideoWriter("/projects/CompVision/seaqueue/yolov11/runs/track/tracking_nemasket0_113.mp4", fourcc, fps, (width, height))
# out = cv2.VideoWriter("/projects/CompVision/seaqueue/yolov11/runs/track/tracking_nemasket_normal_113.mp4", fourcc, fps, (width, height))
out = cv2.VideoWriter("/projects/CompVision/seaqueue/yolov11/runs/track/tracking_nemasket_huge_113.mp4", fourcc, fps, (width, height))
# out = cv2.VideoWriter("/projects/CompVision/seaqueue/yolov11/runs/track/tracking_saco5_113.mp4", fourcc, fps, (width, height))
# out = cv2.VideoWriter("/projects/CompVision/seaqueue/yolov11/runs/track/tracking_teton2_113.mp4", fourcc, fps, (width, height))

# output CSV file
# tracking_output = "tracking_results_nemasket0_113.csv"
# tracking_output = "tracking_results_nemasket_normal.csv"
tracking_output = "tracking_results_nemasket_huge.csv"
# tracking_output = "tracking_results_saco5_113.csv"
# tracking_output = "tracking_results_teton2_113.csv"
with open(tracking_output, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["frame_id", "track_id", "confidence", "class_name", "x", "y", "w", "h"])

frame_id = -1

# 2) Buffers & bookkeeping
track_predictions    = defaultdict(list)   # confidences ≥0.7 → class names
track_positions      = defaultdict(list)   # x-center history for each ID
processed_track_ids  = set()                # IDs already finalized
classwise_track_ids  = {                    # final counts
    'Herring': set(),
    'Non-Herring': set()
}


'''
for each frame:
    run the traind detection model;
    if there is detection:
        get bounding_box; id; confidence; class_id
        
    calculate the exit margin according to different video inputs;
    
    if the confidence is over 70%:
        save {track_id: track_name} to track_prediction
        save {track_id: center_point} to track_position
    
    for id, position in track_position:
        if the id is processed or position are less then 10:
            keep accumulate information
        
        find the direction;
        check if the fish cross the exit point;
        if not:
            keep accumulate information
        if crossed:
            get the history for this id:
            for all the posibilities of this id:
                if one_class's count is over 70% of the total count:
                    add this id to the corresponding class

for each id/fish, if it cross the check point(65% of the frame width from the entering point), 
check all the history of this id, caculate the class detection ratio that are over 70% confidence.
'''

# 3) Exit margin ratio
EXIT_MARGIN_RATIO = 0.35  # 20% of frame width

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_id += 1
    results = model.track(frame, persist=True)

    # extract detections or fall back
    if results[0].boxes and results[0].boxes.id is not None:
        # xywh here is [x_center, y_center, width, height]
        xywh        = results[0].boxes.xywh.cpu().tolist()
        track_ids   = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()
        class_ids   = results[0].boxes.cls.int().cpu().tolist()
        annotated   = results[0].plot()
    else:
        xywh, track_ids, confidences, class_ids = [], [], [], []
        annotated = frame.copy()

    fh, fw = annotated.shape[:2]
    EXIT_MARGIN_PX = int(fw * EXIT_MARGIN_RATIO)

    # 4) Update history buffers
    for (x_c, y_c, w, h), tid, conf, cid in zip(xywh, track_ids, confidences, class_ids):
        cname = model.names[cid]
        if conf >= 0.7:
            track_predictions[tid].append(cname)
            # x_c is already the centroid
            track_positions[tid].append(x_c)

    # 5) Check for “exit” & finalize
    for tid, pos_hist in list(track_positions.items()):
        if tid in processed_track_ids or len(pos_hist) < 10:
            continue

        # positive → left→right; negative → right→left
        direction = pos_hist[-1] - pos_hist[0]

        if direction > 0:
            exited = pos_hist[-1] >= (fw - EXIT_MARGIN_PX)
        else:
            exited = pos_hist[-1] <= EXIT_MARGIN_PX

        if not exited:
            continue

        # finalize based on *all* history
        history = track_predictions.get(tid, [])
        if history:
            cnt = Counter(history)
            total = sum(cnt.values())
            for cls_name, count in cnt.items():
                if count / total >= 0.7:
                    classwise_track_ids[cls_name].add(tid)
                    break

        processed_track_ids.add(tid)

    # 6) Write CSV lines & draw trajectories
    with open(tracking_output, mode='a', newline='') as f:
        writer = csv.writer(f)
        for (x_c, y_c, w, h), tid, conf, cid in zip(xywh, track_ids, confidences, class_ids):
            cname = model.names[cid]
            writer.writerow([frame_id, tid, conf, cname, x_c, y_c, w, h])

            # draw trajectory
            t = track_history[tid]
            t.append((float(x_c), float(y_c)))
            if len(t) > 30:
                t.pop(0)
            pts = np.hstack(t).astype(np.int32).reshape((-1,1,2))
            cv2.polylines(annotated, [pts], False, (230,230,230), 1)

    # 7) Overlay final counts
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = fh / 720 * 2
    thickness = max(1, int(scale))
    text_h    = cv2.getTextSize("Non-Herring: 9999", font, scale, thickness)[0][1]
    margin    = 10
    text_w    = cv2.getTextSize("Non-Herring: 9999", font, scale, thickness)[0][0]
    x0        = fw - margin - text_w
    y0        = margin + text_h

    cv2.putText(
        annotated,
        f"Herring: {len(classwise_track_ids['Herring'])}",
        (x0, y0), font, scale, (0,255,0), thickness
    )
    cv2.putText(
        annotated,
        f"Non-Herring: {len(classwise_track_ids['Non-Herring'])}",
        (x0, y0 + text_h + 5), font, scale, (0,255,255), thickness
    )

    out.write(annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()