"""
Video fish detection, tracking, and counting test script.

Usage:
    python test_video.py --video path/to/video.mp4 --weights model/train113.pt
    python test_video.py --video path/to/video.mp4 --weights model/train113.pt --imgsz 1280 --conf 0.25 --max-det 300
"""

import argparse
import os
from collections import defaultdict, Counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Fish detection + tracking + counting on video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--weights", required=True, help="Path to YOLO model weights (.pt)")
    parser.add_argument("--output-dir", default="runs/video_test", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per frame")
    parser.add_argument("--count-conf", type=float, default=0.7,
                        help="Confidence threshold for counting decisions")
    parser.add_argument("--majority-ratio", type=float, default=0.7,
                        help="Class majority ratio to finalize a track's class")
    parser.add_argument("--exit-margin", type=float, default=0.35,
                        help="Exit margin ratio (fraction of frame width)")
    parser.add_argument("--no-display", action="store_true", help="Disable cv2 window display")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Video:  {args.video}")
    print(f"Model:  {args.weights}")

    # Load model
    model = YOLO(args.weights)

    # Open video
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open video: {args.video}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")

    # Setup output
    os.makedirs(args.output_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    weight_stem = os.path.splitext(os.path.basename(args.weights))[0]
    run_name = f"{video_stem}_{weight_stem}"

    out_video_path = os.path.join(args.output_dir, f"{run_name}.mp4")
    out_csv_path = os.path.join(args.output_dir, f"{run_name}.csv")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    # CSV header
    with open(out_csv_path, "w") as f:
        f.write("frame_id,track_id,confidence,class_name,x,y,w,h\n")

    # Tracking state
    track_history = defaultdict(list)       # trajectory points for drawing
    track_predictions = defaultdict(list)   # class names for high-conf detections
    track_positions = defaultdict(list)     # x-center history
    processed_track_ids = set()
    class_names = None                      # will be set from model
    classwise_track_ids = defaultdict(set)  # final counts per class

    frame_id = -1

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_id += 1

        # Run tracking
        results = model.track(
            frame,
            persist=True,
            imgsz=args.imgsz,
            conf=args.conf,
            max_det=args.max_det,
            device=device,
        )

        # Initialize class names from model on first frame
        if class_names is None:
            class_names = model.names
            for name in class_names.values():
                classwise_track_ids.setdefault(name, set())

        # Extract detections
        if results[0].boxes and results[0].boxes.id is not None:
            xywh = results[0].boxes.xywh.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            annotated = results[0].plot()
        else:
            xywh, track_ids, confidences, class_ids = [], [], [], []
            annotated = frame.copy()

        fh, fw = annotated.shape[:2]
        exit_margin_px = int(fw * args.exit_margin)

        # Update history buffers
        for (x_c, y_c, w, h), tid, conf, cid in zip(xywh, track_ids, confidences, class_ids):
            cname = class_names[cid]
            if conf >= args.count_conf:
                track_predictions[tid].append(cname)
                track_positions[tid].append(x_c)

        # Check for exit & finalize counts
        for tid, pos_hist in list(track_positions.items()):
            if tid in processed_track_ids or len(pos_hist) < 10:
                continue

            direction = pos_hist[-1] - pos_hist[0]
            if direction > 0:
                exited = pos_hist[-1] >= (fw - exit_margin_px)
            else:
                exited = pos_hist[-1] <= exit_margin_px

            if not exited:
                continue

            history = track_predictions.get(tid, [])
            if history:
                cnt = Counter(history)
                total = sum(cnt.values())
                for cls_name, count in cnt.items():
                    if count / total >= args.majority_ratio:
                        classwise_track_ids[cls_name].add(tid)
                        break

            processed_track_ids.add(tid)

        # Write CSV
        with open(out_csv_path, "a") as f:
            for (x_c, y_c, w, h), tid, conf, cid in zip(xywh, track_ids, confidences, class_ids):
                cname = class_names[cid]
                f.write(f"{frame_id},{tid},{conf:.4f},{cname},{x_c:.1f},{y_c:.1f},{w:.1f},{h:.1f}\n")

                # Draw trajectory
                t = track_history[tid]
                t.append((float(x_c), float(y_c)))
                if len(t) > 30:
                    t.pop(0)
                pts = np.hstack(t).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], False, (230, 230, 230), 1)

        # Overlay counts
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = fh / 720 * 2
        thickness = max(1, int(scale))
        margin = 10

        # Build count text lines
        count_lines = []
        for cls_name in sorted(classwise_track_ids.keys()):
            count_lines.append((cls_name, len(classwise_track_ids[cls_name])))

        # Draw from top-right
        if count_lines:
            max_text = max(f"{n}: {c}" for n, c in count_lines)
            text_w, text_h = cv2.getTextSize(max_text, font, scale, thickness)[0]
            x0 = fw - margin - text_w
            y0 = margin + text_h

            colors = [(0, 255, 0), (0, 255, 255), (255, 128, 0), (255, 0, 255)]
            for i, (cls_name, count) in enumerate(count_lines):
                color = colors[i % len(colors)]
                cv2.putText(
                    annotated,
                    f"{cls_name}: {count}",
                    (x0, y0 + i * (text_h + 5)),
                    font, scale, color, thickness,
                )

        out.write(annotated)

        # Progress
        if frame_id % 100 == 0:
            print(f"  Frame {frame_id}/{total_frames}", end="")
            for cls_name, ids in sorted(classwise_track_ids.items()):
                print(f"  | {cls_name}: {len(ids)}", end="")
            print()

        if not args.no_display:
            cv2.imshow("Fish Counting", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Final summary
    print(f"\n{'='*50}")
    print("FINAL COUNTS:")
    for cls_name, ids in sorted(classwise_track_ids.items()):
        print(f"  {cls_name}: {len(ids)}")
    print(f"{'='*50}")
    print(f"Output video: {out_video_path}")
    print(f"Output CSV:   {out_csv_path}")


if __name__ == "__main__":
    main()
