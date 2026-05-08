"""
CountingEvaluator — runs model.track() on video files and counts fish.

Algorithm (unchanged from counting_test.py):
  • Each frame: run model.track(); collect detections above count_conf threshold.
  • Per track: accumulate x-center positions + high-confidence class names.
  • Exit detection: when a track's x-center crosses within exit_margin of the
    frame edge (in its direction of travel) AND has ≥10 position samples:
      - Perform majority-vote class assignment (must reach majority_ratio).
      - Mark that track as finalized so it is never double-counted.

Outputs per (dataset, model) pair:
  - Annotated .mp4 with bounding boxes, trajectories, and live count overlay
  - .csv with per-frame: frame_id, track_id, confidence, class, x, y, w, h
  - Entry in the run's _summary/ JSON + CSV
"""

from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .base import Evaluator
from ..reporting.summary import MetricsSummary


class CountingEvaluator(Evaluator):

    def setup(self) -> None:
        """Resolve and validate the video path."""
        entry = self.registry.get(self.config.dataset_name)

        if entry.get("type") != "counting":
            raise ValueError(
                f"Dataset '{self.config.dataset_name}' has type='{entry.get('type')}'. "
                "Use 'eval.py count' only with counting datasets."
            )

        self._video_path = self._require_file(
            entry["video_path"], label="Video file"
        )
        print(f"  [count] Video → {self._video_path}")

    def run(self, weights_path: str) -> dict:
        """Track and count fish in the video for one set of model weights."""
        self._require_file(weights_path, label="Weights file")
        run_dir = self._make_run_dir(weights_path)

        video_stem  = self._video_path.stem
        weight_stem = Path(weights_path).stem
        out_video   = run_dir / f"{video_stem}_{weight_stem}.mp4"
        out_csv     = run_dir / f"{video_stem}_{weight_stem}.csv"

        model  = YOLO(weights_path)
        counts = self._run_tracking(model, out_video, out_csv)

        result = {
            "dataset":      self.config.dataset_name,
            "weights":      weights_path,
            "video":        str(self._video_path),
            "counts":       counts,
            "output_video": str(out_video),
            "output_csv":   str(out_csv),
            "output_dir":   str(run_dir),
        }

        print(f"\n  --- Counts: {weight_stem} on {self.config.dataset_name} ---")
        for cls_name, cnt in sorted(counts.items()):
            print(f"  {cls_name}: {cnt}")
        print(f"  Video: {out_video}")
        print(f"  CSV  : {out_csv}")

        return result

    def summarize(self) -> None:
        """Write JSON + CSV summary for this dataset's entire counting run."""
        writer = MetricsSummary(
            output_root=self.config.output_root,
            dataset_name=self.config.dataset_name,
            mode="count",
        )
        writer.write(self.results)
        writer.print_table(self.results)

    # ------------------------------------------------------------------ #
    # Core tracking loop
    # ------------------------------------------------------------------ #
    def _run_tracking(
        self,
        model: YOLO,
        out_video_path: Path,
        out_csv_path: Path,
    ) -> dict[str, int]:
        """
        Frame-by-frame tracking loop.

        Returns
        -------
        dict[str, int]
            {class_name: count} for every class the model knows about.
        """
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self._video_path}")

        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        # CSV header
        out_csv_path.write_text(
            "frame_id,track_id,confidence,class_name,x,y,w,h\n"
        )

        # ---- tracking state ----
        track_history    = defaultdict(list)   # (x,y) trajectory for drawing
        track_predictions = defaultdict(list)  # high-conf class names per track
        track_positions  = defaultdict(list)   # x-center history per track
        processed_ids    = set()               # tracks that have been finalized
        class_names      = None
        classwise_counts: dict[str, set] = defaultdict(set)  # class → set of track_ids

        exit_margin    = self.config.exit_margin
        count_conf     = self.config.count_conf
        majority_ratio = self.config.majority_ratio
        min_track_len  = self.config.min_track_len

        frame_id = -1

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_id += 1

            results = model.track(
                frame,
                persist=True,
                imgsz=self.config.imgsz,
                conf=self.config.conf,
                max_det=self.config.max_det,
                device=self.config.device,
            )

            # Initialize class name map from model on first frame
            if class_names is None:
                class_names = model.names
                for name in class_names.values():
                    classwise_counts.setdefault(name, set())

            # Unpack detections
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                xywh       = boxes.xywh.cpu().tolist()
                track_ids  = boxes.id.int().cpu().tolist()
                confs      = boxes.conf.cpu().tolist()
                class_ids  = boxes.cls.int().cpu().tolist()
                annotated  = results[0].plot()
            else:
                xywh, track_ids, confs, class_ids = [], [], [], []
                annotated = frame.copy()

            fh, fw = annotated.shape[:2]
            exit_px = int(fw * exit_margin)

            # --- Accumulate history for high-confidence detections ---
            for (x_c, y_c, w, h), tid, conf, cid in zip(
                xywh, track_ids, confs, class_ids
            ):
                if conf >= count_conf:
                    track_predictions[tid].append(class_names[cid])
                    track_positions[tid].append(x_c)

            # --- Check exit and finalise counts ---
            for tid, pos_hist in list(track_positions.items()):
                if tid in processed_ids or len(pos_hist) < min_track_len:
                    continue

                direction = pos_hist[-1] - pos_hist[0]
                exited = (
                    pos_hist[-1] >= (fw - exit_px)
                    if direction > 0
                    else pos_hist[-1] <= exit_px
                )
                if not exited:
                    continue

                history = track_predictions.get(tid, [])
                if history:
                    cnt   = Counter(history)
                    total = sum(cnt.values())
                    for cls_name, count in cnt.most_common():
                        if count / total >= majority_ratio:
                            classwise_counts[cls_name].add(tid)
                            break

                processed_ids.add(tid)

            # --- Write CSV rows + draw trajectories ---
            with open(out_csv_path, "a") as f:
                for (x_c, y_c, w, h), tid, conf, cid in zip(
                    xywh, track_ids, confs, class_ids
                ):
                    cname = class_names[cid]
                    f.write(
                        f"{frame_id},{tid},{conf:.4f},{cname},"
                        f"{x_c:.1f},{y_c:.1f},{w:.1f},{h:.1f}\n"
                    )
                    t = track_history[tid]
                    t.append((float(x_c), float(y_c)))
                    if len(t) > 30:
                        t.pop(0)
                    pts = np.hstack(t).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated, [pts], False, (230, 230, 230), 1)

            # --- Overlay live count text ---
            annotated = self._draw_counts(annotated, classwise_counts, fw, fh)

            writer.write(annotated)

            # Progress log every 100 frames
            if frame_id % 100 == 0:
                print(f"  Frame {frame_id}/{total_frames}", end="")
                for cls_name, ids in sorted(classwise_counts.items()):
                    print(f"  | {cls_name}: {len(ids)}", end="")
                print()

            if not self.config.no_display:
                cv2.imshow("Fish Counting", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        return {cls: len(ids) for cls, ids in classwise_counts.items()}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _draw_counts(
        frame: np.ndarray,
        classwise_counts: dict[str, set],
        fw: int,
        fh: int,
    ) -> np.ndarray:
        """Render per-class counts in the top-right corner of the frame."""
        if not classwise_counts:
            return frame

        font      = cv2.FONT_HERSHEY_SIMPLEX
        scale     = fh / 720 * 2
        thickness = max(1, int(scale))
        margin    = 10
        colors    = [(0, 255, 0), (0, 255, 255), (255, 128, 0), (255, 0, 255)]

        lines = [(cls, len(ids)) for cls, ids in sorted(classwise_counts.items())]
        max_text = max(f"{n}: {c}" for n, c in lines)
        text_w, text_h = cv2.getTextSize(max_text, font, scale, thickness)[0]
        x0 = fw - margin - text_w
        y0 = margin + text_h

        for i, (cls_name, count) in enumerate(lines):
            color = colors[i % len(colors)]
            cv2.putText(
                frame,
                f"{cls_name}: {count}",
                (x0, y0 + i * (text_h + 5)),
                font, scale, color, thickness,
            )
        return frame
