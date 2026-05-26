"""
ZoneCountingEvaluator — zone-based traversal counting WITHOUT track stitching.

Algorithm (from without_tracking_prediction.ipynb):
  • Each frame: run model.track() with BotSort; collect all detections.
  • Zone classification: each detection's x-center is classified as
    "left" / "middle" / "right" based on ENTRY_MARGIN.
  • Valid traversal: a track that entered from one side (left/right) and
    exited the opposite side past EXIT_MARGIN, with sufficient track length.
  • Majority-vote class assignment across the track's per-frame detections.
  • Absent-frame tracking: marks tracks as "missing" after ABSENT_THRESHOLD
    consecutive frames without detection.

Outputs per (dataset, model) pair:
  - Annotated .mp4 with bounding boxes, zone lines, and live count overlay
  - .csv with per-frame tracking details
  - Entry in the run's _summary/ JSON + CSV
"""

from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from .base import Evaluator
from ..reporting.summary import MetricsSummary


def _get_zone(cx: float, frame_width: int, entry_margin: float) -> str:
    if cx < frame_width * entry_margin:
        return "left"
    elif cx > frame_width * (1 - entry_margin):
        return "right"
    return "middle"


def _is_valid_traversal(track: dict, frame_width: int, exit_margin: float, min_track_length: int) -> bool:
    track_length = track["last_frame"] - track["first_frame"]
    if track_length < min_track_length:
        return False
    first_side = track["first_side"]
    last_side = track["last_side"]
    final_x = track["positions"][-1][0]
    if first_side == "left" and last_side == "right" and final_x > frame_width * (1 - exit_margin):
        return True
    if first_side == "right" and last_side == "left" and final_x < frame_width * exit_margin:
        return True
    return False


def _decide_track_class(class_history: list[str]) -> str:
    return Counter(class_history).most_common(1)[0][0]


class ZoneCountingEvaluator(Evaluator):

    def setup(self) -> None:
        entry = self.registry.get(self.config.dataset_name)
        if entry.get("type") != "counting":
            raise ValueError(
                f"Dataset '{self.config.dataset_name}' has type='{entry.get('type')}'. "
                "Use 'eval.py count' only with counting datasets."
            )
        self._video_path = self._require_file(entry["video_path"], label="Video file")
        print(f"  [zone-count] Video → {self._video_path}")

    def run(self, weights_path: str) -> dict:
        self._require_file(weights_path, label="Weights file")
        run_dir = self._make_run_dir(weights_path)

        video_stem = self._video_path.stem
        weight_stem = Path(weights_path).stem
        out_video = run_dir / f"{video_stem}_{weight_stem}_zone.mp4"
        out_csv = run_dir / f"{video_stem}_{weight_stem}_zone.csv"

        model = YOLO(weights_path)
        counts = self._run_tracking(model, out_video, out_csv)

        result = {
            "dataset": self.config.dataset_name,
            "weights": weights_path,
            "algorithm": "zone",
            "video": str(self._video_path),
            "counts": counts,
            "output_video": str(out_video),
            "output_csv": str(out_csv),
            "output_dir": str(run_dir),
        }

        print(f"\n  --- Counts (zone): {weight_stem} on {self.config.dataset_name} ---")
        for cls_name, cnt in sorted(counts.items()):
            print(f"  {cls_name}: {cnt}")
        print(f"  Video: {out_video}")
        print(f"  CSV  : {out_csv}")
        return result

    def summarize(self) -> None:
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
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self._video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        # CSV header
        csv_file = open(out_csv_path, "w", newline="")
        import csv
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame_id", "track_id", "class_name", "confidence",
            "center_x", "center_y", "status", "track_info",
        ])

        # ── Read margins from config ──
        entry_margin = self.config.zone_entry_margin
        exit_margin = self.config.zone_exit_margin
        absent_threshold = self.config.zone_absent_threshold
        min_track_length = self.config.zone_min_track_length

        print(f"  Zone params: entry_margin={entry_margin}, exit_margin={exit_margin}, "
              f"absent_threshold={absent_threshold}, min_track_length={min_track_length}")

        # ── Tracking state ──
        tracks: dict[int, dict] = {}
        counted_ids: set[int] = set()
        fish_counts: dict[str, int] = defaultdict(int)

        frame_id = -1

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_id += 1

            if self.config.grayscale:
                frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

            results = model.track(
                frame, persist=True,
                imgsz=self.config.imgsz,
                conf=self.config.conf,
                max_det=self.config.max_det,
                device=self.config.device,
                verbose=False,
            )

            current_frame_ids: set[int] = set()
            annotated_frame = frame.copy()

            if results[0].boxes is not None and results[0].boxes.id is not None:
                xywh = results[0].boxes.xywh.cpu().tolist()
                ids = results[0].boxes.id.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()

                for (x_c, y_c, w, h), tid, conf, cid in zip(xywh, ids, confidences, class_ids):
                    cname = model.names[cid]
                    current_side = _get_zone(x_c, width, entry_margin)

                    if tid not in tracks:
                        tracks[tid] = {
                            "first_frame": frame_id,
                            "last_frame": frame_id,
                            "first_side": current_side,
                            "last_side": current_side,
                            "positions": [(x_c, y_c)],
                            "class_history": [cname],
                            "absent_frames": 0,
                            "status": "tracking",
                        }
                    else:
                        t = tracks[tid]
                        t["last_frame"] = frame_id
                        t["last_side"] = current_side
                        t["positions"].append((x_c, y_c))
                        t["class_history"].append(cname)
                        t["absent_frames"] = 0
                        if t["status"] == "missing":
                            t["status"] = "tracking"

                    current_frame_ids.add(tid)
                    track = tracks[tid]

                    # Draw bounding box
                    box_color = (150, 150, 150) if track["first_side"] == "middle" else (255, 0, 0)
                    x1, y1 = int(x_c - w / 2), int(y_c - h / 2)
                    x2, y2 = int(x_c + w / 2), int(y_c + h / 2)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(annotated_frame, f"ID:{tid} {cname} {conf:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                    # Real-time counting check
                    if tid not in counted_ids and _is_valid_traversal(track, width, exit_margin, min_track_length):
                        final_class = _decide_track_class(track["class_history"])
                        fish_counts[final_class] += 1
                        counted_ids.add(tid)
                        track["status"] = f"counted_{final_class}"

                    # CSV row
                    csv_writer.writerow([
                        frame_id, tid, cname, f"{conf:.3f}",
                        f"{x_c:.2f}", f"{y_c:.2f}", track["status"],
                        f"first_side={track['first_side']} last_side={track['last_side']}",
                    ])

            # Update absent counters
            for missing_tid, track in tracks.items():
                if missing_tid not in current_frame_ids and missing_tid not in counted_ids:
                    track["absent_frames"] += 1
                    if track["absent_frames"] >= absent_threshold:
                        track["status"] = "missing"

            # Draw zone lines + counts
            left_line = int(width * entry_margin)
            right_line = int(width * (1 - entry_margin))
            cv2.line(annotated_frame, (left_line, 0), (left_line, height), (255, 255, 0), 2)
            cv2.line(annotated_frame, (right_line, 0), (right_line, height), (255, 255, 0), 2)

            herring_count = fish_counts.get("Herring", 0)
            non_herring_count = sum(c for n, c in fish_counts.items() if n != "Herring")
            cv2.putText(annotated_frame, f"Herring: {herring_count} | Frame: {frame_id}",
                        (width - 350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Non-herring: {non_herring_count}",
                        (width - 350, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            writer.write(annotated_frame)

            if frame_id % 100 == 0:
                print(f"  Frame {frame_id}/{total_frames}  | Herring: {herring_count} | Non-herring: {non_herring_count}")

            if not self.config.no_display:
                cv2.imshow("Zone Counting", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        writer.release()
        csv_file.close()
        cv2.destroyAllWindows()

        return dict(fish_counts)
