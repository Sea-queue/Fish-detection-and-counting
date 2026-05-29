"""
ZoneCountingEvaluator — zone-based traversal counting WITHOUT track stitching.

Algorithm (from without_tracking_prediction.ipynb):
  • Each frame: run model.track() with BotSort (my_botsort.yaml); collect all detections.
  • Zone classification: each detection's x-center is classified as
    "left" / "middle" / "right" based on ENTRY_MARGIN.
  • Valid traversal: a track that entered from one side (left/right) and
    exited the opposite side past EXIT_MARGIN, with sufficient track length.
  • Majority-vote class assignment across the track's per-frame detections.
  • Absent-frame tracking: marks tracks as "missing" after ABSENT_THRESHOLD
    consecutive frames without detection.

Outputs per (dataset, model) pair:
  - Annotated .mp4 with bounding boxes, zone lines, ID lists, and live count overlay
  - .csv with per-frame tracking details
  - Entry in the run's _summary/ JSON + CSV
"""

from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path

import csv
import cv2
from ultralytics import YOLO

from .base import Evaluator
from ..reporting.summary import MetricsSummary

# Path to custom tracker config (relative to project root)
TRACKER_CONFIG = "my_botsort.yaml"


def _get_zone(cx: float, frame_width: int, entry_margin: float) -> str:
    if cx < frame_width * entry_margin:
        return "left"
    elif cx > frame_width * (1 - entry_margin):
        return "right"
    return "middle"


def _is_valid_traversal(track: dict, frame_width: int, exit_margin: float,
                        min_track_length: int, min_dist_px: float = 0.0,
                        min_det_ratio: float = 0.0) -> bool:
    track_length = track["last_frame"] - track["first_frame"]
    if track_length < min_track_length:
        return False
    # Detection ratio: frames detected / total span
    if min_det_ratio > 0 and track_length > 0:
        ratio = len(track["positions"]) / (track_length + 1)
        if ratio < min_det_ratio:
            return False
    # Adaptive distance check
    if min_dist_px > 0:
        displacement = abs(track["positions"][-1][0] - track["positions"][0][0])
        if displacement < min_dist_px:
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


def _draw_id_list(frame, label: str, id_list, x: int, y_start: int,
                  color: tuple, max_per_line: int = 20, scale: float = 0.5) -> int:
    """Draw a labeled list of IDs on the frame. Returns the next y position."""
    y = y_start
    for i in range(0, max(1, len(id_list)), max_per_line):
        chunk = ", ".join(map(str, sorted(id_list)[i:i + max_per_line]))
        if i == 0:
            text = f"{label}: {chunk}" if chunk else f"{label}: (none)"
        else:
            text = chunk
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
        y += int(22 * scale / 0.5)
    return y + 10


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
        vid_writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        # CSV
        csv_file = open(out_csv_path, "w", newline="")
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

        # Scale text size based on resolution
        font_scale = max(0.35, min(0.6, width / 1280))
        count_scale = max(0.5, min(0.8, width / 960))

        # Adaptive thresholds
        min_track_time = self.config.min_track_time
        min_track_dist = self.config.min_track_distance
        if min_track_time > 0 and fps > 0:
            min_track_length = max(1, int(min_track_time * fps))
            print(f"  Adaptive min_track_length: {min_track_length} frames "
                  f"(from min_track_time={min_track_time}s at {fps:.1f}fps)")
        min_dist_px = min_track_dist * width if min_track_dist > 0 else 0.0
        if min_dist_px > 0:
            print(f"  Adaptive min_track_distance: {min_dist_px:.0f}px "
                  f"(from min_track_distance={min_track_dist} * {width}px)")
        min_det_ratio = self.config.min_detection_ratio
        if min_det_ratio > 0:
            print(f"  Adaptive min_detection_ratio: {min_det_ratio:.0%}")

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
                tracker=TRACKER_CONFIG,
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
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 1)

                    # Real-time counting check
                    if tid not in counted_ids and _is_valid_traversal(track, width, exit_margin, min_track_length, min_dist_px, min_det_ratio):
                        final_class = _decide_track_class(track["class_history"])
                        fish_counts[final_class] += 1
                        counted_ids.add(tid)
                        track["status"] = f"counted_{final_class}"

                    # CSV row (full track_info like notebook)
                    track_info = (
                        f"first_frame={track['first_frame']} "
                        f"last_frame={track['last_frame']} "
                        f"first_side={track['first_side']} "
                        f"last_side={track['last_side']} "
                        f"absent_frames={track['absent_frames']}"
                    )
                    csv_writer.writerow([
                        frame_id, tid, cname, f"{conf:.3f}",
                        f"{x_c:.2f}", f"{y_c:.2f}", track["status"], track_info,
                    ])

            # Update absent counters
            for missing_tid, track in tracks.items():
                if missing_tid not in current_frame_ids and missing_tid not in counted_ids:
                    track["absent_frames"] += 1
                    if track["absent_frames"] >= absent_threshold:
                        track["status"] = "missing"

            # ── Draw zone boundaries ──
            left_line = int(width * entry_margin)
            right_line = int(width * (1 - entry_margin))
            cv2.line(annotated_frame, (left_line, 0), (left_line, height), (255, 255, 0), 2)
            cv2.line(annotated_frame, (right_line, 0), (right_line, height), (255, 255, 0), 2)

            # ── Draw ID lists (top-left, matching notebook) ──
            y_pos = _draw_id_list(annotated_frame, "Counted IDs", counted_ids,
                                  10, 30, (0, 255, 255), scale=font_scale)

            entered_list = [
                tid for tid in tracks
                if tracks[tid]["status"] == "tracking"
                and len(tracks[tid]["positions"]) >= min_track_length
            ]
            y_pos = _draw_id_list(annotated_frame, "Entered IDs", entered_list,
                                  10, y_pos, (255, 255, 0), scale=font_scale)

            missing_list = [
                tid for tid in tracks
                if tracks[tid]["status"] == "missing"
                and len(tracks[tid]["positions"]) >= min_track_length
            ]
            _draw_id_list(annotated_frame, "Missing IDs", missing_list,
                          10, y_pos, (0, 0, 255), scale=font_scale)

            # ── Draw counts (top-right, adaptive positioning) ──
            herring_count = fish_counts.get("Herring", 0)
            non_herring_count = sum(c for n, c in fish_counts.items() if n != "Herring")
            count_x = max(10, width - int(300 * count_scale / 0.8))
            cv2.putText(annotated_frame, f"Herring: {herring_count} | Frame: {frame_id}",
                        (count_x, 30), cv2.FONT_HERSHEY_SIMPLEX, count_scale, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Non-herring: {non_herring_count}",
                        (count_x, 60), cv2.FONT_HERSHEY_SIMPLEX, count_scale, (0, 255, 0), 2)

            vid_writer.write(annotated_frame)

            if frame_id % 100 == 0:
                print(f"  Frame {frame_id}/{total_frames}  | Herring: {herring_count} | Non-herring: {non_herring_count}")

            if not self.config.no_display:
                cv2.imshow("Zone Counting", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        vid_writer.release()
        csv_file.close()
        cv2.destroyAllWindows()

        # ── Diagnostic summary ──
        total_tracks = len(tracks) + len(counted_ids)
        middle_entry = sum(1 for t in tracks.values() if t["first_side"] == "middle")
        same_side = sum(1 for tid, t in tracks.items()
                        if t["first_side"] == t["last_side"]
                        and t["first_side"] != "middle"
                        and tid not in counted_ids)
        too_short = sum(1 for t in tracks.values()
                        if (t["last_frame"] - t["first_frame"]) < min_track_length)
        no_exit = sum(1 for tid, t in tracks.items()
                      if t["first_side"] != t["last_side"]
                      and t["first_side"] != "middle"
                      and tid not in counted_ids
                      and not _is_valid_traversal(t, width, exit_margin, min_track_length, min_dist_px, min_det_ratio))

        print(f"\n  ── Zone Diagnostics ──")
        print(f"  Total unique tracks seen: {total_tracks}")
        print(f"  Counted (valid traversal): {len(counted_ids)}")
        print(f"  Rejected — entered from middle: {middle_entry}")
        print(f"  Rejected — exited same side as entry: {same_side}")
        print(f"  Rejected — track too short (<{min_track_length} frames): {too_short}")
        print(f"  Rejected — crossed sides but didn't reach exit boundary: {no_exit}")

        return dict(fish_counts)
