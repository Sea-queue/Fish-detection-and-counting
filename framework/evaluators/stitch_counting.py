"""
StitchCountingEvaluator — zone-based traversal counting WITH track stitching/ReID.

Algorithm (from yolo11_prediction.ipynb):
  • Same zone-based traversal logic as ZoneCountingEvaluator.
  • ADDS track stitching: when a new track appears, attempts to merge it
    with a recently-lost track using velocity-based prediction (or fallback
    net-direction + distance gate for short tracks).
  • ADDS noise garbage collection: short tracks (<=2 detections) that have
    been absent for 10+ frames are deleted.

This is the most sophisticated counting algorithm — it reduces double-counting
caused by tracker ID switches mid-traversal.

Outputs per (dataset, model) pair:
  - Annotated .mp4 with bounding boxes, zone lines, stitch info, and live count overlay
  - .csv with per-frame tracking details including stitch remapping
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


# ── Zone algorithm defaults (overridden by config) ──────────────────────────

# ── Stitching parameters ─────────────────────────────────────────────────────
REID_MAX_GAP = 90
REID_MAX_DY = 120
REID_PRED_SLACK = 250
REID_VELOCITY_LOOKBACK = 5
REID_MIN_OLD_LEN = 5
REID_FALLBACK_MAX_DX = 400
REID_NEW_TRACK_GRACE = 3

# ── Noise GC ─────────────────────────────────────────────────────────────────
NOISE_MAX_LEN = 2
NOISE_GC_AFTER = 10


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


def _estimate_velocity(positions: list[tuple], lookback: int = REID_VELOCITY_LOOKBACK):
    if len(positions) < 2:
        return 0.0, 0.0
    tail = positions[-lookback:] if len(positions) >= lookback else positions
    dx = (tail[-1][0] - tail[0][0]) / max(1, len(tail) - 1)
    dy = (tail[-1][1] - tail[0][1]) / max(1, len(tail) - 1)
    return dx, dy


def _try_stitch(
    new_tid: int,
    tracks: dict,
    counted_ids: set,
    frame_id: int,
    id_remap: dict,
) -> int | None:
    """Attempt to merge a freshly-born track into a recently-lost old track."""
    new_track = tracks[new_tid]
    if len(new_track["positions"]) > REID_NEW_TRACK_GRACE:
        return None

    nx, ny = new_track["positions"][0]
    best, best_score = None, float("inf")

    for old_tid, t in tracks.items():
        if old_tid == new_tid or old_tid in counted_ids or old_tid in id_remap:
            continue
        gap = frame_id - t["last_frame"]
        if gap <= 0 or gap > REID_MAX_GAP:
            continue

        ox, oy = t["positions"][-1]
        if abs(ny - oy) > REID_MAX_DY:
            continue

        dx = nx - ox

        if len(t["positions"]) >= REID_MIN_OLD_LEN:
            vx, vy = _estimate_velocity(t["positions"])
            if vx == 0 or vx * dx <= 0:
                continue
            predicted_x = ox + vx * gap
            pred_err = nx - predicted_x
            if abs(pred_err) > REID_PRED_SLACK:
                continue
            score = abs(pred_err) + abs(ny - oy) * 2 + gap * 1.0
        else:
            first_x = t["positions"][0][0]
            old_dir = ox - first_x
            if old_dir == 0 or old_dir * dx <= 0:
                continue
            if abs(dx) > REID_FALLBACK_MAX_DX:
                continue
            score = abs(dx) * 0.8 + abs(ny - oy) * 2 + gap * 1.5 + 50

        if score < best_score:
            best_score, best = score, old_tid

    return best


def _merge_tracks(old_tid: int, new_tid: int, tracks: dict, id_remap: dict):
    old = tracks[old_tid]
    new = tracks[new_tid]
    old["last_frame"] = new["last_frame"]
    old["last_side"] = new["last_side"]
    old["positions"].extend(new["positions"])
    old["class_history"].extend(new["class_history"])
    old["absent_frames"] = 0
    if old["status"] == "missing":
        old["status"] = "tracking"
    del tracks[new_tid]
    id_remap[new_tid] = old_tid


class StitchCountingEvaluator(Evaluator):

    def setup(self) -> None:
        entry = self.registry.get(self.config.dataset_name)
        if entry.get("type") != "counting":
            raise ValueError(
                f"Dataset '{self.config.dataset_name}' has type='{entry.get('type')}'. "
                "Use 'eval.py count' only with counting datasets."
            )
        self._video_path = self._require_file(entry["video_path"], label="Video file")
        print(f"  [stitch-count] Video → {self._video_path}")

    def run(self, weights_path: str) -> dict:
        self._require_file(weights_path, label="Weights file")
        run_dir = self._make_run_dir(weights_path)

        video_stem = self._video_path.stem
        weight_stem = Path(weights_path).stem
        out_video = run_dir / f"{video_stem}_{weight_stem}_stitch.mp4"
        out_csv = run_dir / f"{video_stem}_{weight_stem}_stitch.csv"

        model = YOLO(weights_path)
        counts, stitch_count = self._run_tracking(model, out_video, out_csv)

        result = {
            "dataset": self.config.dataset_name,
            "weights": weights_path,
            "algorithm": "stitch",
            "video": str(self._video_path),
            "counts": counts,
            "stitches_performed": stitch_count,
            "output_video": str(out_video),
            "output_csv": str(out_csv),
            "output_dir": str(run_dir),
        }

        print(f"\n  --- Counts (stitch): {weight_stem} on {self.config.dataset_name} ---")
        for cls_name, cnt in sorted(counts.items()):
            print(f"  {cls_name}: {cnt}")
        print(f"  Stitches performed: {stitch_count}")
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
    ) -> tuple[dict[str, int], int]:
        """Returns (counts_dict, number_of_stitches)."""
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

        import csv
        csv_file = open(out_csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame_id", "track_id", "raw_track_id", "class_name", "confidence",
            "center_x", "center_y", "status", "track_info",
        ])

        # ── Read margins from config ──
        entry_margin = self.config.zone_entry_margin
        exit_margin = self.config.zone_exit_margin
        absent_threshold = self.config.zone_absent_threshold
        min_track_length = self.config.zone_min_track_length

        print(f"  Stitch params: entry_margin={entry_margin}, exit_margin={exit_margin}, "
              f"absent_threshold={absent_threshold}, min_track_length={min_track_length}")

        # ── Tracking state ──
        tracks: dict[int, dict] = {}
        counted_ids: set[int] = set()
        fish_counts: dict[str, int] = defaultdict(int)
        id_remap: dict[int, int] = {}
        total_stitches = 0

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

                for (x_c, y_c, w, h), raw_tid, conf, cid in zip(xywh, ids, confidences, class_ids):
                    cname = model.names[cid]
                    current_side = _get_zone(x_c, width, entry_margin)
                    tid = id_remap.get(raw_tid, raw_tid)

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

                        # Try stitching into a recently-lost track
                        merge_into = _try_stitch(tid, tracks, counted_ids, frame_id, id_remap)
                        if merge_into is not None:
                            _merge_tracks(merge_into, tid, tracks, id_remap)
                            tid = merge_into
                            total_stitches += 1
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

                    if raw_tid != tid:
                        label = f"ID:{tid}(<-{raw_tid}) {cname} {conf:.2f}"
                    else:
                        label = f"ID:{tid} {cname} {conf:.2f}"
                    cv2.putText(annotated_frame, label,
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                    # Real-time counting check
                    if tid not in counted_ids and _is_valid_traversal(track, width, exit_margin, min_track_length):
                        final_class = _decide_track_class(track["class_history"])
                        fish_counts[final_class] += 1
                        counted_ids.add(tid)
                        track["status"] = f"counted_{final_class}"

                    # CSV row
                    csv_writer.writerow([
                        frame_id, tid, raw_tid, cname, f"{conf:.3f}",
                        f"{x_c:.2f}", f"{y_c:.2f}", track["status"],
                        f"first_side={track['first_side']} last_side={track['last_side']}",
                    ])

            # Update absent counters + noise GC
            stale_noise = []
            for missing_tid, track in tracks.items():
                if missing_tid not in current_frame_ids and missing_tid not in counted_ids:
                    track["absent_frames"] += 1
                    if track["absent_frames"] >= absent_threshold:
                        track["status"] = "missing"
                    if (len(track["positions"]) <= NOISE_MAX_LEN
                            and track["absent_frames"] >= NOISE_GC_AFTER):
                        stale_noise.append(missing_tid)

            for tid_del in stale_noise:
                del tracks[tid_del]
            if stale_noise:
                stale_set = set(stale_noise)
                for raw, canonical in list(id_remap.items()):
                    if canonical in stale_set or raw in stale_set:
                        del id_remap[raw]

            # Draw zone lines + counts
            left_line = int(width * entry_margin)
            right_line = int(width * (1 - entry_margin))
            cv2.line(annotated_frame, (left_line, 0), (left_line, height), (255, 255, 0), 2)
            cv2.line(annotated_frame, (right_line, 0), (right_line, height), (255, 255, 0), 2)

            herring_count = fish_counts.get("Herring", 0)
            non_herring_count = sum(c for n, c in fish_counts.items() if n != "Herring")
            cv2.putText(annotated_frame, f"Herring: {herring_count} | Frame: {frame_id}",
                        (width - 350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Non-herring: {non_herring_count} | Stitches: {total_stitches}",
                        (width - 350, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            vid_writer.write(annotated_frame)

            if frame_id % 100 == 0:
                print(f"  Frame {frame_id}/{total_frames}  | Herring: {herring_count} | Non-herring: {non_herring_count} | Stitches: {total_stitches}")

            if not self.config.no_display:
                cv2.imshow("Stitch Counting", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        vid_writer.release()
        csv_file.close()
        cv2.destroyAllWindows()

        # ── Diagnostic summary ──
        total_tracks = len(tracks) + len(counted_ids)
        middle_entry = sum(1 for t in tracks.values() if t["first_side"] == "middle")
        same_side = sum(1 for t in tracks.values()
                        if t["first_side"] == t["last_side"]
                        and t["first_side"] != "middle"
                        and t not in counted_ids)
        too_short = sum(1 for t in tracks.values()
                        if (t["last_frame"] - t["first_frame"]) < min_track_length)
        no_exit = sum(1 for t in tracks.values()
                      if t["first_side"] != t["last_side"]
                      and t["first_side"] != "middle"
                      and not _is_valid_traversal(t, width, exit_margin, min_track_length))

        print(f"\n  ── Stitch Diagnostics ──")
        print(f"  Total unique tracks seen: {total_tracks}")
        print(f"  Counted (valid traversal): {len(counted_ids)}")
        print(f"  Stitches performed: {total_stitches}")
        print(f"  Rejected — entered from middle: {middle_entry}")
        print(f"  Rejected — exited same side as entry: {same_side}")
        print(f"  Rejected — track too short (<{min_track_length} frames): {too_short}")
        print(f"  Rejected — crossed sides but didn't reach exit boundary: {no_exit}")

        return dict(fish_counts), total_stitches
