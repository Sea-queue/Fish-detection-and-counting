# Testing & Evaluation Guide

This document covers how to run the evaluation framework to benchmark the fish counting algorithms on test datasets.

---

## Prerequisites

- Python 3.10+ with the following packages:
  - `ultralytics` (YOLO11)
  - `opencv-python`
  - `torch` (with CUDA support for GPU)
  - `pyyaml`
- Model weights file (e.g., `model/train113_weights.pt`)
- Test videos placed in `test_scripts/test-videos/`

### Setting up the environment

```bash
# Option 1: Using the project virtual environment (local)
cd Fish-detection-and-counting
source yolo-eval/bin/activate

# Option 2: Using conda on NEU RC cluster
conda activate yolov11
```

---

## Available Test Datasets

Defined in `datasets.yaml`:

| Dataset | Type | Format | Description |
|---------|------|--------|-------------|
| count5 | counting | video | Short clip, 960x720, 539 frames |
| nemasket_normal | counting | video | Normal density, 426x240, 512 frames |
| nemasket_huge | counting | video | High density, 426x240, 543 frames |
| nemasket_extra_huge | counting | video | Very high density, 426x240, 601 frames |
| non_herring_easy | counting | video | Non-herring species, 1280x720, 519 frames |
| herring_hard | detection | coco | 76 labeled frames |
| herring_easy | detection | coco | 89 labeled frames |
| roboflow | detection | yolo | 1720 test images |

---

## Counting Algorithms

The framework supports three counting algorithms selectable via `--algorithm`:

### 1. Original (`--algorithm original`)
- Direction-of-travel exit detection
- No entry-side requirement — counts fish regardless of where they first appeared
- Confidence threshold (0.70) + majority vote for class assignment
- Best general-purpose algorithm for varied camera angles

### 2. Zone (`--algorithm zone`)
- Divides frame into left (0-20%), middle (20-80%), right (80-100%) zones
- Requires fish to enter from one side and exit the opposite side
- Fish entering from middle zone are not counted
- Designed for fish-ladder cameras with horizontal traversal
- Source: `hpc/without_tracking_prediction.ipynb`

### 3. Stitch (`--algorithm stitch`)
- Same zone-based counting as above
- Adds track stitching: merges fragmented tracker IDs using velocity-based prediction
- Adds noise garbage collection for short-lived tracks
- Uses custom BotSort config (`my_botsort.yaml`) with track_buffer=90
- Source: `hpc/yolo11_prediction.ipynb`

---

## Running Evaluations

All commands are run from the project root (`Fish-detection-and-counting/`).

### List all available datasets
```bash
python eval.py --list-datasets
```

### Run a single algorithm on one dataset
```bash
python eval.py count \
    -d count5 \
    -w model/train113_weights.pt \
    --algorithm original \
    --no-display \
    --device cuda
```

### Run all three algorithms on one dataset (comparison mode)
```bash
python eval.py count \
    -d count5 \
    -w model/train113_weights.pt \
    --algorithm all \
    --no-display \
    --device cuda
```

### Run all algorithms on all counting datasets
```bash
python eval.py count \
    -d count5 nemasket_normal nemasket_huge nemasket_extra_huge non_herring_easy \
    -w model/train113_weights.pt \
    --algorithm all \
    --no-display \
    --device cuda
```

### Save output to a log file while viewing in terminal
```bash
python eval.py count \
    -d count5 nemasket_normal nemasket_huge nemasket_extra_huge non_herring_easy \
    -w model/train113_weights.pt \
    --algorithm all \
    --no-display \
    --device cuda 2>&1 | tee eval_results.txt
```

### Run detection evaluation
```bash
python eval.py detect \
    -d herring_hard herring_easy roboflow \
    -w model/train113_weights.pt \
    --device cuda
```

### Compare multiple model weights
```bash
python eval.py count \
    -d count5 \
    -w model/train110.pt model/train113.pt model/train113_weights.pt \
    --algorithm all \
    --no-display \
    --device cuda
```

---

## Running on NEU RC Cluster (HPC)

### Step 1: SSH into the cluster
```bash
ssh <username>@login.explorer.northeastern.edu
```

### Step 2: Navigate to the project
```bash
cd ~/alaska/ml/Fish-detection-and-counting
```

### Step 3: Request an interactive GPU session
```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=10 --mem=12G --time=01:55:00 --pty /bin/bash
```

### Step 4: Activate the environment and run
```bash
conda activate yolov11
python eval.py count \
    -d count5 nemasket_normal nemasket_huge nemasket_extra_huge non_herring_easy \
    -w model/train113_weights.pt \
    --algorithm all \
    --no-display \
    --device cuda 2>&1 | tee eval_results.txt
```

### Step 5: Submit as a batch job (alternative)
```bash
sbatch hpc/hpc.sh
```

Check job status:
```bash
squeue -u <username>
```

---

## Output Structure

Results are written to `runs/eval/`:

```
runs/eval/
  {dataset_name}/
    {weight_stem}/
      {video}_{weight}_original.mp4    # annotated video (original algorithm)
      {video}_{weight}_original.csv    # per-frame tracking CSV
      {video}_{weight}_zone.mp4        # annotated video (zone algorithm)
      {video}_{weight}_zone.csv        # per-frame tracking CSV
      {video}_{weight}_stitch.mp4      # annotated video (stitch algorithm)
      {video}_{weight}_stitch.csv      # per-frame tracking CSV
    _summary/
      count_{timestamp}.json           # JSON summary
      count_{timestamp}.csv            # CSV summary
```

### Annotated Video Overlays

- **Bounding boxes**: Blue for side-entry fish, gray for middle-entry fish
- **Zone lines**: Yellow vertical lines showing entry zone boundaries
- **Counted IDs** (cyan, top-left): Track IDs that completed valid traversals
- **Entered IDs** (yellow, top-left): Currently active tracks
- **Missing IDs** (red, top-left): Tracks not seen for 5+ frames
- **Counts** (green, top-right): Herring and Non-herring counts, frame number
- **Stitch info**: ID remapping shown as `ID:5(<-12)` when tracks are merged

### CSV Columns

**Original algorithm**: `frame_id, track_id, confidence, class_name, x, y, w, h`

**Zone algorithm**: `frame_id, track_id, class_name, confidence, center_x, center_y, status, track_info`

**Stitch algorithm**: `frame_id, track_id, raw_track_id, class_name, confidence, center_x, center_y, status, track_info`

---

## Diagnostic Output

The zone and stitch algorithms print a diagnostic summary after each video showing why tracks were not counted:

```
── Zone Diagnostics ──
Total unique tracks seen: 94
Counted (valid traversal): 0
Rejected — entered from middle: 55
Rejected — exited same side as entry: 3
Rejected — track too short (<5 frames): 9
Rejected — crossed sides but didn't reach exit boundary: 36
```

| Rejection Reason | Meaning |
|-----------------|---------|
| Entered from middle | Fish first appeared in the middle 60% of the frame, not in a left/right entry zone |
| Exited same side as entry | Fish entered from one side but was last seen on the same side (turned around) |
| Track too short | Track lasted fewer frames than the minimum threshold |
| Crossed sides but didn't reach exit boundary | Fish moved from one side to the other but didn't reach the outermost 10% of the frame |

Note: These categories can overlap. A single track may fail multiple conditions.

---

## Configurable Parameters

### CLI flags (apply to all datasets)

| Flag | Default | Description |
|------|---------|-------------|
| `--algorithm` | original | Counting algorithm: original, zone, stitch, or all |
| `--device` | auto | Device: cuda, mps, or cpu |
| `--imgsz` | 640 | Inference image size |
| `--conf` | 0.25 | Detection confidence threshold |
| `--max-det` | 300 | Max detections per frame |
| `--exit-margin` | 0.35 | Exit zone fraction (original algorithm only) |
| `--count-conf` | 0.70 | Min confidence for counting (original algorithm only) |
| `--majority-ratio` | 0.70 | Min fraction for class vote (original algorithm only) |
| `--min-track-len` | 10 | Min track length for counting (original algorithm only) |
| `--grayscale` | false | Convert frames to grayscale before inference |
| `--no-display` | false | Suppress OpenCV preview window |

### Per-dataset overrides in datasets.yaml

Parameters can be overridden per dataset in `datasets.yaml`. CLI flags take precedence over YAML values, which take precedence over defaults.

```yaml
count5:
  type: counting
  format: video
  video_path: test_scripts/test-videos/count5.mp4
  min_track_len: 2                  # original algorithm
  zone_entry_margin: 0.20           # zone/stitch: left/right zone width
  zone_exit_margin: 0.10            # zone/stitch: exit boundary width
  zone_min_track_length: 2          # zone/stitch: min frames for counting
  zone_absent_threshold: 5          # zone/stitch: frames before "missing"
  grayscale: true                   # convert to grayscale
```

### Zone/Stitch-specific defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| zone_entry_margin | 0.20 | Fraction of frame width for left/right entry zones |
| zone_exit_margin | 0.10 | Fraction of frame width for exit boundary |
| zone_min_track_length | 5 | Minimum frames a track must span to be counted |
| zone_absent_threshold | 5 | Consecutive missing frames before track is marked "missing" |

---

## Adding a New Test Video

1. Place the video file in `test_scripts/test-videos/`

2. Add an entry to `datasets.yaml`:
```yaml
my_new_video:
  type: counting
  format: video
  video_path: test_scripts/test-videos/my_new_video.mp4
```

3. Run evaluation:
```bash
python eval.py count -d my_new_video -w model/train113_weights.pt --algorithm all --no-display
```

No code changes required.
