# Testing & Evaluation Guide

This document covers how to run the evaluation framework to benchmark the fish counting algorithms on test datasets.

---

## Test Video Downloads

Test videos are stored on OneDrive (not in the git repo due to file size):

**Download link:** [test-video (OneDrive)](https://northeastern-my.sharepoint.com/:f:/r/personal/tengli_a_northeastern_edu/Documents/test-video?csf=1&web=1&e=kd8xxK)

After downloading, place all `.mp4` files in `test_scripts/test-videos/`:

```
Fish-detection-and-counting/
  test_scripts/
    test-videos/
      count5.mp4
      nemasket_normal 2.mp4
      nemasket_huge 4- 2.mp4
      nemasket_exta_huge 4.mp4
      non-herring-easy.mp4
```

The filenames must match exactly what is referenced in `datasets.yaml`. If you add a new video, add a matching entry in `datasets.yaml` (see [Adding a New Test Video](#adding-a-new-test-video) below).

### Uploading test videos to NEU RC cluster

From your local machine:

```bash
# Step 1: Create the directory on the cluster
ssh <username>@login.explorer.northeastern.edu \
    "mkdir -p ~/alaska/ml/Fish-detection-and-counting/test_scripts/test-videos"

# Step 2: Upload all test videos
scp test_scripts/test-videos/*.mp4 \
    <username>@login.explorer.northeastern.edu:~/alaska/ml/Fish-detection-and-counting/test_scripts/test-videos/
```

To upload model weights (if not already on the cluster):
```bash
scp model/train113_weights.pt \
    <username>@login.explorer.northeastern.edu:~/alaska/ml/Fish-detection-and-counting/model/
```

To verify files are in place on the cluster:
```bash
ssh <username>@login.explorer.northeastern.edu \
    "ls ~/alaska/ml/Fish-detection-and-counting/test_scripts/test-videos/"
```

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

| Dataset | Type | Format | Video File | Description |
|---------|------|--------|------------|-------------|
| count5 | counting | video | `count5.mp4` | Short clip, 960x720, 539 frames |
| nemasket_normal | counting | video | `nemasket_normal 2.mp4` | Normal density, 426x240, 512 frames |
| nemasket_huge | counting | video | `nemasket_huge 4- 2.mp4` | High density, 426x240, 543 frames |
| nemasket_extra_huge | counting | video | `nemasket_exta_huge 4.mp4` | Very high density, 426x240, 601 frames |
| non_herring_easy | counting | video | `non-herring-easy.mp4` | Non-herring species, 1280x720, 519 frames, grayscale |
| herring_hard | detection | coco | — | 76 labeled frames |
| herring_easy | detection | coco | — | 89 labeled frames |
| roboflow | detection | yolo | — | 1720 test images |

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

## Adaptive Track Filtering

The framework uses adaptive thresholds that normalize across different video FPS, resolutions, and fish speeds. These apply to **all three algorithms** automatically.

### How it works

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `min_track_time` | **0.15s** | Converts to frame count using the video's actual FPS. At 30fps = 4 frames, at 60fps = 9 frames. Replaces the old fixed `min_track_len`. |
| `min_detection_ratio` | **0.30** | Track must be detected in at least 30% of the frames it was present. Normalizes for fish speed — a fast fish (8 frames, 6 detected = 75%) and a slow fish (200 frames, 150 detected = 75%) are treated equally. Filters out flickering noise. |
| `min_track_distance` | **0.0** (disabled) | Minimum horizontal displacement as fraction of frame width. Disabled by default because it's camera-dependent. |

### Why adaptive instead of fixed frame counts

A fixed threshold like `min_track_len=10` means different things depending on the video:
- At 30fps, 10 frames = 0.33 seconds
- At 60fps, 10 frames = 0.17 seconds
- A fast fish might only be visible for 8 frames even though it's real
- A slow fish gets 200 frames easily

The adaptive approach uses **time** (works across FPS) and **detection ratio** (works across fish speeds) so no per-video tuning is needed.

### Example output
```
Resolution: 960x720, FPS: 30.0, Frames: 539
Adaptive min_track_len: 4 frames (from min_track_time=0.15s at 30.0fps)
Adaptive min_detection_ratio: 30% (track must be detected in 30% of frames present)
```

### Overriding per-dataset

If a specific video needs different thresholds, override in `datasets.yaml`:
```yaml
my_tricky_video:
  type: counting
  format: video
  video_path: test_scripts/test-videos/tricky.mp4
  min_track_time: 0.3              # stricter: 0.3 seconds
  min_detection_ratio: 0.5         # stricter: 50% detection rate required
  min_track_distance: 0.10         # enable: fish must travel 10% of frame width
```

---

## Running Evaluations

All commands are run from the project root (`Fish-detection-and-counting/`).

### Quick start — run all algorithms on all videos
```bash
python eval.py count \
    -d count5 nemasket_normal nemasket_huge nemasket_extra_huge non_herring_easy \
    -w model/train113_weights.pt \
    --algorithm all \
    --no-display \
    --device cuda 2>&1 | tee eval_results.txt
```

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

### Save output to a log file while viewing in terminal
```bash
python eval.py count \
    -d count5 \
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

### Step 3: Pull latest code
```bash
git pull origin alaska/validation-testing
```

### Step 4: Verify test videos and model weights are in place
```bash
ls test_scripts/test-videos/
ls model/train113_weights.pt
```

If missing, upload from your local machine (see [Uploading test videos](#uploading-test-videos-to-neu-rc-cluster) above).

### Step 5: Request an interactive GPU session
```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=10 --mem=12G --time=01:55:00 --pty /bin/bash
```

If V100 is specifically needed:
```bash
srun --partition=gpu --gres=gpu:v100-pcie:1 --cpus-per-task=10 --mem=12G --time=01:55:00 --pty /bin/bash
```

Note: you may need to wait in the queue. Check status with `squeue -u <username>` from another terminal.

### Step 6: Activate the environment and verify GPU
```bash
conda activate yolov11
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Step 7: Run evaluation
```bash
python eval.py count \
    -d count5 nemasket_normal nemasket_huge nemasket_extra_huge non_herring_easy \
    -w model/train113_weights.pt \
    --algorithm all \
    --no-display \
    --device cuda 2>&1 | tee eval_results.txt
```

### Step 8 (alternative): Submit as a batch job
```bash
sbatch hpc/hpc.sh
```

Check job status:
```bash
squeue -u <username>
```

View output after completion:
```bash
cat output_<job_id>.txt
```

### Step 9: Download results to local machine
From your local machine:
```bash
scp -r <username>@login.explorer.northeastern.edu:~/alaska/ml/Fish-detection-and-counting/runs/eval ./eval_results/
```

---

## Output Structure

Results are written to `runs/eval/`:

```
runs/eval/
  {dataset_name}/
    {weight_stem}/
      {video}_{weight}.mp4             # annotated video (original algorithm)
      {video}_{weight}.csv             # per-frame tracking CSV
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
| `--min-track-len` | 10 | Min track length for counting (original algorithm only, overridden by min_track_time) |
| `--grayscale` | false | Convert frames to grayscale before inference |
| `--no-display` | false | Suppress OpenCV preview window |

### Adaptive filtering defaults (apply to all algorithms)

| Parameter | Default | Description |
|-----------|---------|-------------|
| min_track_time | 0.15 | Seconds. Converted to frame count using video FPS. Replaces fixed min_track_len. |
| min_detection_ratio | 0.30 | Track must be detected in at least 30% of frames it was present. |
| min_track_distance | 0.0 | Fraction of frame width. Disabled by default. |

### Zone/Stitch-specific defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| zone_entry_margin | 0.20 | Fraction of frame width for left/right entry zones |
| zone_exit_margin | 0.10 | Fraction of frame width for exit boundary |
| zone_min_track_length | 5 | Minimum frames a track must span to be counted |
| zone_absent_threshold | 5 | Consecutive missing frames before track is marked "missing" |

### Per-dataset overrides in datasets.yaml

Any parameter can be overridden per dataset. CLI flags take precedence over YAML values, which take precedence over defaults.

```yaml
count5:
  type: counting
  format: video
  video_path: test_scripts/test-videos/count5.mp4
  min_track_time: 0.3              # override adaptive time threshold
  min_detection_ratio: 0.5         # override detection ratio
  min_track_len: 2                 # override fixed frame count (original only)
  zone_entry_margin: 0.20          # zone/stitch: left/right zone width
  zone_exit_margin: 0.10           # zone/stitch: exit boundary width
  zone_min_track_length: 2         # zone/stitch: min frames for counting
  zone_absent_threshold: 5         # zone/stitch: frames before "missing"
  grayscale: true                  # convert to grayscale
```

---

## Adding a New Test Video

### Step 1: Place the video file
```bash
cp /path/to/my_video.mp4 test_scripts/test-videos/
```

### Step 2: Add an entry to `datasets.yaml`

The dataset name (key) can be anything. The `video_path` must match the filename exactly.

```yaml
  my_new_video:
    type: counting
    format: video
    video_path: test_scripts/test-videos/my_video.mp4
```

### Step 3: Run evaluation
```bash
python eval.py count -d my_new_video -w model/train113_weights.pt --algorithm all --no-display
```

No code changes required. Adaptive filtering applies automatically.

### Step 4: Upload to cluster (if testing on HPC)
From local machine:
```bash
scp test_scripts/test-videos/my_video.mp4 \
    <username>@login.explorer.northeastern.edu:~/alaska/ml/Fish-detection-and-counting/test_scripts/test-videos/
```

Make sure the same `datasets.yaml` entry exists on the cluster too (via git push/pull).

---

## Important Notes

- **Hardware affects results**: Different GPU architectures (MPS vs CUDA) produce slightly different floating-point results, which can change counts by a few fish. Always benchmark on the same hardware you'll deploy on.
- **Always use `--no-display` on HPC**: The cluster has no display server. Without this flag, OpenCV will crash.
- **Always use `--device cuda` on HPC**: Without it, inference runs on CPU which is much slower.
- **Test videos are not in git**: They're in `.gitignore` due to file size. Download from [OneDrive](https://northeastern-my.sharepoint.com/:f:/r/personal/tengli_a_northeastern_edu/Documents/test-video?csf=1&web=1&e=kd8xxK) and upload to the cluster manually.
- **Model weights are not in git**: Same reason. Keep them in `model/` locally and on the cluster.
