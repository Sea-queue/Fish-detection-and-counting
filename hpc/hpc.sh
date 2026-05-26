#!/bin/bash
#SBATCH --job-name=fish_eval            # Job name
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-pcie:1          # gpu:v100-sxm2:1   gpu:h200
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --cpus-per-task=10              # Number of CPUs per task
#SBATCH --mem=12G                       # Total CPU memory (not GPU memory)
#SBATCH --time=01:55:00                 # Time limit hh:mm:ss
#SBATCH --output=output_%j.txt          # Standard output and error log
#SBATCH --error=error_%j.txt            # Standard error log

# Load any required modules
module load cuda/11.0
conda activate yolov11

# Navigate to project root (where eval.py and datasets.yaml live)
cd ~/alaska/ml/Fish-detection-and-counting

# ============================================================================
# EVALUATION — run all 3 counting algorithms on test videos
# ============================================================================
# Usage: uncomment the section you want to run

# --- Run all 3 algorithms (original, zone, stitch) on a single video ---
# python eval.py count \
#     -d count5 \
#     -w model/train113_weights.pt \
#     --algorithm all \
#     --no-display \
#     --device cuda

# --- Run all 3 algorithms on ALL counting datasets ---
python eval.py count \
    -d count5 nemasket_normal nemasket_huge nemasket_extra_huge non_herring_easy \
    -w model/train113_weights.pt \
    --algorithm all \
    --no-display \
    --device cuda

# --- Detection evaluation ---
# python eval.py detect \
#     -d herring_hard herring_easy roboflow \
#     -w model/train113_weights.pt \
#     --device cuda

# ============================================================================
# LEGACY COMMANDS (kept for reference)
# ============================================================================
# python train.py
# python validate.py
# python predict.py
# python track.py
# python counting.py
