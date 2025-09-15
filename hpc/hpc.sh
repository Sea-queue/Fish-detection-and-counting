#!/bin/bash
#SBATCH --job-name=trraficc             # Job name
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-pcie:1        # gpu:v100-sxm2:1   gpu:h200 
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --cpus-per-task=10            # Number of CPUs per task
#SBATCH --mem=12G                     # Total CPU memory (not GPU memory)
#SBATCH --time=01:55:00              # Time limit hh:mm:ss
#SBATCH --output=output_%j.txt       # Standard output and error log
#SBATCH --error=error_%j.txt         # Standard error log

# Load any required modules
module load cuda/11.0
conda activate yolov11


# training
# yolo detect train data='/work/CompVision/seaqueue/util/fish.yaml' model='/work/CompVision/seaqueue/yolov11/runs/train/train55/weights/best.pt' name='train59' epochs=200 imgsz=640 batch=32 lr0=0.005 lrf=0.05 momentum=0.999 optimizer='AdamW' weight_decay=0.01 dropout=0.5 project=/work/CompVision/seaqueue/yolov11/runs/train
# python train.py

# validating
# python validate.py

# predicting
# python predict.py

# tracking 
python track.py

# counting
# python counting.py

# dubugging confusion matrix
# python debug_confusion_matrix.py
