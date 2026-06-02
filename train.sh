#!/bin/bash
#SBATCH --job-name=earnings-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

source venv/bin/activate
python scripts/train.py \
    --epochs 5 \
    --batch_size 16 \
    --lr 2e-5 \
    --return_window 3 \
    --output_dir checkpoints/
