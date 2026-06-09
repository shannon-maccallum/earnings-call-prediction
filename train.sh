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
mkdir -p logs checkpoints
python train_model.py \
    --data_dir notebooks/data/transcripts \
    --output_dir checkpoints/
