#!/bin/bash
#SBATCH --job-name=qwen3_layer
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=400GB
#SBATCH --cpus-per-task=32
#SBATCH --gpus=8
#SBATCH --partition=gpu-h200-141g-ellis


python layer.py