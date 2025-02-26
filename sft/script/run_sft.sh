#!/bin/bash
#SBATCH --job-name=sft_qwen_3b
#SBATCH --output=sft_qwen_3b.out
#SBATCH --error=sft_qwen_3b.err
#SBATCH --partition=shire-general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16  # Adjusted to match the number of workers in your Python script
#SBATCH --gres=gpu:A100_80GB:2  # Requesting one A100 80GB GPU
#SBATCH --mem=64G  # Memory allocation as per your interactive command
#SBATCH --time=60:00:00  # Adjusted to match your interactive command

# Load necessary modules or activate environments
source /home/mkapadni/.bashrc  # Ensure this points to the correct file if different
conda activate gpu_env

# Navigate to the project directory
cd /home/mkapadni/work/crscore_plus_plus/sft

# Create the save directory if it doesn't exist and run the Python script


export NCCL_P2P_DISABLE=1 && export CUDA_VISIBLE_DEVICES=0 && ACCELERATE_LOG_LEVEL=info accelerate launch --config_file /home/mkapadni/work/crscore_plus_plus/sft/accelerate_configs/deepspeed_zero3.yaml /home/mkapadni/work/crscore_plus_plus/sft/train_sft.py /home/mkapadni/work/crscore_plus_plus/sft/sft_config/config_full.yaml