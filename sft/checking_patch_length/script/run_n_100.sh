#!/bin/bash
#SBATCH --job-name=n_100
#SBATCH --output=n_100.out
#SBATCH --error=n_100.err
#SBATCH --partition=shire-general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12  # Adjusted to match the number of workers in your Python script
#SBATCH --gres=gpu:A100_80GB:1  # Requesting one A100 80GB GPU
#SBATCH --mem=16G  # Memory allocation as per your interactive command
#SBATCH --time=12:00:00  # Adjusted to match your interactive command

# Load necessary modules or activate environments
source /home/mkapadni/.bashrc  # Ensure this points to the correct file if different
conda activate gpu_env

# Navigate to the project directory
cd /home/mkapadni/work/crscore_plus_plus/sft/checking_patch_length

# Create the save directory if it doesn't exist and run the Python script


python run_n.py --n=100