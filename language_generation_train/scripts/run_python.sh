#!/bin/bash
#SBATCH --job-name=run_python
#SBATCH --output=run_python.out
#SBATCH --error=run_python.err
#SBATCH --partition=shire-general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12  # Adjusted to match the number of workers in your Python script
#SBATCH --gres=gpu:A6000:1  # Requesting one A100 80GB GPU
#SBATCH --mem=16G  # Memory allocation as per your interactive command
#SBATCH --time=24:00:00  # Adjusted to match your interactive command

# Load necessary modules or activate environments
source /home/mkapadni/.bashrc  # Ensure this points to the correct file if different
conda activate gpu_env

# Navigate to the project directory
cd /home/mkapadni/work/crscore_plus_plus/language_generation_train

# Create the save directory if it doesn't exist and run the Python script


python check_python.py
# python check_java.py
# python check_js.py
