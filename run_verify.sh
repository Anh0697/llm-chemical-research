#!/bin/bash
#SBATCH --job-name=verify_fix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs_verify_%j.out

module load anaconda
module load cuda/11.8
source ~/.bashrc
conda activate chemical
export HF_HOME=/work/ock/anhnguyen/hf_cache
python verify_padding_fix.py
