#!/bin/bash
#SBATCH --job-name=direct_recall_fg_8B
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/ock/anhnguyen/llm-chemical-research/functional_groups
#SBATCH --output=logs_direct_recall_%j.out

module load anaconda
source ~/.bashrc
conda activate chemical

python Direct_recall/direct_recall_fg.py
