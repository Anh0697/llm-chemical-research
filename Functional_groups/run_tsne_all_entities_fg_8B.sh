#!/bin/bash
#SBATCH --job-name=tsne_all_fg_8B
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/work/ock/anhnguyen/llm-chemical-research/functional_groups
#SBATCH --output=logs_tsne_all_%j.out

module load anaconda
source ~/.bashrc
conda activate chemical

python tsne_all_entities_fg.py
