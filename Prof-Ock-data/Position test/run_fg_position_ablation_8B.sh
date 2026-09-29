#!/bin/bash
#SBATCH --job-name=fg_position_ablation_8B
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=fg_position_ablation_8B-%j.out
#SBATCH --error=fg_position_ablation_8B-%j.err

set -euo pipefail

# Override PROJECT_ROOT at submission time if the project lives elsewhere:
#   sbatch --export=ALL,PROJECT_ROOT=/different/path run_fg_position_ablation_8B.sh
PROJECT_ROOT="${PROJECT_ROOT:-/lustre/work/ock/khang/llm-chemical-research/Functional_groups}"
PYTHON_BIN="/mnt/nrdstor/ock/khang/conda-envs/research/bin/python"

EXTRACT_SCRIPT="$PROJECT_ROOT/extract_activations.py"
CONFIG_FILE="$PROJECT_ROOT/config_extract_fg_position_ablation_8B.yaml"
DATASET_FILE="$PROJECT_ROOT/functional_group_position_ablation_24mol.csv"
PROBE_SCRIPT="$PROJECT_ROOT/position_probe_fg_ablation.py"
ACTIVATIONS_ROOT="$PROJECT_ROOT/activation_datasets_fg_position_ablation/meta-llama-Meta-Llama-3.1-8B"
OUTPUT_DIR="$PROJECT_ROOT/Results/fg_position_ablation_probe_8B"

# Set RUN_EXTRACTION=0 to rerun only the probe using existing activations.
RUN_EXTRACTION="${RUN_EXTRACTION:-1}"
RUN_PROBE="${RUN_PROBE:-1}"

for required_file in \
    "$EXTRACT_SCRIPT" \
    "$CONFIG_FILE" \
    "$DATASET_FILE" \
    "$PROBE_SCRIPT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: Required file was not found: $required_file"
        exit 1
    fi
done

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python interpreter was not found: $PYTHON_BIN"
    exit 1
fi

cd "$PROJECT_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

echo "Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Working directory: $(pwd)"
echo "Python: $PYTHON_BIN"

if [[ "$RUN_EXTRACTION" == "1" ]]; then
    echo "Starting activation extraction for four prompt conditions."
    "$PYTHON_BIN" -u "$EXTRACT_SCRIPT" --config "$CONFIG_FILE"
else
    echo "Skipping activation extraction because RUN_EXTRACTION=$RUN_EXTRACTION."
fi

if [[ "$RUN_PROBE" == "1" ]]; then
    if [[ ! -d "$ACTIVATIONS_ROOT" ]]; then
        echo "ERROR: Activation directory was not found: $ACTIVATIONS_ROOT"
        exit 1
    fi
    echo "Starting matched-pair-grouped position probe."
    "$PYTHON_BIN" -u "$PROBE_SCRIPT" \
        --csv "$DATASET_FILE" \
        --activations-root "$ACTIVATIONS_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --templates-per-molecule 10 \
        --folds 4 \
        --pca-components 20 \
        --group-by pair
    # No --layers argument is intentional: the probe uses every shared layer.
else
    echo "Skipping the position probe because RUN_PROBE=$RUN_PROBE."
fi

echo "Functional-group position ablation experiment finished successfully."
