#!/bin/bash
#SBATCH --job-name=dtd-hopper-seeds
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --time=24:00:00
#SBATCH --array=0-44
#SBATCH --output=slurm_logs/hopper_seeds_%A_%a.out
#SBATCH --error=slurm_logs/hopper_seeds_%A_%a.err

set -euo pipefail

cd /cluster/home/fstarke/differential_TD

module purge
module load stack/2024-05
module load python/3.10.13_cuda

unset PYTHONHOME
unset PYTHONPATH

source .venv/bin/activate

export HYDRA_FULL_ERROR=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90

TDS=(baseline dtd dtd_full)
NOISES=(0.00 0.01 0.05)
SEEDS=(42 43 44 45 46)

SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
NOISE_INDEX=$(((SLURM_ARRAY_TASK_ID / 5) % 3))
TD_INDEX=$((SLURM_ARRAY_TASK_ID / 15))

export ENV_NAME=hopper
export TD="${TDS[$TD_INDEX]}"
export NOISE_LVL="${NOISES[$NOISE_INDEX]}"
export SEED="${SEEDS[$SEED_INDEX]}"
export RUN_TIME="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_seed${SEED}"
export MAX_BUDGET=2500000

bash dtd/runs/main.sh \
  algorithm.seed="${SEED}"
