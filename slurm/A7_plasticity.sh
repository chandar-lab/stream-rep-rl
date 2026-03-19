#!/bin/bash
# A7: Plasticity baseline (dormant neuron reset)
# Compare QRC+plasticity vs QRC+SPR+orth.
# Atari 4 games, 3 seeds = 12 tasks

#SBATCH --job-name=A7_plasticity
#SBATCH --time=55:00:00
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs_slurm/%x-%j.out
#SBATCH --error=logs_slurm/%x-%j.err

source "$(dirname "$0")/common.sh"

EXP_CLASS="rebuttal-A7"

launch_runs "qrc-plasticity.py" ATARI_4 \
    --exp_class "$EXP_CLASS" \
    --plasticity_reset_freq 10000 \
    --plasticity_threshold 0.01

wait
echo "A7: All experiments completed."
