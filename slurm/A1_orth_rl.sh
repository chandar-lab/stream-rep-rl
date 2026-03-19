#!/bin/bash
# A1: Orth on RL gradients too
# Tests whether RL gradients should also be decorrelated.
# Atari 6 games, 3 seeds = 18 tasks (MinAtar disabled)

#SBATCH --job-name=A1_orth_rl
#SBATCH --time=55:00:00
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs_slurm/%x-%j.out
#SBATCH --error=logs_slurm/%x-%j.err

source "$(dirname "$0")/common.sh"

EXP_CLASS="rebuttal-A1"

# MinAtar runs (fast sanity check) - DISABLED
# launch_runs_minatar "qrc-spr-orth-rl.py" MINATAR_5 \
#     --exp_class "$EXP_CLASS"

# Atari runs
launch_runs "qrc-spr-orth-rl.py" ATARI_6 \
    --exp_class "$EXP_CLASS"

wait
echo "A1: All experiments completed."
