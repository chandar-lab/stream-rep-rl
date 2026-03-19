#!/bin/bash
# A2: DQN+SPR+orth
# Tests whether orth is general or QRC-specific.
# Atari 6 games, 3 seeds = 18 tasks

#SBATCH --job-name=A2_dqn_spr_orth
#SBATCH --time=55:00:00
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs_slurm/%x-%j.out
#SBATCH --error=logs_slurm/%x-%j.err

source "$(dirname "$0")/common.sh"

EXP_CLASS="rebuttal-A2"

launch_runs "dqn-spr-orth.py" ATARI_6 \
    --exp_class "$EXP_CLASS"

wait
echo "A2: All experiments completed."
