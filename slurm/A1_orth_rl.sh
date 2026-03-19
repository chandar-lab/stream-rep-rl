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

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/common.sh"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
fi

EXP_CLASS="rebuttal-A1"

# MinAtar runs (fast sanity check) - DISABLED
# launch_runs_minatar "qrc-spr-orth-rl.py" MINATAR_5 \
#     --exp_class "$EXP_CLASS"

# Atari runs
launch_runs "qrc-spr-orth-rl.py" ATARI_6 \
    --exp_class "$EXP_CLASS"

wait
echo "A1: All experiments completed."
