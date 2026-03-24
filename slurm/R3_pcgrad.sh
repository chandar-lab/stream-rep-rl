#!/bin/bash
#SBATCH --job-name=R3_pcgrad
#SBATCH --time=55:00:00
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/R3_pcgrad_%j.out
#SBATCH --error=logs/R3_pcgrad_%j.err

# Rebuttal Experiment 3: PCGrad comparison for Stream Q(lambda)+SPR.
# Replaces orth2 with PCGrad (Yu et al., 2020) for gradient conflict resolution.
# 6 envs x 3 seeds = 18 tasks.

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/common.sh"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
fi

launch_runs "streamq-spr-pcgrad.py" ATARI_6 \
    --exp_class strq-spr-pcgrad --exp_name strq-spr-pcgrad

wait
echo "All R3 PCGrad tasks completed."
