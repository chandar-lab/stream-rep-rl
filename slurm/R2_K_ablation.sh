#!/bin/bash
#SBATCH --job-name=R2_K_ablation
#SBATCH --time=55:00:00
#SBATCH --ntasks=36
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/R2_K_ablation_%j.out
#SBATCH --error=logs/R2_K_ablation_%j.err

# Rebuttal Experiment 2: K=0 and K=10 ablations for QRC+SPR+orth on Atari-6.
# K=0: no transition network, representation regularizer only.
# K=10: extended prediction horizon.

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/common.sh"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
fi

# K=0 ablation (6 envs x 3 seeds = 18 tasks)
launch_runs "qrc-spr-orth.py" ATARI_6 \
    --exp_class qrc-spr-K0 --exp_name qrc-spr-K0 \
    --spr_prediction_depth 0

# K=10 ablation (6 envs x 3 seeds = 18 tasks)
launch_runs "qrc-spr-orth.py" ATARI_6 \
    --exp_class qrc-spr-K10 --exp_name qrc-spr-K10 \
    --spr_prediction_depth 10

wait
echo "All R2 K-ablation tasks completed."
