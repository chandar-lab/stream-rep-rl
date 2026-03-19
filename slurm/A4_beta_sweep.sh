#!/bin/bash
# A4: Orth beta sensitivity
# Sweep beta in {0.9, 0.95, 0.99, 0.999} on Atari 4 games, 3 seeds
# Total: 4 betas x 4 games x 3 seeds = 48 tasks

#SBATCH --job-name=A4_beta_sweep
#SBATCH --time=55:00:00
#SBATCH --ntasks=48
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

EXP_CLASS="rebuttal-A4"
BETAS=(0.9 0.95 0.99 0.999)

for BETA in "${BETAS[@]}"; do
    launch_runs "qrc-spr-orth.py" ATARI_4 \
        --exp_class "${EXP_CLASS}-beta${BETA}" \
        --orth_beta "$BETA"
done

wait
echo "A4: All experiments completed."
