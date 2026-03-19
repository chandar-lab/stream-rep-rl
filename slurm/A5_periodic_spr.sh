#!/bin/bash
# A5: SPR every K steps
# Sweep spr_update_freq in {1, 5, 10, 20} on Atari 4, 3 seeds (MinAtar disabled)
# Total: 4 freqs x 4 games x 3 seeds = 48 tasks
# Split into separate jobs per freq for manageability

#SBATCH --job-name=A5_periodic_spr
#SBATCH --time=55:00:00
#SBATCH --ntasks=12
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

EXP_CLASS="rebuttal-A5"
FREQ=${1:-5}  # Pass spr_update_freq as arg, default 5

# MinAtar - DISABLED
# launch_runs_minatar "qrc-spr-orth-periodic.py" MINATAR_5 \
#     --exp_class "${EXP_CLASS}-freq${FREQ}" \
#     --spr_update_freq "$FREQ"

# Atari
launch_runs "qrc-spr-orth-periodic.py" ATARI_4 \
    --exp_class "${EXP_CLASS}-freq${FREQ}" \
    --spr_update_freq "$FREQ"

wait
echo "A5 (freq=$FREQ): All experiments completed."
