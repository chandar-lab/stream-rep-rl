#!/bin/bash
# A3: QRC+target net+SPR+orth
# Tests target net vs same-network bootstrapping with SPR+orth.
# 3 target_network_frequency values x Atari 6 = 18 tasks per freq (MinAtar disabled)
# Total: 3 freqs x 18 = 54

#SBATCH --job-name=A3_target_net
#SBATCH --time=55:00:00
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs_slurm/%x-%j.out
#SBATCH --error=logs_slurm/%x-%j.err

source "$(dirname "$0")/common.sh"

EXP_CLASS="rebuttal-A3"
FREQ=${1:-1000}  # Pass target_network_frequency as arg, default 1000

# MinAtar - DISABLED
# launch_runs_minatar "qrc-spr-orth-target.py" MINATAR_5 \
#     --exp_class "${EXP_CLASS}-freq${FREQ}" \
#     --target_network_frequency "$FREQ"

# Atari
launch_runs "qrc-spr-orth-target.py" ATARI_6 \
    --exp_class "${EXP_CLASS}-freq${FREQ}" \
    --target_network_frequency "$FREQ"

wait
echo "A3 (freq=$FREQ): All experiments completed."
