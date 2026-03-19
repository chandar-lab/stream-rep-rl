#!/bin/bash
# A6: Ortho vs EMA of ObGD grads
# Project SPR against EMA of past ObGD updates.
# MinAtar 5 games, 3 seeds = 15 tasks - DISABLED

#SBATCH --job-name=A6_ema_obgd
#SBATCH --time=55:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs_slurm/%x-%j.out
#SBATCH --error=logs_slurm/%x-%j.err

source "$(dirname "$0")/common.sh"

EXP_CLASS="rebuttal-A6"

# MinAtar runs disabled
# launch_runs_minatar "streamq-spr-ortho.py" MINATAR_5 \
#     --exp_class "$EXP_CLASS" \
#     --use_ema_obgd_orth

wait
echo "A6: All experiments completed."
