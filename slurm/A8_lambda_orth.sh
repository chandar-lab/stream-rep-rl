#!/bin/bash
# A8: Lambda-orth interaction
# Sweep lambda in {0.0, 0.4, 0.8, 0.95} with orth on (qrc-spr-orth) vs off (qrc-spr, but we use qrc-spr-orth --orth_beta 0 as proxy... actually no).
# For "orth off" we'd need to zero out orth. Simplest: just run qrc-spr-orth with different lambda values.
# The comparison with orth-off comes from existing qrc-spr runs or qrc runs.
# Atari 3 games, 3 seeds, 4 lambdas = 36 tasks

#SBATCH --job-name=A8_lambda_orth
#SBATCH --time=55:00:00
#SBATCH --ntasks=36
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs_slurm/%x-%j.out
#SBATCH --error=logs_slurm/%x-%j.err

source "$(dirname "$0")/common.sh"

EXP_CLASS="rebuttal-A8"
LAMBDAS=(0.0 0.4 0.8 0.95)

for LAMBDA in "${LAMBDAS[@]}"; do
    launch_runs "qrc-spr-orth.py" ATARI_3 \
        --exp_class "${EXP_CLASS}-lam${LAMBDA}" \
        --lamda "$LAMBDA"
done

wait
echo "A8: All experiments completed."
