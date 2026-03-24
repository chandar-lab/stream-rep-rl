#!/bin/bash
#SBATCH --job-name=R6_qrc_sweep
#SBATCH --time=55:00:00
#SBATCH --ntasks=90
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/R6_qrc_sweep_%j.out
#SBATCH --error=logs/R6_qrc_sweep_%j.err

# Rebuttal Experiment 6: QRC(lambda) hyperparameter sweep on Atari-6.
# Reduced grid: 5 configs x 6 envs x 3 seeds = 90 tasks.
# Shows that the QRC baseline was fairly tuned.

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/common.sh"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
fi

CONFIGS=(
    "5e-5 0.8"
    "1e-4 0.4"
    "1e-4 0.8"
    "1e-4 0.95"
    "2e-4 0.8"
)

for CONFIG in "${CONFIGS[@]}"; do
    read LR LAM <<< "$CONFIG"
    EXP_NAME="qrc-lr${LR}-lam${LAM}"

    for SEED in "${SEEDS[@]}"; do
        for ENV in "${ATARI_6[@]}"; do
            srun --ntasks=1 --cpus-per-task=4 --exclusive \
                python "algorithms/qrc.py" \
                --env_type atari \
                --env_id "$ENV" \
                --seed "$SEED" \
                --exp_name "$EXP_NAME" \
                --exp_class "$EXP_NAME" \
                --q_lr "$LR" \
                --lamda "$LAM" \
                --log_dir "$CHECKPOINT_DIR" \
                --track \
                --resume \
                --total_timesteps 10000000 \
                --explore_frac 0.10 &
        done
    done
done

wait
echo "All R6 QRC sweep tasks completed."
