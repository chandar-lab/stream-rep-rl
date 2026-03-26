#!/bin/bash
#SBATCH --job-name=R7_stream_ac
#SBATCH --time=24:00:00
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/R7_stream_ac_%j.out
#SBATCH --error=logs/R7_stream_ac_%j.err

# Rebuttal Experiment 7: Continuous control with Stream AC +/- SPR.
# 4 envs x 3 seeds x 2 methods = 24 tasks, 2M steps each.

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/common.sh"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
fi

MUJOCO_ENVS=("HalfCheetah-v4" "Hopper-v4" "Walker2d-v4" "Ant-v4")

for SEED in "${SEEDS[@]}"; do
    for ENV in "${MUJOCO_ENVS[@]}"; do
        srun --ntasks=1 --cpus-per-task=4 --exclusive \
            python algorithms/stream_ac.py \
            --env_id "$ENV" --seed "$SEED" \
            --exp_name "stream-ac" --exp_class "stream-ac" \
            --log_dir "$CHECKPOINT_DIR" --track \
            --total_timesteps 2000000 &
        srun --ntasks=1 --cpus-per-task=4 --exclusive \
            python algorithms/stream_ac_spr.py \
            --env_id "$ENV" --seed "$SEED" \
            --exp_name "stream-ac-spr" --exp_class "stream-ac-spr" \
            --log_dir "$CHECKPOINT_DIR" --track \
            --total_timesteps 2000000 &
    done
done

wait
echo "All R7 Stream AC tasks completed."
