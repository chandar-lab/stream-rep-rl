#!/bin/bash
#SBATCH --job-name=R7_ddpg
#SBATCH --time=12:00:00
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/R7_ddpg_%j.out
#SBATCH --error=logs/R7_ddpg_%j.err

# Rebuttal Experiment 7: Continuous control with streaming DDPG +/- SPR.
# 3 envs x 3 seeds x 2 methods = 18 tasks, 1M steps each.

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/common.sh"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
fi

MUJOCO_ENVS=("HalfCheetah-v4" "Hopper-v4" "Walker2d-v4")

for SEED in "${SEEDS[@]}"; do
    for ENV in "${MUJOCO_ENVS[@]}"; do
        srun --ntasks=1 --cpus-per-task=4 --exclusive \
            python algorithms/ddpg_streaming.py \
            --env_id "$ENV" --seed "$SEED" \
            --exp_name "ddpg-streaming" --exp_class "ddpg-streaming" \
            --log_dir "$CHECKPOINT_DIR" --track \
            --total_timesteps 1000000 &
        srun --ntasks=1 --cpus-per-task=4 --exclusive \
            python algorithms/ddpg_streaming_spr.py \
            --env_id "$ENV" --seed "$SEED" \
            --exp_name "ddpg-streaming-spr" --exp_class "ddpg-streaming-spr" \
            --log_dir "$CHECKPOINT_DIR" --track \
            --total_timesteps 1000000 &
    done
done

wait
echo "All R7 DDPG tasks completed."
