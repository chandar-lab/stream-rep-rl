#!/bin/bash
#SBATCH --job-name=R5_lr_ratio
#SBATCH --time=55:00:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/R5_lr_ratio_%j.out
#SBATCH --error=logs/R5_lr_ratio_%j.err

# Rebuttal Experiment 5: LR ratio analysis for Stream Q(lambda)+SPR+orth2.
# Runs on 3 Atari envs (Pong, Breakout, Seaquest) x 1 seed with diagnostic logging.
# After training, analyze with: python scripts/lr_ratio_analysis.py --wandb_project stream-rl-ablations

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/common.sh"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
fi

LR_ENVS=("Pong-v4" "Breakout-v4" "Seaquest-v4")
SEED=1

for ENV in "${LR_ENVS[@]}"; do
    srun --ntasks=1 --cpus-per-task=4 --exclusive \
        python "algorithms/streamq-spr-ortho.py" \
        --env_type atari \
        --env_id "$ENV" \
        --seed "$SEED" \
        --exp_class strq-spr-orth2-lr-diag \
        --exp_name strq-spr-orth2-lr-diag \
        --log_dir "$CHECKPOINT_DIR" \
        --track \
        --resume \
        --total_timesteps 10000000 \
        --explore_frac 0.10 &
done

wait
echo "All R5 LR ratio tasks completed."
