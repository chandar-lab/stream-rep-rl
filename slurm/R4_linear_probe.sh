#!/bin/bash
#SBATCH --job-name=R4_linear_probe
#SBATCH --time=55:00:00
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/R4_linear_probe_%j.out
#SBATCH --error=logs/R4_linear_probe_%j.err

# Rebuttal Experiment 4: Linear probing.
# Phase A: Train QRC and QRC+SPR+orth with periodic checkpointing.
# 3 envs x 1 seed x 2 methods = 6 tasks.
# Phase B: Run scripts/linear_probe.py offline after training.

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/common.sh"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
fi

PROBE_ENVS=("Pong-v4" "Breakout-v4" "Seaquest-v4")
SEED=1

# QRC baseline (no SPR)
for ENV in "${PROBE_ENVS[@]}"; do
    srun --ntasks=1 --cpus-per-task=4 --exclusive \
        python "algorithms/qrc.py" \
        --env_type atari \
        --env_id "$ENV" \
        --seed "$SEED" \
        --exp_class qrc-probe \
        --exp_name "qrc-probe" \
        --periodic_checkpointing \
        --save_model \
        --num_checkpoints 20 \
        --log_dir "$CHECKPOINT_DIR" \
        --track \
        --resume \
        --total_timesteps 10000000 \
        --explore_frac 0.10 &
done

# QRC+SPR+orth
for ENV in "${PROBE_ENVS[@]}"; do
    srun --ntasks=1 --cpus-per-task=4 --exclusive \
        python "algorithms/qrc-spr-orth.py" \
        --env_type atari \
        --env_id "$ENV" \
        --seed "$SEED" \
        --exp_class qrc-spr-orth-probe \
        --exp_name "qrc-spr-orth-probe" \
        --periodic_checkpointing \
        --save_model \
        --num_checkpoints 20 \
        --log_dir "$CHECKPOINT_DIR" \
        --track \
        --resume \
        --total_timesteps 10000000 \
        --explore_frac 0.10 &
done

wait
echo "All R4 linear probe training tasks completed."
echo "Now run Phase B (offline probing) for each env:"
echo "  python scripts/linear_probe.py \\"
echo "    --checkpoint_dirs \$CHECKPOINT_DIR/Pong-v4__qrc-probe__1 \$CHECKPOINT_DIR/Pong-v4__qrc-spr-orth-probe__1 \\"
echo "    --labels 'QRC' 'QRC+SPR+orth' \\"
echo "    --network_types base spr \\"
echo "    --env_id Pong-v4 --env_type atari"
