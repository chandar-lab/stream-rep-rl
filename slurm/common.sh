#!/bin/bash
# Common setup for all rebuttal ablation experiments.
# Source this file from individual job scripts.

module load httpproxy
module load python/3.10

WANDB_LOG_DIR="~/scratch/wandb_logs"

# check if wandb logging dir exists, if not create it
if [ ! -d "$WANDB_LOG_DIR" ]; then
    mkdir -p "$WANDB_LOG_DIR"
    echo "Created WANDB_DIR at $WANDB_LOG_DIR"
fi

CHECKPOINT_DIR="~/scratch/stream_rl_checkpoints"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    mkdir -p "$CHECKPOINT_DIR"
    echo "Created CHECKPOINT_DIR at $CHECKPOINT_DIR"
fi

export WANDB_DIR="$WANDB_LOG_DIR"
export WANDB_ENTITY=""
export WANDB_PROJECT="stream-rl-rebuttal"
export WANDB_MODE=offline

if [ $? -ne 0 ]; then
    echo "Module loading failed. Exiting."
    exit 1
fi

source ~/.venv/bin/activate

if [ $? -ne 0 ]; then
    echo "Virtual environment activation failed. Exiting."
    exit 1
fi

cd ~/stream-rep-rl

# --- Shared constants ---
ATARI_6=(
    "Pong-v4"
    "Breakout-v4"
    "Seaquest-v4"
    "MsPacman-v4"
    "Alien-v4"
    "SpaceInvaders-v4"
)

ATARI_4=(
    "Pong-v4"
    "Breakout-v4"
    "Seaquest-v4"
    "SpaceInvaders-v4"
)

ATARI_3=(
    "Pong-v4"
    "Breakout-v4"
    "Seaquest-v4"
)

MINATAR_5=(
    "MinAtar/Breakout-v1"
    "MinAtar/Asterix-v1"
    "MinAtar/Freeway-v1"
    "MinAtar/Seaquest-v1"
    "MinAtar/SpaceInvaders-v1"
)

SEEDS=(1 42 999)

# Helper: launch one srun task per (env, seed) combination
# Usage: launch_runs ALGORITHM ENVS_ARRAY_NAME EXTRA_ARGS...
launch_runs() {
    local algo="$1"
    local -n envs_ref="$2"
    shift 2
    local extra_args="$@"

    for SEED in "${SEEDS[@]}"; do
        for ENV in "${envs_ref[@]}"; do
            srun --ntasks=1 --cpus-per-task=4 --exclusive \
                python "algorithms/${algo}" \
                --env_type atari \
                --env_id "$ENV" \
                --seed "$SEED" \
                --log_dir "$CHECKPOINT_DIR" \
                --track \
                --resume \
                --total_timesteps 10000000 \
                --explore_frac 0.10 \
                $extra_args &
        done
    done
}

# Same but for MinAtar (5M steps default)
launch_runs_minatar() {
    local algo="$1"
    local -n envs_ref="$2"
    shift 2
    local extra_args="$@"

    for SEED in "${SEEDS[@]}"; do
        for ENV in "${envs_ref[@]}"; do
            srun --ntasks=1 --cpus-per-task=4 --exclusive \
                python "algorithms/${algo}" \
                --env_type minatar \
                --env_id "$ENV" \
                --seed "$SEED" \
                --track \
                --resume \
                --log_dir "$CHECKPOINT_DIR" \
                --total_timesteps 5000000 \
                $extra_args &
        done
    done
}
