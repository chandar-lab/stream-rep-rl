#!/bin/bash
# Launch CRITICAL ablations (A1, A2) - Week 1 priority
# Usage: bash slurm/launch_critical.sh

set -e
cd "$(dirname "$0")"
mkdir -p ../logs_slurm


echo "=== Submitting CRITICAL ablations ==="

sbatch A1_orth_rl.sh
echo "  Submitted A1: Orth on RL gradients"

sbatch A2_dqn_spr_orth.sh
echo "  Submitted A2: DQN+SPR+orth"

echo "=== CRITICAL ablations submitted (2 jobs) ==="
