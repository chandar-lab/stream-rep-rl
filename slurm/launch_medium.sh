#!/bin/bash
# Launch MEDIUM priority ablations (A5, A6, A7, A8) - Week 2-3
# Usage: bash slurm/launch_medium.sh

set -e
cd "$(dirname "$0")"
mkdir -p logs_slurm

echo "=== Submitting MEDIUM priority ablations ==="

# A5: periodic SPR with different frequencies
for FREQ in 1 5 10 20; do
    sbatch A5_periodic_spr.sh "$FREQ"
    echo "  Submitted A5: Periodic SPR (freq=$FREQ)"
done

# sbatch A6_ema_obgd.sh
# echo "  Submitted A6: EMA ObGD orth"

sbatch A7_plasticity.sh
echo "  Submitted A7: Plasticity baseline"

sbatch A8_lambda_orth.sh
echo "  Submitted A8: Lambda-orth interaction"

echo "=== MEDIUM priority ablations submitted (7 jobs) ==="
