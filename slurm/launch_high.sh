#!/bin/bash
# Launch HIGH priority ablations (A3, A4) - Week 2 priority
# Usage: bash slurm/launch_high.sh

set -e
cd "$(dirname "$0")"
mkdir -p ../logs_slurm

echo "=== Submitting HIGH priority ablations ==="

# A3: target net with 3 different frequencies
for FREQ in 100 1000 8000; do
    sbatch A3_target_net.sh "$FREQ"
    echo "  Submitted A3: Target net (freq=$FREQ)"
done

sbatch A4_beta_sweep.sh
echo "  Submitted A4: Beta sensitivity sweep"

echo "=== HIGH priority ablations submitted (4 jobs) ==="
