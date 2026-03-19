#!/bin/bash
# Launch ALL rebuttal ablation experiments.
# Usage: bash slurm/launch_all.sh
#
# Job summary (MinAtar disabled):
#   A1: 18 tasks (Atari 6, 3 seeds)
#   A2: 18 tasks (Atari 6, 3 seeds)
#   A3: 54 tasks (3 freqs x Atari 6 x 3 seeds)
#   A4: 48 tasks (4 betas x Atari 4 x 3 seeds)
#   A5: 48 tasks (4 freqs x Atari 4 x 3 seeds)
#   A6: DISABLED (MinAtar 5, 3 seeds)
#   A7: 12 tasks (Atari 4, 3 seeds)
#   A8: 36 tasks (4 lambdas x Atari 3 x 3 seeds)
#   Total: ~234 tasks across 13 SLURM jobs (reduced from ~369)

set -e
cd "$(dirname "$0")"
mkdir -p logs_slurm

echo "================================================"
echo "  Rebuttal Ablation Experiment Launcher"
echo "================================================"
echo ""

bash launch_critical.sh
echo ""
bash launch_high.sh
echo ""
bash launch_medium.sh

echo ""
echo "================================================"
echo "  All 13 SLURM jobs submitted (~234 total tasks, MinAtar disabled)"
echo "  Monitor with: squeue -u \$USER"
echo "================================================"
