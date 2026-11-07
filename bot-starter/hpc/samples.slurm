#!/bin/bash
# Does more thinking help? Strategy C with the default parameters at
# 30, 100 and 300 rollouts per decision, against the rival field.
#
# If 300 clearly beats 30, the bot is limited by computation, and a faster
# rack evaluator (a learned value network) would pay off. If not, the edge
# left is in strategy, not speed. (Live, the bot gets about 30-100 rollouts
# in its 300 ms per decision.)
#
#   cd bot-starter && mkdir -p logs
#   sbatch --array=0-59 hpc/samples.slurm      # 3 settings × 20 tasks
#   # when it's done:
#   npx tsx src/sim/merge.ts results/samples/*/*.json
#
# Compare strategy-c's "place" between the three groups (lower is better).

#SBATCH --job-name=scrabble-samples
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=nocona
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=04:00:00
#SBATCH --output=logs/samples_%A_%a.out

eval "$(conda shell.bash hook)" && conda activate node20

set -euo pipefail

# tsx is a dev dependency; an npm set up for production skips it, and every
# task would fail with "tsx: command not found".
if [ ! -x node_modules/.bin/tsx ]; then
  echo "node_modules/.bin/tsx is missing: run npm ci --include=dev on the login node" >&2
  exit 1
fi

SETTINGS=(30 100 300)
TASKS_PER_SETTING=${TASKS_PER_SETTING:-20}
MATCHES_PER_TASK=${MATCHES_PER_TASK:-10}

task=${SLURM_ARRAY_TASK_ID:?run this with sbatch --array}
samples=${SETTINGS[$((task / TASKS_PER_SETTING))]}
slot=$((task % TASKS_PER_SETTING))
out="results/samples/${samples}"
mkdir -p "$out"

npx tsx src/sim/run.ts --lineup field --samples "$samples" --worker \
  --first $((slot * MATCHES_PER_TASK)) --count "$MATCHES_PER_TASK" \
  > "$out/${slot}.json"
