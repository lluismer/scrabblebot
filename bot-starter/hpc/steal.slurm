#!/bin/bash
# "What to steal": strategy C with each candidate change, at a table with
# 5 imitation stand-ins of the current top real bots (hpc/imitators.json,
# fitted by analysis/fit.py + report.py on all-real tables).
#
#   cd bot-starter && npm ci --include=dev && mkdir -p logs
#   sbatch --array=0 --export=ALL,MATCHES_PER_TASK=1,SAMPLES=4 hpc/steal.slurm   # smoke test
#   sbatch --array=0-79 hpc/steal.slurm          # 4 variants x 20 tasks x 25 matches
#   # when done, one summary per variant:
#   for d in results/steal/*/; do echo "== $d"; npx tsx src/sim/merge.ts "$d"*.json; done
#
# Compare strategy-c's place (lower is better; 3.5 = average of 6) and tiles.

#SBATCH --job-name=scrabble-steal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=nocona
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=02:00:00
#SBATCH --output=logs/steal_%A_%a.out

eval "$(conda shell.bash hook)" && conda activate node20

set -euo pipefail

if [ ! -x node_modules/.bin/tsx ]; then
  echo "node_modules/.bin/tsx is missing: run npm ci --include=dev on the login node" >&2
  exit 1
fi

# Space-separated key=value overrides per variant ("" = current defaults).
#  tiles:  top bots win 13-15 tiles at ~13 coins -> bid a floor until on pace for 14
#  hold7:  top bots hold ~7.5 tiles and 45-59% of their words are 7+ letters
VARIANTS=(
  "baseline|"
  "tiles|tileTarget=14 floorBid=13"
  "hold7|minPlayLen=7"
  "tiles+hold7|tileTarget=14 floorBid=13 minPlayLen=7"
)
TASKS_PER_VARIANT=${TASKS_PER_VARIANT:-20}
MATCHES_PER_TASK=${MATCHES_PER_TASK:-25}
SAMPLES=${SAMPLES:-30}

task=${SLURM_ARRAY_TASK_ID:?run this with sbatch --array}
entry=${VARIANTS[$((task / TASKS_PER_VARIANT))]}
name=${entry%%|*}
overrides=${entry#*|}
slot=$((task % TASKS_PER_VARIANT))
out="results/steal/${name}"
mkdir -p "$out" logs

params=()
for kv in $overrides; do params+=(--param "$kv"); done

# Same seeds for every variant (--first), so the comparison is paired.
npx tsx src/sim/run.ts --lineup field --rivals-file hpc/imitators.json \
  ${params[@]+"${params[@]}"} --samples "$SAMPLES" --worker \
  --first $((slot * MATCHES_PER_TASK)) --count "$MATCHES_PER_TASK" \
  > "$out/${slot}.json"
