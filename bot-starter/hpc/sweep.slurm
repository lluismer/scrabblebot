#!/bin/bash
# Parameter sweep for strategy C on a SLURM cluster.
#
# Each array task plays MATCHES_PER_TASK seeded matches of one variant
# against default copies of the bot and writes a JSON summary. Tasks are
# spread over the variants listed in VARIANTS, so the array size is
# (number of variants) × TASKS_PER_VARIANT.
#
# LINEUP picks the table (see src/sim/run.ts). The tournament has no sim
# bots, so the two that matter are:
#   table  a variant + 5 default copies: a Swiss table (the default)
#   final  a variant vs 1 default: the two-bot final
#
#   cd bot-starter && npm ci
#   mkdir -p logs        # SLURM opens logs/ before this script runs
#   sbatch --array=0 --export=ALL,MATCHES_PER_TASK=1,SAMPLES=4 hpc/sweep.slurm   # smoke test
#   sbatch --array=0-159 hpc/sweep.slurm                       # Swiss tables
#   sbatch --array=0-159 --export=ALL,LINEUP=final hpc/sweep.slurm
#   # when they're done:
#   npx tsx src/sim/merge.ts results/table/*/*.json
#   npx tsx src/sim/merge.ts results/final/*/*.json
#
# At a table, "variant" place below 3.5 (the average of 6 seats) means the
# change helps; in the final, a win rate above 50%.
#
# Adjust the #SBATCH lines to your cluster (partition, account, module load).

#SBATCH --job-name=scrabble-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=nocona
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%A_%a.out

# TTU HPCC: Node 20+ comes from a conda env made once with
#   conda create -n node20 -c conda-forge nodejs=20 -y
# Activate it before set -u (conda's activation scripts trip on unset variables).
eval "$(conda shell.bash hook)" && conda activate node20

set -euo pipefail

# Centred on the local duels (128 matches each, vs defaults earnFactor 0.3,
# lambdaFloor 0, playMargin 0): playMargin=3 won 63% vs 39%,
# earnFactor=0.6 56% vs 45%, lambdaFloor=0.2 54% vs 45%, earnFactor=0 lost.
VARIANTS=(
  "playMargin=3"
  "playMargin=6"
  "playMargin=10"
  "earnFactor=0.6"
  "earnFactor=0.9"
  "lambdaFloor=0.2"
  "lambdaFloor=0.4"
  "sideShare=0.5"
)
TASKS_PER_VARIANT=${TASKS_PER_VARIANT:-20}
MATCHES_PER_TASK=${MATCHES_PER_TASK:-25}
SAMPLES=${SAMPLES:-30}

task=${SLURM_ARRAY_TASK_ID:?run this with sbatch --array}
variant=${VARIANTS[$((task / TASKS_PER_VARIANT))]}
slot=$((task % TASKS_PER_VARIANT))
out="results/${LINEUP:-table}/${variant}"
mkdir -p "$out" logs

npx tsx src/sim/run.ts --lineup "${LINEUP:-table}" --vparam "$variant" \
  --samples "$SAMPLES" --worker \
  --first $((slot * MATCHES_PER_TASK)) --count "$MATCHES_PER_TASK" \
  > "$out/${slot}.json"
