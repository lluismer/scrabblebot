#!/bin/bash
# CMA-ES tuning of strategy C on one whole node.
#
# One job, one node: the tuner (src/sim/cmaes.ts) runs each generation's
# candidates in parallel child processes on the node's cores. It saves its
# state after every generation, so if the job hits its time limit, submit
# it again and it carries on.
#
#   cd bot-starter && mkdir -p logs
#   sbatch hpc/cmaes.slurm                                   # tables + final
#   sbatch --export=ALL,OBJECTIVE=field hpc/cmaes.slurm       # tables only
#   tail -f logs/cmaes_<jobid>.out                            # progress
#
# The line per generation to watch is "mean": the score of the current
# best guess, lower is better. With the default objective (mix) it is the
# average place at a table of six plus 2 × the loss rate in the two-bot
# final against the Bron stand-in. The best parameters so far are under
# "best" in results/cmaes-<objective>.json.
#
# Cost per generation (7 parameters → population 9, plus the mean):
# 10 × 200 table matches plus 10 × 48 final matches, roughly 50-60
# CPU-hours, so about half an hour on 128 cores; 60 generations fit in
# about 30 h.

#SBATCH --job-name=scrabble-cmaes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=nocona
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/cmaes_%j.out

eval "$(conda shell.bash hook)" && conda activate node20

set -euo pipefail

OBJECTIVE=${OBJECTIVE:-mix}
MATCHES=${MATCHES:-200}
FINAL_MATCHES=${FINAL_MATCHES:-48}
GENERATIONS=${GENERATIONS:-60}
SAMPLES=${SAMPLES:-30}

npx tsx src/sim/cmaes.ts \
  --workers "${SLURM_CPUS_PER_TASK:-128}" \
  --objective "$OBJECTIVE" \
  --matches "$MATCHES" \
  --final-matches "$FINAL_MATCHES" \
  --generations "$GENERATIONS" \
  --samples "$SAMPLES" \
  --state "results/cmaes-${OBJECTIVE}.json"
