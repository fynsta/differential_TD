#!/bin/bash
set -euo pipefail

HOPPER=$(sbatch --parsable slurm_hopper.sbatch)
ANT=$(sbatch --parsable slurm_ant.sbatch)
HALFCHEETAH=$(sbatch --parsable slurm_halfcheetah.sbatch)
HUMANOID=$(sbatch --parsable slurm_humanoid.sbatch)

echo "Submitted: hopper=$HOPPER ant=$ANT halfcheetah=$HALFCHEETAH humanoid=$HUMANOID"

sbatch \
  --dependency=afterany:${HOPPER},${ANT},${HALFCHEETAH},${HUMANOID} \
  --job-name=notify-all-done \
  --ntasks=1 --mem-per-cpu=100M --time=00:01:00 \
  --mail-type=END \
  --mail-user=fynsta1904@gmail.com \
  --wrap="echo 'All experiments done: hopper=$HOPPER ant=$ANT halfcheetah=$HALFCHEETAH humanoid=$HUMANOID'"
