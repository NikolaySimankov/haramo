#!/bin/bash
#SBATCH --job-name=haramo_PO_exsp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G                    # bump higher — cheap insurance
#SBATCH --time=00:10:00              
#SBATCH --output=logs/%x_%j.out      # %x=jobname, %j=jobid
#SBATCH --error=logs/%x_%j.err

echo '########################################'
echo 'Date:' $(date --iso-8601=seconds)
echo 'User:' $USER
echo 'Host:' $HOSTNAME
echo 'Job Name:' $SLURM_JOB_NAME
echo 'Job Id:' $SLURM_JOB_ID
echo 'Directory:' $(pwd)
scontrol show job $SLURM_JOB_ID
echo '########################################'

# Activate your env (sbatch starts with a clean shell)
source ~/tabml/bin/activate

# Fail fast on errors
set -euo pipefail
mkdir -p logs

python haramo_cluster.py --d PO_exsp --b slurm --t 50