#!/bin/bash -l
#SBATCH --job-name=exconwork
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --error=_log_slurm_%j.stderr
#SBATCH --output=_log_slurm_%j.stdout
#SBATCH --time=00:30:00
#SBATCH --time-min=00:05:00
#SBATCH --qos=30min

# SLURM sets TMPDIR

HERE=/scicore/home/lilienfeld/bexusi36/dev/2019-clockwork/clockwork

cd $HERE

PYTHON=src/worker/env/bin/python

# $PYTHON -u src/worker/worker.py --sdf $HOME/db/example_pentane_nosym.sdf --redis-task hello

SDF=$HOME/db/qm9.c7o2h10.sdf.gz
SDFTOR=$HOME/db/qm9.c7o2h10.torsions
TASK=case1
$PYTHON -u src/worker/worker.py --sdf $SDF --sdftor $SDFTOR --redis-task $TASK

# SDF=$HOME/db/edgecase_1.sdf
# TASK=case1
# $PYTHON -u src/worker/worker.py --sdf $HOME/db/edgecase_1.sdf --redis-task edge

