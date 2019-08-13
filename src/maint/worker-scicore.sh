#!/bin/bash -l
#SBATCH --job-name=exconwork
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --error=/dev/null
#SBATCH --output=/dev/null
#SBATCH --time=00:30:00
#SBATCH --time-min=00:05:00
#SBATCH --qos=30min

HERE=/scicore/home/lilienfeld/bexusi36/dev/2019-clockwork/clockwork

cd $HERE

PYTHON=src/worker/env/bin/python

# $PYTHON -u src/worker/worker.py --sdf $HOME/db/example_pentane_nosym.sdf --redis-task hello
$PYTHON -u src/worker/worker.py --sdf $HOME/db/edgecase_1.sdf --redis-task edge


