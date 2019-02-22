#!/bin/bash
#SBATCH --job-name=exconwork
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --error=/dev/null
#SBATCH --output=/dev/null
#SBATCH --time=00:30:00
#SBATCH --time-min=00:05:00
#SBATCH --partition=WAT
#SBATCH --no-requeue

connection=$1

anaconda worker.py \
        -f \
        -torsion-list \
        --connect-redis "$1"

