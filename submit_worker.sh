#!/bin/bash
#SBATCH --job-name=exconwork
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --error=/dev/null
#SBATCH --output=/dev/null
#SBATCH --time=00:30:00
#SBATCH --time-min=00:05:00
#SBATCH --qos=30min

module load Anaconda3/5.0.1

source activate /scicore/home/lilienfeld/rudorff/opt/conda/rdkit
connection=$(cat /scicore/home/lilienfeld/rudorff/.redis-credentials)

python3 worker.py \
        -f \
        -torsion-list \
        --connect-redis "$1"

