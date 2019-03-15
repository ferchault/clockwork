#!/bin/bash
PROJECT="$1"
# connect to REDIS
. /home/vonrudorff/opt/conda/install/etc/profile.d/conda.sh
conda activate /home/vonrudorff/opt/conda/rdkit
connection=$(cat /home/vonrudorff/.redis-credentials)
CHEMSPACELAB_REDIS_CONNECTION="$connection" python /home/vonrudorff/workcopies/clockwork/src/worker/rediscomm.py --haswork $PROJECT || exit

# work left, let's submit
CURRENT=$(/home/admin/slurm/slurm16/bin/squeue -u $(whoami) | grep exconwo | wc -l)
TARGET=200
DIFF=$(($TARGET-CURRENT))
#echo $DIFF
[ $DIFF -lt 0 ] && DIFF=0

for i in $(seq 1 $DIFF)
do
	/home/admin/slurm/slurm16/bin/sbatch <(cat /home/vonrudorff/workcopies/clockwork/src/maint/worker-alchemy.sh | sed "s/PROJECTPROJECTPROJECT/$PROJECT/") &> /dev/null
done
