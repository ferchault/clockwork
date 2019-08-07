
PYTHON=src/worker/env/bin/python

workers:
	${PYTHON} src/worker/worker.py --sdf examples/pentane_nosymmetry.sdf --jobfile _tmp_joblist_unconv.txt -j 1
	@# ${PYTHON} src/worker/worker.py --sdf _tmp_examples/pentane_nosymmetry.sdf --jobfile _tmp_joblist.txt -j 30

worker:
	@# pyprofile ${PYTHON} src/worker/worker.py --sdf examples/pentane_nosymmetry.sdf --jobfile _tmp_joblist_empty.txt
	@# pyprofile ${PYTHON} src/worker/worker.py --sdf examples/pentane_nosymmetry.sdf --jobfile _tmp_joblist.txt
	pyprofile ${PYTHON} src/worker/worker.py --sdf examples/pentane_nosymmetry.sdf --jobfile _tmp_joblist_unconv.txt

merge:
	pyprofile ${PYTHON} src/worker/merge.py

test:
	${PYTHON} src/worker/merge.py --sdf _tmp_apentane_cost/1_1.sdf _tmp_apentane_cost/1_2.sdf

unconv:
	${PYTHON} src/worker/worker.py --sdf _tmp_test/unconv.sdf

redis:
	${PYTHON} src/worker/worker.py --sdf examples/pentane_nosymmetry.sdf --redis-task hello

energies:
	# ${PYTHON} src/worker/plot.py --sdf *dump*sdf
	${PYTHON} src/worker/plot.py --sdf _tmp_apentane_cost/2_6.sdf
	# ${PYTHON} src/worker/plot.py --sdf _tmp_apentane/1.*_*.6.sdf
	# ${PYTHON} src/worker/plot.py --sdf _tmp_apentane/1.10_15.6.sdf
	# ${PYTHON} src/worker/plot.py --sdf _tmp_data/*0*.sdf


edge_prepare_jobs:
	@${PYTHON} src/worker/worker.py --sdf examples/edgecase_1.sdf

edge_local_jobs:
	${PYTHON} src/worker/worker.py --sdf examples/edgecase_1.sdf --jobfile _edge_joblist.txt



