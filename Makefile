
PYTHON=src/worker/env/bin/python

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

energies:
	${PYTHON} src/worker/plot.py --sdf _tmp_apentane_cost/2_4.sdf
	# ${PYTHON} src/worker/plot.py --sdf _tmp_apentane/1.*_*.4.sdf
	# ${PYTHON} src/worker/plot.py --sdf _tmp_apentane/1.10_15.6.sdf
	# ${PYTHON} src/worker/plot.py --sdf _tmp_data/0.sdf

