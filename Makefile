
PYTHON=src/worker/env/bin/python

workers:
	# ${PYTHON} src/worker/worker.py --sdf _tmp_examples/pentane_nosymmetry.sdf --jobfile _tmp_joblist_test.txt -j 2
	# ${PYTHON} src/worker/worker.py --sdf _tmp_examples/pentane_nosymmetry.sdf --jobfile _tmp_joblist_test.txt -j 30
	${PYTHON} src/worker/worker.py --sdf _tmp_examples/pentane_nosymmetry.sdf --jobfile _tmp_joblist.txt -j 30

worker:
	${PYTHON} src/worker/worker.py --sdf _tmp_examples/pentane_nosymmetry.sdf --jobfile _tmp_joblist_test.txt
	# pyprofile ${PYTHON} src/worker/worker.py --sdf examples/pentane_nosymmetry.sdf --jobfile _tmp_joblist.txt
	# pyprofile ${PYTHON} src/worker/worker.py --sdf examples/pentane_nosymmetry.sdf --jobfile _tmp_joblist_unconv.txt

merge:
	pyprofile ${PYTHON} src/worker/merge.py

test:
	${PYTHON} src/worker/merge.py --sdf _tmp_apentane_cost/1_1.sdf _tmp_apentane_cost/1_2.sdf

unconv:
	${PYTHON} src/worker/worker.py --sdf _tmp_test/unconv.sdf

energies:
	# ${PYTHON} src/worker/plot.py --sdf _tmp_apentane_cost/2_6.sdf
	# ${PYTHON} src/worker/plot.py --sdf _tmp_apentane/1.4_5.6.sdf
	${PYTHON} src/worker/plot.py --sdf _tmp_apentane/1.4_6.6.sdf

