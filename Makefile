
PYTHON=src/worker/env/bin/python

workers:
	${PYTHON} src/worker/worker.py --sdf examples/pentane_nosymmetry.sdf --jobfile _tmp_joblist_unconv.txt -j 1
	@# ${PYTHON} src/worker/worker.py --sdf _tmp_examples/pentane_nosymmetry.sdf --jobfile _tmp_joblist.txt -j 30

worker:
	@# pyprofile ${PYTHON} src/worker/worker.py --sdf examples/pentane_nosymmetry.sdf --jobfile _tmp_joblist_empty.txt
	@# pyprofile ${PYTHON} src/worker/worker.py --sdf examples/pentane_nosymmetry.sdf --jobfile _tmp_joblist.txt
	@# pyprofile ${PYTHON} src/worker/worker.py --sdf examples/pentane_nosymmetry.sdf --jobfile _tmp_joblist_unconv.txt
	@# ${PYTHON} src/worker/worker.py --sdf examples/pentane_nosymmetry.sdf -j 1 --jobfile _tmp_joblist_1.txt
	${PYTHON} src/worker/worker.py --sdf examples/pentane_nosymmetry.sdf -j 1 --jobfile _tmp_joblist_1-1.txt

new_jobs:
	@# pyprofile ${PYTHON} src/worker/worker.py --sdf examples/pentane_nosymmetry.sdf --jobcombos "1,7" "1,8"
	@pyprofile ${PYTHON} src/worker/worker.py --sdf examples/pentane_nosymmetry.sdf --jobcombos "1,9" "1,10"

merge:
	# @pyprofile ${PYTHON} src/worker/merge.py --sdf _tmp_data/0.*.7.sdf --dump > _tmp_conv_1/1_7.sdf
	# @pyprofile ${PYTHON} src/worker/merge.py --sdf _tmp_data/0.*.8.sdf --dump > _tmp_conv_1/1_8.sdf
	@pyprofile ${PYTHON} src/worker/merge.py --sdf _tmp_data/0.*.9.sdf --dump > _tmp_conv_1/1_9.sdf
	@pyprofile ${PYTHON} src/worker/merge.py --sdf _tmp_data/0.*.10.sdf --dump > _tmp_conv_1/1_10.sdf

d=_tmp_apentane_cost
merge_conv:
	@#pyprofile ${PYTHON} src/worker/merge.py --sdf _tmp_conv_1/1_*.sdf --debug
	pyprofile ${PYTHON} src/worker/merge.py --debug --sdf $d/1_1.sdf $d/2_1.sdf $d/1_2.sdf $d/3_1.sdf $d/2_2.sdf $d/1_3.sdf $d/4_1.sdf $d/3_2.sdf $d/2_3.sdf $d/1_4.sdf $d/1_10.sdf

merge_cost:
	pyprofile ${PYTHON} src/worker/merge.py --sdf _tmp_apentane_cost/

test:
	${PYTHON} src/worker/merge.py --sdf _tmp_apentane_cost/1_1.sdf _tmp_apentane_cost/1_2.sdf

unconv:
	${PYTHON} src/worker/worker.py --sdf _tmp_test/unconv.sdf

redis:
	${PYTHON} src/worker/worker.py --sdf examples/pentane_nosymmetry.sdf --redis-task hello

scicore:
	${PYTHON} src/worker/worker.py --sdf ~/db/example_pentane_nosym.sdf  --redis-task hello

energies:
	@# ${PYTHON} src/worker/plot.py --sdf *dump*sdf
	@# ${PYTHON} src/worker/plot.py --sdf _tmp_apentane_cost/2_5.sdf
	@# ${PYTHON} src/worker/plot.py --sdf _tmp_apentane/1.*_*.5.sdf
	@# ${PYTHON} src/worker/plot.py --sdf _tmp_apentane/1.10_15.6.sdf
	@# ${PYTHON} src/worker/plot.py --sdf _tmp_data/*0*.sdf
	@# ${PYTHON} src/worker/plot.py --sdf _tmp_apentane_cum/all.sdf
	@# ${PYTHON} src/worker/plot.py --sdf _tmp_conv_1/*.sdf
	@${PYTHON} src/worker/plot.py --sdf ~/db/qm9.c7o2h10.sdf.gz


edge_prepare_jobs:
	@${PYTHON} src/worker/worker.py --sdf examples/edgecase_1.sdf

edge_local_jobs:
	${PYTHON} src/worker/worker.py --sdf examples/edgecase_1.sdf --jobfile _edge_joblist.txt

edge_submit_redis:
	${PYTHON} src/worker/communication/redis_submit.py --jobfile _edge_joblist.txt --redis-task edge


c7o2_make_torsions:
	${PYTHON} src/worker/admin.py --sdf ~/db/qm9.c7o2h10.sdf.gz

c7o2_prepare_jobs:
	@${PYTHON} src/worker/worker.py --sdf ~/db/qm9.c7o2h10.sdf.gz --sdftor ~/db/qm9.c7o2h10.torsions --jobcombos "1,1" "2,1" "1,2" "3,1" "2,2" "1,3" "3,2" "2,3" "1,4""3,3" "2,4" "1,5" > _case1_joblist2.txt

c7o2_submit_redis:
	${PYTHON} src/worker/communication/redis_submit.py --jobfile _case1_joblist2.txt --redis-task case1

