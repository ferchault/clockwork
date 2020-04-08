#!/usr/bin/env python
from redis import Redis
import uuid
import sys
import os
import subprocess 
import shutil 
import json

redis = Redis.from_url("redis://" + os.environ.get("EXECUTOR_CONSTR", "127.0.0.1:6379/0"))

def xtb_bonds(xyzgeometry):
	_tmpdir = "/run/shm/" + str(uuid.uuid4())
	os.makedirs(_tmpdir)
	os.chdir(_tmpdir)

	with open("run.xyz", "w") as fh:
		fh.write(xyzgeometry)

	# call xtb
	with open("run.log", "w") as fh:
		subprocess.run(["/mnt/c/Users/guido/opt/xtb/6.2.2/bin/xtb", "run.xyz", "--wbo"], stdout=fh, stderr=fh)

	# read bonds
	with open("wbo") as fh:
		lines = fh.readlines()
	bonds = []
	for line in lines:
		parts = line.strip().split()
		parts = parts[:2]
		parts = [int(_)-1 for _ in parts]
		parts = (min(parts), max(parts))
		bonds.append(parts)

	os.chdir("..")
	shutil.rmtree(_tmpdir)
	return bonds

def get_relevant_dihedrals(heavy, bonds, natoms):
	dihs = []
	for i in range(natoms):
		for j in range(natoms):
			if i == j:
				continue
			if not ((i, j) in bonds or (j, i) in bonds):
				continue
			for k in range(natoms):
				if k in (i, j):
					continue
				if not ((k, j) in bonds or (j, k) in bonds):
					continue
				for l in range(natoms):
					if l in (i, j, k):
						continue
					if not ((l, k) in bonds or (k, l) in bonds):
						continue
					if (i not in heavy) and (l not in heavy):
						continue
					if ((i, j, k, l) not in dihs):
						dihs.append((l, k, j, i))
	return dihs

def find_heavy(xyzgeometry):
	lines = xyzgeometry.split('\n')
	natoms = int(lines[0].strip())
	lines = [lines[0], ""] + [" ".join(_.strip().split()[:4]) for _ in lines[2:natoms+2]]

	elements = [_.strip().split()[0] for _ in lines[2:]]
	heavy = [i for i in range(natoms) if elements[i] != "H"]
	return natoms, heavy, "\n".join(lines)

molid = sys.argv[1]
xyzgeometry = open(sys.argv[2]).read()
sdfgeometry = open(sys.argv[3]).read()
natoms, heavy, short_xyz = find_heavy(xyzgeometry)
bonds = xtb_bonds(short_xyz)
dihedrals = get_relevant_dihedrals(heavy, bonds, natoms)

redis.set(f'clockwork:{molid}:xyz', short_xyz)
redis.set(f'clockwork:{molid}:sdf', sdfgeometry)
redis.set(f'clockwork:{molid}:dihedrals', json.dumps(dihedrals))
redis.set(f'clockwork:{molid}:bonds', json.dumps(bonds))