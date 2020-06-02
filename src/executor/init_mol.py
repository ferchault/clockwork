#!/usr/bin/env python
from redis import Redis
from rdkit import Chem
import uuid
import sys
import os
import subprocess 
import shutil 
import json
import xyz2mol

redis = Redis.from_url("redis://" + os.environ.get("EXECUTOR_CONSTR", "127.0.0.1:6379/0"))

def get_bonds_smiles(xyzgeometry):
	atoms = []
	coords = []
	lines = xyzgeometry.split("\n")
	natoms = int(lines[0])
	for line in lines[2:natoms+2]:
		parts = line.strip().split()
		atoms.append(xyz2mol.int_atom(parts[0]))
		coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
	
	mol = xyz2mol.xyz2mol(atoms, coords, charge=0)
	bonds = []
	for bond in mol.GetBonds():
		a, b = bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()
		a, b = min(a, b), max(a, b)
		bonds.append((a, b))
	
	# canonicalize
	smiles = Chem.MolToSmiles(mol)
	m = Chem.MolFromSmiles(smiles)
	smiles = Chem.MolToSmiles(m)

	return bonds, smiles

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
					if (j not in heavy) and (k not in heavy):
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
bonds, smiles = get_bonds_smiles(short_xyz)
dihedrals = get_relevant_dihedrals(heavy, bonds, natoms)

redis.set(f'clockwork:{molid}:xyz', short_xyz)
redis.set(f'clockwork:{molid}:sdf', sdfgeometry)
redis.set(f'clockwork:{molid}:dihedrals', json.dumps(dihedrals))
redis.set(f'clockwork:{molid}:bonds', json.dumps(bonds))
redis.set(f'clockwork:{molid}:smiles', smiles)
