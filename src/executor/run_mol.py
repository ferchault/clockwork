#!/usr/bin/env python
from redis import Redis
import uuid
import sys
import os
import subprocess 
import shutil 
import numpy as np
import itertools as it
import json

from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalForceFields

redis = Redis.from_url("redis://" + os.environ.get("EXECUTOR_CONSTR", "127.0.0.1:6379/0"))

ENERGY_THRESHOLD = 1e-4
ANGLE_DELTA = 1e-7
FF_RELAX_STEPS = 50

def clockwork(resolution):
	if resolution == 0:
		start = 0
		step = 360
		n_steps = 1
	else:
		start = 360.0 / 2.0 ** (resolution)
		step = 360.0 / 2.0 ** (resolution-1)
		n_steps = 2 ** (resolution - 1)
	return start, step, n_steps

def get_classical_constrained_geometry(sdfstr, torsions, molname, dihedrals, angles):
	mol = Chem.MolFromMolBlock(sdfstr, removeHs=False)

	ffprop = ChemicalForceFields.MMFFGetMoleculeProperties(mol)
	ffc = ChemicalForceFields.MMFFGetMoleculeForceField(mol, ffprop)
	conformer = mol.GetConformer()

	# Set angles and constrains for all torsions
	for dih_id, angle in zip(dihedrals, angles):
		# Set clockwork angle
		try: Chem.rdMolTransforms.SetDihedralDeg(conformer, *torsions[dih_id], float(angle))
		except: pass

		# Set forcefield constrain
		ffc.MMFFAddTorsionConstraint(*torsions[dih_id], False, angle-ANGLE_DELTA, angle+ANGLE_DELTA, 1.0e10)

	# reduce bad contacts
	try:
		ffc.Minimize(maxIts=FF_RELAX_STEPS, energyTol=1e-2, forceTol=1e-3)
	except RuntimeError:
		pass

	atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
	coordinates = conformer.GetPositions()

	return f'{len(atoms)}\n\n' + '\n'.join([f'{element} {coords[0]} {coords[1]} {coords[2]}' for element, coords in zip(atoms, coordinates)])

def do_workpackage(molname, dihedrals, resolution):
	ndih = len(dihedrals)
	start, step, n_steps = clockwork(resolution)
	scanangles = np.arange(start, start+step*n_steps, step)

	# fetch input
	sdfstr = redis.get(f'clockwork:{molname}:sdf').decode("ascii")
	torsions = json.loads(redis.get(f'clockwork:{molname}:dihedrals').decode("ascii"))

	accepted_geometries = []
	accepted_energies = []
	for angles in it.product(scanangles, repeat=ndih):
		xyzfile = get_classical_constrained_geometry(sdfstr, torsions, molname, dihedrals, angles)
		print (xyzfile)
		#optxyzfile, energy, bonds = get_xtb_geoopt(xyzfile)
		#if set(bonds) != set(refbonds):
		#	continue

		#for i in range(len(accepted_energies)):
		#	if abs(accepted_energies[i] - energy) < ENERGY_THRESHOLD:
		#		# compare geometries optxyzfile vs accepted_geometries
		#else:
		#	accepted_energies.append(energy)
		#	accepted_geometries.append(optxyzfile)
	
	results = {}
	results['mol'] = molname
	results['ndih'] = ndih
	results['res'] = resolution
	results['geometries'] = accepted_geometries
	results['energies'] = accepted_energies
	return json.dumps(results)

do_workpackage("debug", (1, 2, 3), 2)

