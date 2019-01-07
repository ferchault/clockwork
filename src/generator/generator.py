#!/usr/bin/env python
""" Usage: generator.py FILE_WITH_PATHS_TO_XYZ_ONE_PER_LINE """

import sys
import struct
import openbabel as ob

def binarify_molecules(filename):
	obconv = ob.OBConversion()
	obconv.SetInFormat('xyz')
	for molidx, xyzfile in enumerate(open(filename).readlines()):
		mol = ob.OBMol()
		obconv.ReadFile(mol, xyzfile.strip())
		
		molecule_id = molidx
		number_of_atoms = mol.NumAtoms()

		# atom properties
		nuclear_coordinates, element_numbers = [], []
		first = None
		for atom in ob.OBMolAtomIter(mol):
			element_numbers.append(atom.GetAtomicNum())
			# Shift molecule to implicitly place the first atom at the origin
			if first is None:
				first = [atom.GetX(), atom.GetY(), atom.GetZ()]
			else:
				nuclear_coordinates.append([atom.GetX() - first[0], atom.GetY() - first[1], atom.GetZ() - first[2]])

		# bond properties
		bond_orders = []
		bonds = []
		for bond in ob.OBMolBondIter(mol):
			bond_orders.append(bond.GetBO())
			bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
		number_of_bonds = len(bonds)

		# dihedral properties
		dihedrals = [list(_) for _ in ob.OBMolTorsionIter(mol)]
		number_of_dihedrals = len(dihedrals)

		# binary packing
		formatstring = 'IB%dB%ddB%dB%dBB%dB' % (number_of_atoms, # element numbers
			3*(number_of_atoms - 1), 	# atomic coordinates
			2*number_of_bonds, 			# atom indices in bonds
			number_of_bonds,			# bond orders
			4*number_of_dihedrals)		# atom indices in dihedrals
		nuclear_coordinates = sum(nuclear_coordinates, [])
		bonds = sum(bonds, [])
		dihedrals = sum(dihedrals, [])
		struct.pack(formatstring, 
			molecule_id, 
			number_of_atoms, 
			*element_numbers, 
			*nuclear_coordinates, 
			number_of_bonds, 
			*bonds, 
			*bond_orders,
			number_of_dihedrals,
			*dihedrals)

if __name__ == '__main__':
	binarify_molecules(sys.argv[1])
