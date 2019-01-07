#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <openbabel/mol.h>
#include <openbabel/forcefield.h>
#include <openbabel/obconversion.h>

class Workpackage {
// read binary string	
}

int main(int argc,char **argv)
{
	// Needed such that openbabel does not try to parallelise
	omp_set_num_threads(1);

	OpenBabel::OBForceField * ff = OpenBabel::OBForceField::FindForceField("MMFF94");

	Workpackage * wp;

	OpenBabel::OBMol mol;
	for (atom = wp->atoms.begin(); atom != wp->atoms.end(); ++atom) {
		OpenBabel::OBAtom obatom;
		obatom.SetAtomicNum(OpenBabel::OBElements::GetAtomicNum(atom->number);
		obatom.SetVector(atom->x, atom->y, atom->z);
		mol->AddAtom(obatom);
	}
	
	for (bond = wp->bonds.begin(); bond != wp->bonds.end(); ++bond) {
		mol->AddBond(bond->begin, bond->end, bond->order);
	}
	
	// constraints
	constraints = OpenBabel::OBForceField::OBFFConstraints()
	ff->SetConstraints(constraints);
	for (frozen = wp->frozen_dihedrals.begin(); frozen != wp->frozen_dihedrals.end(); ++frozen) {
		constraints->AddTorsionConstraint(frozen->i, frozen->j, frozen->k, frozen->l, frozen->value)
	}

	for (scan in wp->scans.begin(); scan != wp->scans.end(); ++scan) {
		constraints->AddTorsionConstraint(scan->i, scan->j, scan->k, scan->l, scan->value);
		ff->ConjugateGradients(numsteps, threshold);
		constraints->DeleteConstraint(constraints->Size());
		ff->ConjugateGradients(numsteps, threshold);
		
		// Use mol.GetCoordinates() for rmsd check
	}
  
  return 0;
}
