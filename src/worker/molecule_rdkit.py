
import rmsd

import numpy as np
import time

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalForceFields

import matplotlib.pyplot as plt

ATOM_LIST = [x.strip() for x in [
    'h ', 'he', \
    'li', 'be', 'b ', 'c ', 'n ', 'o ', 'f ', 'ne', \
    'na', 'mg', 'al', 'si', 'p ', 's ', 'cl', 'ar', \
    'k ', 'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', \
    'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',  \
    'rb', 'sr', 'y ', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', \
    'cd', 'in', 'sn', 'sb', 'te', 'i ', 'xe',  \
    'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', \
    'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w ', 're', 'os', 'ir', 'pt', \
    'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn', \
    'fr', 'ra', 'ac', 'th', 'pa', 'u ', 'np', 'pu']]


def get_atom(atom):
    """

    *

    """
    atom = atom.lower()
    return ATOM_LIST.index(atom) + 1


def clockwork(n, debug=False):
    """

    get start, step size and no. of steps from clockwork resolution n

    """

    if n == 1:
        start = 0
        step = 0
        n_steps = 0

    else:
        start = 360.0 / 2 ** (n-1)
        step = 360.0 / (2**(n-2))
        n_steps = 2**(n-2)

    if debug:
        print(n, step, n_steps, start)

    return start, step, n_steps


def get_angles(n, debug=False):
    """

    return angles to check at clockwork resolution n

    """

    start, step, n_steps = clockwork(n)
    if n_steps > 1:
        angles = np.arange(start, start+step*n_steps, step)
    else:
        angles = [start]

    return angles


def compare_positions(pos, pos_list, threshold=0.005):
    """

    views_list
    threshold_list

    iterate over views
    - hydrogen views on carbons (Based on bonds)
    - heavy atoms

    """

    for posc in pos_list:
        comparision = rmsd.kabsch_rmsd(pos, posc)

        if comparision < threshold:
            return False

    return True


def align(q_coord, p_coord):
    """

    align q and p.

    return q coord rotated

    """

    U = rmsd.kabsch(q_coord, p_coord)
    q_coord = np.dot(q_coord, U)

    return q_coord


def scan_torsion(resolution, unique_conformers=[]):
    """

    """

    # Load the molecule
    sdf_filename = "sdf/pentane.sdf"
    suppl = Chem.SDMolSupplier(sdf_filename, removeHs=False, sanitize=True)
    mol = next(suppl)

    # Get molecule information
    # n_atoms = mol.GetNumAtoms()
    atoms = mol.GetAtoms()
    # atoms = [atom.GetAtomicNum() for atom in atoms]
    atoms = [atom.GetSymbol() for atom in atoms]

    # Origin
    conformer = mol.GetConformer()
    origin = conformer.GetPositions()
    origin -= rmsd.centroid(origin)

    # Dihedral angle
    a = 0
    b = 1
    c = 5
    d = 8

    # Origin angle
    origin_angle = Chem.rdMolTransforms.GetDihedralDeg(conformer, a, b, c, d)

    # Setup forcefield
    mp = ChemicalForceFields.MMFFGetMoleculeProperties(mol)
    ff = ChemicalForceFields.MMFFGetMoleculeForceField(mol, mp)

    # Define all angles to scan
    angles = get_angles(resolution)

    debug_file = "test.xyz"
    debug_file = open(debug_file, 'a+')

    if len(unique_conformers) == 0:
        xyz = rmsd.calculate_rmsd.set_coordinates(atoms, origin)
        debug_file.write(xyz)
        debug_file.write("\n")

        unique_conformers = [origin]

    for angle in angles:

        # Reset position
        for i, pos in enumerate(origin):
            conformer.SetAtomPosition(i, pos)

        # Set clockwork angle
        Chem.rdMolTransforms.SetDihedralDeg(conformer, a, b, c, d, origin_angle + angle)

        # Setup constrained ff
        ffc = ChemicalForceFields.MMFFGetMoleculeForceField(mol, mp)
        ffc.MMFFAddTorsionConstraint(a, b, c, d, False,
                                     origin_angle+angle, origin_angle + angle, 1.0e10)
        ffc.Minimize(maxIts=1000, energyTol=1e-2, forceTol=1e-3)

        # angle1 = Chem.rdMolTransforms.GetDihedralDeg(conformer, a, b, c, d)

        ff.Minimize(maxIts=1000, energyTol=1e-2, forceTol=1e-4)

        # angle2 = Chem.rdMolTransforms.GetDihedralDeg(conformer, a, b, c, d)

        pos = conformer.GetPositions()
        pos -= rmsd.centroid(pos)

        print("debug", len(unique_conformers))

        unique = compare_positions(pos, unique_conformers)

        if not unique:
            continue

        pos = align(pos, origin)

        unique_conformers.append(pos)

        xyz = rmsd.calculate_rmsd.set_coordinates(atoms, pos)
        debug_file.write(xyz)
        debug_file.write("\n")

        print(angle, unique)

    debug_file.close()

    return unique_conformers


def example_worker(N):

    sdf_filename = "sdf/pentane.sdf"

    suppl = Chem.SDMolSupplier('sdf/pentane.sdf', removeHs=False, sanitize=True)
    mol = next(suppl)

    n_atoms = mol.GetNumAtoms()
    conf = mol.GetConformer()

    add_angle = 30.0

    mp = ChemicalForceFields.MMFFGetMoleculeProperties(mol)

    a = 0
    b = 1
    c = 5
    d = 8

    # Start angle
    angle = Chem.rdMolTransforms.GetDihedralDeg(conf, a, b, c, d)

    ff = ChemicalForceFields.MMFFGetMoleculeForceField(mol, mp)

    pos = ff.Positions()
    pos = np.array(pos)
    pos = pos.reshape((n_atoms, 3))

    # print(pos)

    # help(ff)
    # quit()

    for n in range(N):

        angle = angle + add_angle

        ffc = ChemicalForceFields.MMFFGetMoleculeForceField(mol, mp)
        # ffc.MMFFAddTorsionConstraint(a, b, c, d, False, angle+add_angle, angle+add_angle, 1.0e10)
        Chem.rdMolTransforms.SetDihedralDeg(conf, a, b, c, d, angle + add_angle)
        ffc.Minimize(maxIts=200, energyTol=1e-2, forceTol=1e-2)

        # ff = ChemicalForceFields.MMFFGetMoleculeForceField(mol, mp)
        # ff.Minimize(maxIts=200, energyTol=1e-2, forceTol=1e-2)


    return

if __name__ == '__main__':

    # (clockwork(1))
    # (clockwork(2))
    # (clockwork(3))
    # (clockwork(4))
    # clockwork(5, debug=True)
    # (clockwork(6))

    with open("test.xyz", 'w') as f:
        pass

    unique_conformers = scan_torsion(2)
    unique_conformers = scan_torsion(3, unique_conformers=unique_conformers)
    unique_conformers = scan_torsion(4, unique_conformers=unique_conformers)
    unique_conformers = scan_torsion(5, unique_conformers=unique_conformers)
    unique_conformers = scan_torsion(6, unique_conformers=unique_conformers)
    unique_conformers = scan_torsion(7, unique_conformers=unique_conformers)
    unique_conformers = scan_torsion(8, unique_conformers=unique_conformers)
    unique_conformers = scan_torsion(9, unique_conformers=unique_conformers)
    unique_conformers = scan_torsion(10, unique_conformers=unique_conformers)
    unique_conformers = scan_torsion(11, unique_conformers=unique_conformers)


    quit()
    (clockwork(1))
    (clockwork(2))
    (clockwork(3))
    (clockwork(4))
    (clockwork(5))
    (clockwork(6))

    quit()
    n_list = range(10, 1000, 30)

    xaxis = []
    yaxis = []

    for N in n_list:

        start = time.time()

        example_worker(N)

        end = time.time()

        print(N, end-start)

        ratio = float(N) / (end-start)

        xaxis.append(N)
        yaxis.append(ratio)


    plt.plot(xaxis, yaxis, "k.-")
    plt.savefig("scale_worker_python")

