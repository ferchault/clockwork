
import numpy as np

import itertools

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalForceFields

import rmsd

def load_sdf(filename):
    """

    load sdf file and return rdkit mol

    """

    suppl = Chem.SDMolSupplier(filename, removeHs=False, sanitize=True)
    mol = next(suppl)

    return mol


def clockwork(res, debug=False):
    """

    get start, step size and no. of steps from clockwork resolution n

    @param
        res int resolution
        debug boolean

    """

    if res == 1:
        start = 0
        step = 0
        n_steps = 0

    else:
        start = 360.0 / 2 ** (res-1)
        step = 360.0 / (2**(res-2))
        n_steps = 2**(res-2)

    if debug:
        print(res, step, n_steps, start)

    return start, step, n_steps


def get_angles(res, num_torsions):
    """

    Setup angle iterator based on number of torsions

    """

    start, step, n_steps = clockwork(res)

    if n_steps > 1:
        angles = np.arange(start, start+step*n_steps, step)
    else:
        angles = [start]

    angles = [0] + list(angles)

    iterator = itertools.product(angles, repeat=num_torsions)

    next(iterator)

    return iterator


def align(q_coord, p_coord):
    """

    align q and p.

    return q coord rotated

    """

    U = rmsd.kabsch(q_coord, p_coord)
    q_coord = np.dot(q_coord, U)

    return q_coord


def scan_angles(mol, n_steps, torsions, globalopt=False):
    """

    scan torsion and get energy landscape

    """

    # Load mol info
    n_atoms = mol.GetNumAtoms()
    n_torsions = len(torsions)
    atoms = mol.GetAtoms()
    atoms = [atom.GetSymbol() for atom in atoms]

    # Setup forcefield for molecule
    # no constraints
    ffprop_mmff = ChemicalForceFields.MMFFGetMoleculeProperties(mol)
    forcefield = ChemicalForceFields.MMFFGetMoleculeForceField(mol, ffprop_mmff)

    # Get conformer and origin
    conformer = mol.GetConformer()
    origin = conformer.GetPositions()
    origin -= rmsd.centroid(origin)

    # Origin angle
    origin_angles = []

    for idxs in torsions:
        angle = Chem.rdMolTransforms.GetDihedralDeg(conformer, *idxs)
        origin_angles.append(angle)


    angles = np.linspace(0.0, 360.0, n_steps)

    axis_angles = []
    axis_energies = []

    f = open("test.xyz", 'w')

    # Get resolution angles
    for angles in itertools.product(angles, repeat=n_torsions):

        # Reset positions
        for i, pos in enumerate(origin):
            conformer.SetAtomPosition(i, pos)

        # Setup constrained forcefield
        ffc = ChemicalForceFields.MMFFGetMoleculeForceField(mol, ffprop_mmff)
        # ffu = ChemicalForceFields.UFFGetMoleculeForceField(mol)

        # Set angles and constrains for all torsions
        for i, angle in enumerate(angles):

            set_angle = origin_angles[i] + angle

            # Set clockwork angle
            try:
                Chem.rdMolTransforms.SetDihedralDeg(conformer, *torsions[i], set_angle)
            except:
                pass

            # Set forcefield constrain
            eps = 1e-5
            eps = 0.05
            ffc.MMFFAddTorsionConstraint(*torsions[i], False,
                                         set_angle-eps, set_angle+eps, 1.0e6)

            # ffu.UFFAddTorsionConstraint(*torsions[i], False,
            #                             set_angle, set_angle, 1.0e10)

        # minimize constrains
        conv = ffc.Minimize(maxIts=1000, energyTol=1e-2, forceTol=1e-2)
        # conv = ffu.Minimize(maxIts=1000, energyTol=1e-2, forceTol=1e-3)

        if conv == 1:
            # unconverged
            print("unconverged", globalopt)
        else:
            print("converged", globalopt)

        if globalopt:
            forcefield.Minimize(maxIts=1000, energyTol=1e-3, forceTol=1e-3)
            energy = forcefield.CalcEnergy()
        else:
            energy = forcefield.CalcEnergy()

        # Get current positions
        pos = conformer.GetPositions()
        pos -= rmsd.centroid(pos)
        pos = align(pos, origin)

        xyz = rmsd.set_coordinates(atoms, pos)
        f.write(xyz)
        f.write("\n")

        angles = []
        for idxs in torsions:
            angle = Chem.rdMolTransforms.GetDihedralDeg(conformer, *idxs)
            angles.append(angle)

        axis_angles.append(angles)
        axis_energies.append(energy)

    f.close()

    return axis_angles, axis_energies


def get_conformation(mol, res, torsions):
    """

    param:
        rdkit mol
        clockwork resolution
        torsions indexes

    return
        unique conformations

    """

    # Load mol info
    n_atoms = mol.GetNumAtoms()
    n_torsions = len(torsions)
    atoms = mol.GetAtoms()
    atoms = [atom.GetSymbol() for atom in atoms]

    # Setup forcefield for molecule
    # no constraints
    ffprop = ChemicalForceFields.MMFFGetMoleculeProperties(mol)
    forcefield = ChemicalForceFields.MMFFGetMoleculeForceField(mol, ffprop)

    # Get conformer and origin
    conformer = mol.GetConformer()
    origin = conformer.GetPositions()
    origin -= rmsd.centroid(origin)

    # Origin angle
    origin_angles = []

    for idxs in torsions:
        angle = Chem.rdMolTransforms.GetDihedralDeg(conformer, *idxs)
        origin_angles.append(angle)

    # Get resolution angles
    # TODO make iterator of all combos

    for angles in get_angles(res, n_torsions):

        # Reset positions
        for i, pos in enumerate(origin):
            conformer.SetAtomPosition(i, pos)

        # Setup constrained forcefield
        ffc = ChemicalForceFields.MMFFGetMoleculeForceField(mol, ffprop)

        # Set angles and constrains for all torsions
        for i, angle in enumerate(angles):

            set_angle = origin_angles[i] + angle

            # Set clockwork angle
            Chem.rdMolTransforms.SetDihedralDeg(conformer, *torsions[i], set_angle)

            # Set forcefield constrain
            ffc.MMFFAddTorsionConstraint(*torsions[i], False,
                                         set_angle, set_angle, 1.0e10)

        # minimize constrains
        ffc.Minimize(maxIts=1000, energyTol=1e-2, forceTol=1e-3)

        # minimize global
        forcefield.Minimize(maxIts=1000, energyTol=1e-2, forceTol=1e-4)

        # Get current energy
        energy = forcefield.CalcEnergy()

        # Get current positions
        pos = conformer.GetPositions()
        pos -= rmsd.centroid(pos)

        xyz = rmsd.set_coordinates(atoms, pos)

        print(angles, energy)

    return []


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help='', metavar='file')
    args = parser.parse_args()


    ext = args.filename.split(".")[-1]

    if ext == "sdf":

        mol = load_sdf(args.filename)

        # TODO Find all torsion angles

    elif ext == "archive":

        # TODO Guido archive read
        quit()

    else:
        print("unknown format")
        quit()



    torsions = []

    # Pentane
    # torsions.append([0, 1, 5, 8]) # C
    # torsions.append([1, 5, 8, 11]) # C
    # torsions.append([5-1, 1-1, 2-1, 6-1]) # H

    # torsions.append([5-1, 1-1, 2-1, 9-1])
    torsions.append([1-1, 2-1, 6-1, 9-1])
    torsions.append([2-1, 6-1, 9-1, 12-1])
    # torsions.append([6-1, 9-1, 12-1, 16-1])


    # Ethanol
    # torsions.append([9-1, 3-1, 2-1, 1-1])
    # torsions.append([6-1, 1-1, 2-1, 3-1])

    # 044614
    # torsions.append([6-1, 7-1, 8-1, 2-1])
    # torsions.append([6-1, 5-1, 9-1, 15-1])
    # torsions.append([1-1, 2-1, 3-1, 4-1])

    # di-benzene
    # torsions.append([6-1, 1-1, 7-1, 8-1]) # bridge
    # torsions.append([7-1, 8-1, 9-1, 10-1]) # just wrong

    # get_conformation(mol, 3, torsions)

    import matplotlib.pyplot as plt


    axis_angles, axis_energies = scan_angles(mol, 25, torsions, globalopt=False)
    axis_angles = np.array(axis_angles)
    axis_angles = axis_angles.T

    if axis_angles.shape[0] == 1:

        plt.plot(axis_angles[0], axis_energies, '.')

        axis_angles, axis_energies = scan_angles(mol, 25, torsions, globalopt=True)
        axis_angles = np.array(axis_angles)
        axis_angles = axis_angles.T
        plt.plot(axis_angles[0], axis_energies, '.')

        axis_energies = np.array(axis_energies)

        print(np.max(axis_energies), np.argmax(axis_energies))

        # idx, = np.where(axis_energies > 100.0)
        # axis_energies[idx] = 0
        # print(np.max(axis_energies))

        # plt.ylim((-200, 200))
        # plt.xlim((-200, 200))

        plt.plot(axis_angles[0], axis_energies, 'k.')
        plt.savefig("fig_torsion_scan.png")

    else:

        axis_energies = np.array(axis_energies)
        idx, = np.where(axis_energies > 0.0)
        axis_energies[idx] = 0.0

        # print(axis_energies[idx])

        cb1 = plt.scatter(*axis_angles, c=axis_energies,
                    cmap=plt.cm.get_cmap('Blues'))
        plt.colorbar(cb1)

        axis_angles, axis_energies = scan_angles(mol, 10, torsions, globalopt=True)
        axis_angles = np.array(axis_angles)
        axis_angles = axis_angles.T
        plt.scatter(*axis_angles, c="k")

        axis_energies = np.array(axis_energies)


        plt.savefig("fig_torsion_double_scan.png")

        plt.clf()


        # histogram
        bins = np.linspace(-10, 100, 100)
        n, bins, patches = plt.hist(axis_energies, bins)
        plt.savefig("fig_torsion_his_energies.png")



        # for i, (x, y) in enumerate(zip(axis_angles, axis_energies)):
        #     print(i, x, y)


    # for x in range(1, 5):
    #     n = len(list(get_angles(4, x)))
    #     print(x, n)

    return

if __name__ == '__main__':
    main()


