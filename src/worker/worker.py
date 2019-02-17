
import time

import numpy as np

import itertools

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalForceFields

import rmsd
from qml import fchl

import matplotlib.pyplot as plt

import cProfile


def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func



def load_sdf(filename):
    """

    load sdf file and return rdkit mol

    """

    suppl = Chem.SDMolSupplier(filename, removeHs=False, sanitize=True)
    mol = next(suppl)

    return mol




def get_representations_fchl(atoms, coordinates_list):

    replist = []

    for coordinates in coordinates_list:
        rep = fchl.generate_representation(coordinates, atoms, max_size=30, cut_distance=10**6)
        replist.append(rep)

    return replist


def get_kernel_fchl(rep_alpha, rep_beta):

    # sigmas = [0.625, 1.25, 2.5, 5.0, 10.0]
    sigmas = [0.8]

    if id(rep_alpha) == id(rep_beta):

        kernel, = fchl.get_global_symmetric_kernels(rep_alpha,
                kernel_args={"sigma":sigmas},
                cut_distance=10**6,
                alchemy="off")
    else:

        kernel, = fchl.get_global_kernels(rep_alpha, rep_beta,
                kernel_args={"sigma":sigmas},
                cut_distance=10**6,
                alchemy="off")

    return kernel


def exists(origin, new, method="rmsd", threshold=None):
    """

    TODO return what idx in origin new exists in

    """

    threshold = 0.004

    for i, pos in enumerate(origin):

        if method == "rmsd":
            value = rmsd.kabsch_rmsd(pos, new)
        elif method == "fchl":
            value = 5.0

        if value < threshold:
            # Same
            return True

    return False


def unique_from_kernel(kernel, threshold=0.98, symmetric=True):
    """

    Returns uniquenes idx based on second axis.

    """

    N = kernel.shape[0]

    if symmetric:
        unique_idx = [0]
    else:
        unique_idx = []

    for i in range(1, N):

        if symmetric:
            subkernel = kernel[i, unique_idx]
        else:
            subkernel = kernel[i]

        idx, = np.where(subkernel > threshold)

        if len(idx) > 0: continue
        else: unique_idx.append(i)

    return unique_idx


def unique(atoms, coordinates_list, method="rmsd", threshold=None):
    """

    @param
        coordinates_list
        method

    @return
        unique_list

    """

    unique_list = [coordinates_list[0]]
    idx_list = [0]

    if method == "qml":
        replist = []

        for coordinates in coordinates_list:
            rep = fchl.generate_representation(coordinates, atoms, max_size=20, cut_distance=10**6)
            replist.append(rep)

        replist = np.array(replist)

        # fchl uniqueness
        sigmas = [0.625, 1.25, 2.5, 5.0, 10.0]
        sigmas = [0.8]
        fchl_kernels = fchl.get_global_symmetric_kernels(replist, kernel_args={"sigma":sigmas}, cut_distance=10**6, alchemy="off")
        idx_list = unique_from_kernel(fchl_kernels[0])

    elif method == "rmsd":

        threshold = 0.004

        for i, coordinates in enumerate(coordinates_list):
            if not exists(unique_list, coordinates):
                unique_list.append(coordinates)
                idx_list.append(i)

    return idx_list


def unique_energy(global_representations, atoms, positions, energies, debug=False):

    # tol = 1.0e-3

    # find unique
    energies = np.round(energies, decimals=1)

    unique_energies, unique_idx = np.unique(energies, return_index=True)

    n_unique = 0
    rtn_idx = []

    if not global_representations:

        global_representations += list(unique_energies)
        n_unique = len(unique_idx)
        rtn_idx = unique_idx

    else:

        for energy, idx in zip(unique_energies, unique_idx):

            if energy not in global_representations:
                n_unique += 1
                rtn_idx.append(idx)
                global_representations.append(energy)

    return n_unique, rtn_idx


def unique_fchl(global_representations, atoms, positions, energies, debug=False):
    """
    Based on FF atoms, postions, and energies

    merge to global_representations

    return
        new added

    """

    # Calculate uniqueness
    representations_comb = get_representations_fchl(atoms, positions)
    representations_comb = np.array(representations_comb)

    kernel_comb = get_kernel_fchl(representations_comb, representations_comb)
    unique_comb = unique_from_kernel(kernel_comb)

    representations_comb = representations_comb[unique_comb]
    kernel_comb = kernel_comb[np.ix_(unique_comb, unique_comb)]

    # Merge to global
    if not global_representations:

        global_representations += list(representations_comb)
        n_unique = len(representations_comb)
        uidxs = unique_comb

        if debug: print("debug: init global_representations. {:} conformations.".format(len(global_representations)))

    else:

        new_kernel_overlap = get_kernel_fchl(np.array(representations_comb), np.array(global_representations))
        uidxs = unique_from_kernel(new_kernel_overlap, symmetric=False)
        n_unique = len(uidxs)

        if n_unique != 0:

            global_representations += list(representations_comb[uidxs])

    return n_unique, uidxs


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
    axis_pos = []

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
        axis_pos.append(pos)


    f.close()

    return axis_angles, axis_energies


def get_forcefield(mol):

    ffprop = ChemicalForceFields.MMFFGetMoleculeProperties(mol)
    forcefield = ChemicalForceFields.MMFFGetMoleculeForceField(mol, ffprop) # 0.01 overhead

    return ffprop, forcefield


def run_forcefield(ff, steps, energy=1e-2, force=1e-3):

    status = ff.Minimize(maxIts=steps, energyTol=energy, forceTol=force)

    return status


def run_forcefield2(ff, steps, energy=1e-2, force=1e-3):

    status = ff.Minimize(maxIts=steps, energyTol=energy, forceTol=force)

    return status



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
    n_torsions = len(torsions)

    # init energy
    energies = []

    # no constraints
    ffprop, forcefield = get_forcefield(mol)

    # Get conformer and origin
    conformer = mol.GetConformer()
    origin = conformer.GetPositions()
    origin -= rmsd.centroid(origin)

    # Origin angle
    origin_angles = []

    # type of idxs
    torsions = [[int(y) for y in x] for x in torsions]

    for idxs in torsions:
        angle = Chem.rdMolTransforms.GetDihedralDeg(conformer, *idxs)
        origin_angles.append(angle)

    # Axis holder
    axis_pos = []

    # Get resolution angles
    for angles in get_angles(res, n_torsions):

        # Reset positions
        for i, pos in enumerate(origin):
            conformer.SetAtomPosition(i, pos)

        # Setup constrained forcefield
        ffc = ChemicalForceFields.MMFFGetMoleculeForceField(mol, ffprop)

        # Set angles and constrains for all torsions
        for i, angle in enumerate(angles):

            set_angle = origin_angles[i] + angle

            try:
                # Set clockwork angle
                Chem.rdMolTransforms.SetDihedralDeg(conformer, *torsions[i], set_angle)
            except:
                pass

            # Set forcefield constrain
            ffc.MMFFAddTorsionConstraint(*torsions[i], False,
                                         set_angle, set_angle, 1.0e10)


        # minimize constrains
        # ffc.Minimize(maxIts=500, energyTol=1e-2, forceTol=1e-3)
        status = run_forcefield(ffc, 500)

        # minimize global
        # status = forcefield.Minimize(maxIts=700, energyTol=1e-2, forceTol=1e-4)
        status = run_forcefield2(forcefield, 700, force=1e-4)

        # Get current energy
        energy = forcefield.CalcEnergy()

        # Get current positions
        pos = conformer.GetPositions()
        # pos -= rmsd.centroid(pos)
        # pos = align(pos, origin)

        axis_pos += [pos]
        energies += [energy]

        # now_angles = []
        # for idxs in torsions:
        #     angle = Chem.rdMolTransforms.GetDihedralDeg(conformer, *idxs)
        #     now_angles.append(angle)

    return energies, axis_pos


def get_torsion_atoms(mol, torsion):
    """
    """

    atoms = mol.GetAtoms()
    # atoms = list(atoms)
    atoms = [atom.GetSymbol() for atom in atoms]
    atoms = np.array(atoms)
    atoms = atoms[torsion]

    return atoms


def get_torsions(mol):
    """ return idx of all torsion pairs
    All heavy atoms, and one end can be a hydrogen
    """

    any_atom = "[*]"
    not_hydrogen = "[!H]"

    smarts = [
        any_atom,
        any_atom,
        any_atom,
        any_atom]

    smarts = "~".join(smarts)

    idxs = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
    idxs = [list(x) for x in idxs]
    idxs = np.array(idxs)

    rtnidxs = []

    for idx in idxs:

        atoms = get_torsion_atoms(mol, idx)
        atoms = np.array(atoms)
        idxh, = np.where(atoms == "H")

        if idxh.shape[0] > 1: continue
        elif idxh.shape[0] > 0:
            if idxh[0] == 1: continue
            if idxh[0] == 2: continue

        rtnidxs.append(idx)

    return np.array(rtnidxs, dtype=int)


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help='', metavar='file')

    parser.add_argument('--unique', action="store", help='[none, rmsd, fchl, energy]', default="fchl")

    parser.add_argument('--torsion-body', nargs="+", default=[3], action="store", help="how many torsions to scan at the same time", metavar="int", type=int)
    parser.add_argument('--torsion-res', nargs="+", default=[4], action="store", help="Resolution of clockwork scan", metavar="int", type=int)


    parser.add_argument('--plot-convergence')


    parser.add_argument('--scan', action="store_true", help='')

    parser.add_argument('--debug', action="store_true", help='')

    args = parser.parse_args()

    # Read the file
    # archive or sdf

    ext = args.filename.split(".")[-1]

    if ext == "sdf":

        # load molecule from filename
        mol = load_sdf(args.filename)

        if args.debug: print("debug: loaded sdf file")

        # Find all torsion angles
        torsions = get_torsions(mol)

        if args.debug: print("debug: found {:} torsions".format(torsions.shape[0]))

    elif ext == "archive":

        # TODO Guido archive read
        print('not implemented yet')
        quit()

    else:
        print("unknown format")
        quit()



    if not args.scan:

        # intel on molecule
        atoms = [atom for atom in mol.GetAtoms()]
        atoms_str = [atom.GetSymbol() for atom in atoms]
        atoms_int = [atom.GetAtomicNum() for atom in atoms]

        atoms_str = np.array(atoms_str)
        atoms_int = np.array(atoms_int)

        # init global arrays
        global_energies = []
        global_positions = []

        # uniquenes representations
        global_representations = []

        if args.unique == "fchl":

            unique_method = unique_fchl

        elif args.unique == "energy":

            unique_method = unique_energy

        else:
            print(args.unique, "not implemented")
            quit()


        # found
        n_counter_all = 0
        n_counter_unique = 0
        n_counter_list = []
        n_counter_flag = []

        # TODO Need to check clockwork to exclude
        # 0 - 0
        # 0 - 1
        # or anything that is done with lower-body iterators

        for torsion_body in args.torsion_body:

            if args.debug: print("debug: scanning {:}-body torsions".format(torsion_body))

            for res in args.torsion_res:

                torsion_iterator = itertools.combinations(list(range(len(torsions))), torsion_body)
                for i, idx in enumerate(torsion_iterator):

                    idx = list(idx)
                    torsions_comb = [list(x) for x in torsions[idx]]

                    here = time.time()

                    energies, positions = get_conformation(mol, res, torsions_comb)

                    n_counter_all += len(energies)

                    if args.debug: print("debug: work {:} res {:} - got {:} conformations - time {:}".format(i, res, len(energies), time.time()-here))

                    # Calculate uniqueness

                    n_unique, unique_idx = unique_method(global_representations, atoms_int, positions, energies, debug=args.debug)
                    n_counter_unique += n_unique

                    if n_unique > 0:
                        if args.debug: print("found {:} new conformers {:}".format(n_unique, np.array(energies)[unique_idx]))

                        energies = np.array(energies)
                        positions = np.array(positions)

                        # append global
                        global_energies += list(energies[unique_idx])
                        global_positions += list(positions[unique_idx])

                    # administration
                    n_counter_list.append(n_counter_unique)
                    n_counter_flag.append(torsion_body)


        print("found {:} unique conformations".format(len(global_representations)))
        qml_n, idx_qml = unique_fchl([], atoms_int, global_positions, global_energies)
        print("found {:} unique qml conformations".format(qml_n))

        bins = np.linspace(min(global_energies)-10, max(global_energies)+10, 50)
        n, bins, patches = plt.hist(global_energies, bins)
        plt.savefig("fig_conf_his_energies.png")
        plt.clf()

        # convergence

        n_counter_list = np.array(n_counter_list)
        x_axis = np.arange(n_counter_list.shape[0])
        x_flags = np.unique(n_counter_flag)

        for flag in x_flags:

            y_view = np.where(n_counter_flag == flag)

            plt.plot(x_axis[y_view], n_counter_list[y_view], ".")


        plt.savefig("fig_conf_convergence.png")

        print("out of total {:} minimizations".format(n_counter_all))



    else:

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




    return

if __name__ == '__main__':
    main()


