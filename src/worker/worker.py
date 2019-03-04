
import gzip
import json

import itertools
import time

# import matplotlib.pyplot as plt
import numpy as np
from qml import fchl

from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalForceFields

import adm_merge
import adm_molecule

def load_sdf(filename):
    """

    load sdf file and return rdkit mol list

    args:
        filename sdf

    return:
        list of mol objs

    """

    suppl = Chem.SDMolSupplier(filename,
                               removeHs=False,
                               sanitize=True)

    mollist = [mol for mol in suppl]

    return mollist


def load_sdf_file(obj):

    suppl = Chem.ForwardSDMolSupplier(obj,
                               removeHs=False,
                               sanitize=True)

    mollist = [mol for mol in suppl]

    return mollist


def get_representations_fchl(atoms, coordinates_list):

    replist = []

    for coordinates in coordinates_list:
        rep = fchl.generate_representation(coordinates, atoms, max_size=30, cut_distance=10**6)
        replist.append(rep)

    return replist


def get_kernel_fchl(rep_alpha, rep_beta):

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

    if res == 0:
        start = 0
        step = 360
        n_steps = 1

    else:

        # res = max([res, 1])

        start = 360.0 / 2.0 ** (res)
        step = 360.0 / 2.0 ** (res-1)
        n_steps = 2 ** (res - 1)

    if debug:
        print(res, step, n_steps, start)

    return start, step, n_steps


def get_angles(res, num_torsions):
    """

    Setup angle iterator based on number of torsions

    """

    if type(res) == int:
        res = [res]*num_torsions

    angles = []

    for r in res:
        start, step, n_steps = clockwork(r)
        scan = np.arange(start, start+step*n_steps, step)
        angles.append(scan)

    iterator = itertools.product(*angles)

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
        clockwork resolutions
        torsions indexes

    return
        unique conformations

    """

    # Load mol info
    n_torsions = len(torsions)

    # init energy
    energies = []
    states = []

    # no constraints
    ffprop, forcefield = get_forcefield(mol)

    # Get conformer and origin
    conformer = mol.GetConformer()
    origin = conformer.GetPositions()
    # TODO origin -= rmsd.centroid(origin)

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
    angle_iterator = get_angles(res, n_torsions)

    for angles in angle_iterator:

        # Reset positions
        for i, pos in enumerate(origin):
            conformer.SetAtomPosition(i, pos)

        # Setup constrained forcefield
        ffc = ChemicalForceFields.MMFFGetMoleculeForceField(mol, ffprop)

        # Set angles and constrains for all torsions
        for i, angle in enumerate(angles):

            set_angle = origin_angles[i] + angle

            # Set clockwork angle
            try: Chem.rdMolTransforms.SetDihedralDeg(conformer, *torsions[i], set_angle)
            except: pass

            # Set forcefield constrain
            ffc.MMFFAddTorsionConstraint(*torsions[i], False,
                                         set_angle, set_angle, 1.0e10)

        # minimize constrains
        status = run_forcefield(ffc, 500)

        # minimize global
        status = run_forcefield2(forcefield, 700, force=1e-4)

        # Get current energy
        energy = forcefield.CalcEnergy()

        # Get current positions
        pos = conformer.GetPositions()

        axis_pos += [pos]
        energies += [energy]
        states += [status]

    return energies, axis_pos, states


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


def conformers(mol, torsions, resolutions, unique_method=unique_energy, debug=False):

    # time
    if debug:
        here = time.time()

    # Find all
    energies, positions, states = get_conformation(mol, resolutions, torsions)

    # Ignore unconverged
    states = np.array(states)
    success = np.argwhere(states == 0)
    success = success.flatten()

    energies = np.array(energies)
    positions = np.array(positions)

    N = energies.shape[0]

    energies = energies[success]
    positions = positions[success]

    # Find uniques
    u_len, u_idx = unique_method([], [], positions, energies, debug=debug)

    # timestamp = N/ (time.time() - here)
    if debug:
        timestamp = (time.time() - here)
        print("debug: {:}, converged={:}, speed={:5.1f}, new={:}".format("", N, timestamp, np.round(energies, 1)))

    return energies, positions


def run_jobs(moldb, tordb, jobs, debug=False, dump_sdf=None):
    """
    calculate conformers

    args:
        list of molecules
        list of molecule torsions
        list of str jobs

    return:
        str results conformers

    """

    if type(jobs) is str:
        jobs = jobs.split(";")

    # we assume all jobs are the same molecule
    molidx = jobs[0].split(",")[0]
    molidx = int(molidx)
    mol = moldb[molidx]

    atoms = [atom for atom in mol.GetAtoms()]
    atoms_str = [atom.GetSymbol() for atom in atoms]
    atoms_int = [atom.GetAtomicNum() for atom in atoms]

    results = []

    for job in jobs:

        job = job.split(",")
        idx_mol = job[0]
        idx_mol = int(idx_mol)
        idx_torsions = job[1].split()
        idx_torsions = [int(x) for x in idx_torsions]
        resolution = job[2]
        resolution = int(resolution)

        torsions = tordb[idx_mol][idx_torsions]

        origin = [molidx, idx_torsions]

        jobresults = run_job(mol, torsions, resolution, atoms_int, origin, debug=debug)

        results += jobresults

    # wp level merge
    if len(results) == 0:
        print(results, job)

    results = adm_merge.energyandfchl(results, atoms_int, debug=debug)

    #
    # if dump_sdf is not None:
    #     print("saving results to", dump_sdf)
    #     adm_molecule.save_results(mol, results, dump_sdf)

    # encode and send back
    for i, result in enumerate(results):
        result = json.dumps(result)
        result = result.replace(" ", "")
        results[i] = result

    results = "\n".join(results)

    return results, None


def run_job(mol, torsions, resolution, atoms, origin, debug=False):


    N_torsions = len(torsions)

    # generator all combinations for this level of torsions

    rest = range(0, resolution)
    rest = list(rest) + [resolution]

    combinations = itertools.product(rest, repeat=N_torsions)
    combinations = list(combinations)
    combinations = [list(x) for x in combinations if resolution in x]

    rtnresults = []

    stamp1 = time.time()

    for resolutions in combinations:

        energies, positions = conformers(mol, torsions, resolutions)

        results = []

        for energy, coord in zip(energies, positions):
            result = [*origin, resolutions, 0, energy, np.round(coord, 5).flatten().tolist()]
            results.append(result)

        # merge job
        if len(results) == 0:
            continue

        results = adm_merge.energyandfchl(results, atoms)

        if debug:
            print("job", resolutions, len(energies), "->", len(results), np.round(energies, decimals=1).tolist())

        rtnresults += results

    stamp2 = time.time()

    if debug:
        print("timestamp {:4.2f}".format(stamp2-stamp1), len(rtnresults))

    # merge job job
    rtnresults = adm_merge.energyandfchl(rtnresults, atoms, debug=debug)

    return rtnresults


def run_jobs_nein(moldb, tordb, jobs, debug=False):

    # if str from redit, convert to list
    if type(jobs) != list: jobs = eval(jobs.decode())

    data = []

    # assume same molecule for all jobs!
    molidx = jobs[0].split(",")[0]
    molidx = int(molidx)
    mol = moldb[molidx]

    atoms = [atom for atom in mol.GetAtoms()]
    atoms_str = [atom.GetSymbol() for atom in atoms]
    atoms_int = [atom.GetAtomicNum() for atom in atoms]

    for job in jobs:

        here = time.time()

        job = job.split(",")

        mol_idx = int(job[0])
        torsions_idx = job[1].split()
        resolutions = job[2].split()

        torsions_idx = [int(x) for x in torsions_idx]
        resolutions = [int(x) for x in resolutions]

        mol = moldb[mol_idx]
        torsions = tordb[mol_idx]

        # isolate torsions
        torsions = torsions[torsions_idx]

        # minimize
        energies, positions = conformers(moldb[mol_idx], torsions, resolutions)

        origin = [mol_idx, torsions_idx, resolutions]

        jobdata = []

        for energy, coord in zip(energies, positions):
            # job syntax
            result = [*origin, 0, energy, np.round(coord, 5).flatten().tolist()]
            jobdata.append(result)

        # merge job-wiese
        jobdata = merge.energyandfchl(jobdata, atoms_int)
        data += jobdata

        if debug:
            now = time.time() - here
            speed = len(energies) / now
            print("calculated", job, len(data), "time={:5.2f}, speed={:5.2f}".format(now, speed))

    # Merge same conformers, by energy
    # merge wp-wise
    if debug:
        here = time.time()
    data = merge.energyandfchl(data, atoms_int, debug=debug)
    if debug:
        print("merged in", time.time()-here, "sec")

    for i, result in enumerate(data):
        result = json.dumps(result)
        result = result.replace(" ", "")
        data[i] = result

    data = "\n".join(data)

    return data, None


def wraprunjobs(moldb, tordb, jobs, debug=False):

    # try:
    rtn = run_jobs(moldb, tordb, jobs, debug=debug)
    # except:
    #     rtn = ([], "arrgh it crashed")

    return rtn


def getthoseconformers(mol, torsions, torsion_bodies, clockwork_resolutions, debug=False, unique_method=unique_energy):

    # intel on molecule
    atoms = [atom for atom in mol.GetAtoms()]
    atoms_str = [atom.GetSymbol() for atom in atoms]
    atoms_int = [atom.GetAtomicNum() for atom in atoms]
    atoms_str = np.array(atoms_str)
    atoms_int = np.array(atoms_int)

    # init global arrays
    global_energies = []
    global_positions = []
    global_origins = []

    # uniquenes representation (fchl, energy and rmsd)
    global_representations = []

    # TODO Need to check clockwork to exclude
    # 0 - 0
    # 0 - 1
    # or anything that is done with lower-body iterators

    # found
    n_counter_all = 0
    n_counter_unique = 0
    n_counter_list = []
    n_counter_flag = []

    for torsion_body in torsion_bodies:
        for res in clockwork_resolutions:

            # TODO this is basically moved to workpackages
            # with the res as well
            torsion_iterator = itertools.combinations(list(range(len(torsions))), torsion_body)

            for i, idx in enumerate(torsion_iterator):

                idx = list(idx)
                torsions_comb = [list(x) for x in torsions[idx]]

                if debug:
                    here = time.time()

                energies, positions, states = get_conformation(mol, res, torsions_comb)

                N = len(energies)
                n_counter_all += N

                # Calculate uniqueness
                n_unique, unique_idx = unique_method(global_representations, atoms_int, positions, energies, debug=debug)
                n_counter_unique += n_unique

                if n_unique > 0:

                    energies = np.array(energies)
                    positions = np.array(positions)

                    # append global
                    global_energies += list(energies[unique_idx])
                    global_positions += list(positions[unique_idx])

                    # TODO is this enough?
                    global_origins += [(res, i)]

                    # print("states", states, np.round(energies, 2)[unique_idx])


                if debug:

                    workname = "n{:} r{:} t{:}".format(torsion_body, res, i)
                    # timestamp = N/ (time.time() - here)
                    timestamp = (time.time() - here)
                    print("debug: {:}, converged={:}, time={:5.1f}, new={:}".format(workname, N, timestamp, np.round(energies, 2)[unique_idx]))


                # administration
                n_counter_list.append(n_counter_unique)
                n_counter_flag.append(torsion_body)


    if debug: print("found {:} unique conformations".format(len(global_representations)))
    qml_n, idx_qml = unique_fchl([], atoms_int, global_positions, global_energies)
    if debug: print("found {:} unique qml conformations".format(qml_n))

    # convergence
    # n_counter_list = np.array(n_counter_list)
    # x_axis = np.arange(n_counter_list.shape[0])
    # x_flags = np.unique(n_counter_flag)
    #
    # for flag in x_flags:
    #     y_view = np.where(n_counter_flag == flag)
    #     plt.plot(x_axis[y_view], n_counter_list[y_view], ".")

    # plt.savefig("fig_conf_convergence.png")

    print("out of total {:} minimizations".format(n_counter_all))

    return global_energies, global_positions, global_origins


def main():

    # TODO restartable
    # - from .archive file
    # - from reddis

    # TODO Worker from one SDF file
    # - add mol lib

    # TODO Sane defaults

    # TODO Check energy and fchl are the same?

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action="store_true", help='')

    # Read molecules
    parser.add_argument('-f', '--filename', type=str, help='', metavar='file')
    parser.add_argument('--torsions-file', type=str, help='', metavar='file', default=None)

    # Do a local scan
    parser.add_argument('--unique', action="store", help='Method to find uniqeness of conformers [none, rmsd, fchl, energy]', default="energy")
    parser.add_argument('--torsion-body', nargs="+", default=[3], action="store", help="How many torsions to scan at the same time", metavar="int", type=int)
    parser.add_argument('--torsion-res', nargs="+", default=[4], action="store", help="Resolution of clockwork scan", metavar="int", type=int)

    # How to calculate the jobs, local or server?
    parser.add_argument('--read-job', action="store_true", help='')
    parser.add_argument('--redis-connect', help="connection to redis server")
    parser.add_argument('--redis-task', help="redis task name")

    args = parser.parse_args()

    # Read the file
    # archive or sdf
    ext = args.filename.split(".")[-1]

    if ext == "sdf":

        moldb = load_sdf(args.filename)

    elif ext == "gz":

        inf = gzip.open(args.filename)
        moldb = load_sdf_file(inf)

    else:

        print("unknown format")
        quit()


    tordb = []

    if args.torsions_file is None:

        # make db on worker
        for mol in moldb:

            torsions = get_torsions(mol)
            torsions = np.array(torsions)
            tordb.append(torsions)

    else:

        # Read file and make db
        with open(args.torsions_file) as f:

            for line in f:

                line = eval(line)
                line = np.array(line)
                tordb.append(line)


    # Get some conformations
    if args.redis_connect is not None:

        import os
        import rediscomm as rediswrap

        # make queue
        tasks = rediswrap.Taskqueue(args.redis_connect, args.redis_task)

        if args.debug:
            print("connected to redis")

        do_work = lambda x: wraprunjobs(moldb, tordb, x, debug=args.debug)
        tasks.main_loop(do_work)


    else:

        # Read archive workers from stdin

        workpackage = "2405,8 6 4,3"
        workpackage += ";" + "2405,8 6 4,1"
        workpackage += ";" + "2405,8 6 4,2"
        workpackage += ";" + "2405,8 6 5,3"
        workpackage += ";" + "2405,8 6 3,3"
        workpackage += ";" + "2405,8 6 2,3"

        workpackage = "2405,8 6 4,2"

        workpackage = "0,39 40,0;0,39 40,1;0,39 40,2"

        run_jobs(moldb, tordb, workpackage, dump_sdf="test.sdf", debug=debug)

        quit()

    return

if __name__ == '__main__':
    main()



    # if False:
    #
    #     # intel on molecule
    #     atoms = [atom for atom in mol.GetAtoms()]
    #     atoms_str = [atom.GetSymbol() for atom in atoms]
    #     atoms_int = [atom.GetAtomicNum() for atom in atoms]
    #
    #     atoms_str = np.array(atoms_str)
    #     atoms_int = np.array(atoms_int)
    #
    #     # init global arrays
    #     global_energies = []
    #     global_positions = []
    #
    #     # uniquenes representations
    #     global_representations = []
    #
    #     if args.unique == "fchl":
    #
    #         unique_method = unique_fchl
    #
    #     elif args.unique == "energy":
    #
    #         unique_method = unique_energy
    #
    #     else:
    #         print(args.unique, "not implemented")
    #         quit()
    #
    #
    #     energies, positions = getthoseconformers(mol, torsions, args.torsion_body, args.torsion_res,
    #         debug=args.debug,
    #         unique_method=unique_method)
    #
    #     print(energies)
    #
    #     f = open("test.xyz", 'w')
    #
    #     for pos in positions:
    #
    #         xyz = rmsd.set_coordinates(atoms_str, pos)
    #         f.write(xyz)
    #         f.write("\n")
    #
    #     f.close()
