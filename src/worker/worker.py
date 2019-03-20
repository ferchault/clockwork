
import traceback

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

    # Forcefield generation failed
    # TODO What to do?
    if forcefield is None:
        return [], [], []

    # Get conformer and origin
    conformer = mol.GetConformer()
    origin = conformer.GetPositions()

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
    energies = []

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

        job_energies = []

        for result in jobresults:
            job_energies.append(result[4])

        job_energies = np.round(job_energies, 1)

        # continuesly merge
        results = adm_merge.energyandfchl(results, atoms_int, debug=debug, only_energies=job_energies)

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

        if len(energies) == 0:
            continue

        results = []

        for energy, coord in zip(energies, positions):
            result = [*origin, resolutions, 0, energy, np.round(coord, 5).flatten().tolist()]
            results.append(result)

        results = adm_merge.energyandfchl(results, atoms)

        if debug:
            print("job", resolutions, len(energies), "->", len(results), np.round(energies, decimals=1).tolist())

        rtnresults += results

    stamp2 = time.time()

    if debug:
        print("timestamp {:4.2f}".format(stamp2-stamp1), len(rtnresults))

    if len(rtnresults) == 0:
        return rtnresults

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


    f = open("jobs_log", "a+")

    f.write("\n")
    f.write(jobs)

    stamp1 = time.time()

    try:
        rtn = run_jobs(moldb, tordb, jobs, debug=debug)
    except:
        error = traceback.format_exc()
        print(error)
        rtn = ("", error)

    stamp2 = time.time()

    f.close()

    print("workpackage done {:5.3f}".format(stamp2-stamp1))

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


    # elif args.workpackages:

    else:

        wp = "0,0,0,1;0,0,1,1;0,0,2,1;0,1,0,1;0,1,1,1;0,1,2,1;0,2,0,1;0,2,1,1;0,2,2,1;0,3,0,1;0,3,1,1;0,3,2,1;0,4,0,1;0,4,1,1;0,4,2,1;0,5,0,1;0,5,1,1;0,5,2,1;0,6,0,1;0,6,1,1;0,6,2,1;0,7,0,1;0,7,1,1;0,7,2,1;0,8,0,1;0,8,1,1;0,8,2,1;0,9,0,1;0,9,1,1;0,9,2,1;0,10,0,1;0,10,1,1;0,10,2,1;0,11,0,1;0,11,1,1;0,11,2,1;0,12,0,1;0,12,1,1;0,12,2,1;0,13,0,1;0,13,1,1;0,13,2,1;0,14,0,1;0,14,1,1;0,14,2,1;0,15,0,1;0,15,1,1;0,15,2,1;0,16,0,1;0,16,1,1;0,16,2,1;0,17,0,1;0,17,1,1;0,17,2,1;0,18,0,1;0,18,1,1;0,18,2,1;0,19,0,1;0,19,1,1;0,19,2,1;0,20,0,1;0,20,1,1;0,20,2,1;0,21,0,1;0,21,1,1;0,21,2,1;0,22,0,1;0,22,1,1;0,22,2,1;0,23,0,1;0,23,1,1;0,23,2,1;0,24,0,1;0,24,1,1;0,24,2,1;0,25,0,1;0,25,1,1;0,25,2,1;0,26,0,1;0,26,1,1;0,26,2,1;0,27,0,1;0,27,1,1;0,27,2,1;0,28,0,1;0,28,1,1;0,28,2,1;0,29,0,1;0,29,1,1;0,29,2,1;0,30,0,1;0,30,1,1;0,30,2,1;0,31,0,1;0,31,1,1;0,31,2,1;0,32,0,1;0,32,1,1;0,32,2,1;0,33,0,1;0,33,1,1;0,33,2,1;0,34,0,1;0,34,1,1;0,34,2,1;0,35,0,1;0,35,1,1;0,35,2,1;0,36,0,1;0,36,1,1;0,36,2,1;0,37,0,1;0,37,1,1;0,37,2,1;0,38,0,1;0,38,1,1;0,38,2,1;0,39,0,1;0,39,1,1;0,39,2,1;0,40,0,1;0,40,1,1;0,40,2,1;0,0 1,0,1;0,0 1,1,3;0,0 1,2,5;0,0 2,0,1;0,0 2,1,3;0,0 2,2,5;0,0 3,0,1;0,0 3,1,3;0,0 3,2,5;0,0 4,0,1;0,0 4,1,3;0,0 4,2,5;0,0 5,0,1;0,0 5,1,3;0,0 5,2,5;0,0 6,0,1;0,0 6,1,3;0,0 6,2,5;0,0 7,0,1;0,0 7,1,3;0,0 7,2,5;0,0 8,0,1;0,0 8,1,3;0,0 8,2,5;0,0 9,0,1;0,0 9,1,3;0,0 9,2,5;0,0 10,0,1;0,0 10,1,3;0,0 10,2,5;0,0 11,0,1;0,0 11,1,3;0,0 11,2,5;0,0 12,0,1;0,0 12,1,3;0,0 12,2,5;0,0 13,0,1;0,0 13,1,3;0,0 13,2,5;0,0 14,0,1;0,0 14,1,3;0,0 14,2,5;0,0 15,0,1;0,0 15,1,3;0,0 15,2,5;0,0 16,0,1;0,0 16,1,3;0,0 16,2,5;0,0 17,0,1;0,0 17,1,3;0,0 17,2,5;0,0 18,0,1;0,0 18,1,3;0,0 18,2,5;0,0 19,0,1;0,0 19,1,3;0,0 19,2,5;0,0 20,0,1;0,0 20,1,3;0,0 20,2,5;0,0 21,0,1;0,0 21,1,3;0,0 21,2,5;0,0 22,0,1;0,0 22,1,3;0,0 22,2,5;0,0 23,0,1;0,0 23,1,3;0,0 23,2,5;0,0 24,0,1;0,0 24,1,3;0,0 24,2,5;0,0 25,0,1;0,0 25,1,3;0,0 25,2,5;0,0 26,0,1;0,0 26,1,3;0,0 26,2,5;0,0 27,0,1;0,0 27,1,3;0,0 27,2,5;0,0 28,0,1;0,0 28,1,3;0,0 28,2,5;0,0 29,0,1;0,0 29,1,3;0,0 29,2,5;0,0 30,0,1;0,0 30,1,3;0,0 30,2,5;0,0 31,0,1;0,0 31,1,3;0,0 31,2,5;0,0 32,0,1;0,0 32,1,3;0,0 32,2,5;0,0 33,0,1;0,0 33,1,3;0,0 33,2,5;0,0 34,0,1;0,0 34,1,3;0,0 34,2,5;0,0 35,0,1;0,0 35,1,3;0,0 35,2,5;0,0 36,0,1;0,0 36,1,3;0,0 36,2,5;0,0 37,0,1;0,0 37,1,3;0,0 37,2,5;0,0 38,0,1;0,0 38,1,3;0,0 38,2,5;0,0 39,0,1;0,0 39,1,3;0,0 39,2,5;0,0 40,0,1;0,0 40,1,3;0,0 40,2,5;0,1 2,0,1;0,1 2,1,3;0,1 2,2,5;0,1 3,0,1;0,1 3,1,3;0,1 3,2,5;0,1 4,0,1;0,1 4,1,3;0,1 4,2,5;0,1 5,0,1;0,1 5,1,3;0,1 5,2,5;0,1 6,0,1;0,1 6,1,3;0,1 6,2,5;0,1 7,0,1;0,1 7,1,3;0,1 7,2,5;0,1 8,0,1;0,1 8,1,3;0,1 8,2,5;0,1 9,0,1;0,1 9,1,3;0,1 9,2,5;0,1 10,0,1;0,1 10,1,3;0,1 10,2,5;0,1 11,0,1;0,1 11,1,3;0,1 11,2,5;0,1 12,0,1;0,1 12,1,3;0,1 12,2,5;0,1 13,0,1;0,1 13,1,3;0,1 13,2,5;0,1 14,0,1;0,1 14,1,3;0,1 14,2,5;0,1 15,0,1;0,1 15,1,3;0,1 15,2,5;0,1 16,0,1;0,1 16,1,3;0,1 16,2,5;0,1 17,0,1;0,1 17,1,3;0,1 17,2,5;0,1 18,0,1;0,1 18,1,3;0,1 18,2,5;0,1 19,0,1;0,1 19,1,3;0,1 19,2,5;0,1 20,0,1;0,1 20,1,3;0,1 20,2,5;0,1 21,0,1;0,1 21,1,3;0,1 21,2,5;0,1 22,0,1;0,1 22,1,3;0,1 22,2,5;0,1 23,0,1;0,1 23,1,3;0,1 23,2,5;0,1 24,0,1;0,1 24,1,3;0,1 24,2,5;0,1 25,0,1;0,1 25,1,3;0,1 25,2,5;0,1 26,0,1;0,1 26,1,3;0,1 26,2,5;0,1 27,0,1;0,1 27,1,3;0,1 27,2,5;0,1 28,0,1;0,1 28,1,3;0,1 28,2,5;0,1 29,0,1;0,1 29,1,3;0,1 29,2,5;0,1 30,0,1;0,1 30,1,3;0,1 30,2,5;0,1 31,0,1;0,1 31,1,3;0,1 31,2,5;0,1 32,0,1;0,1 32,1,3;0,1 32,2,5;0,1 33,0,1;0,1 33,1,3;0,1 33,2,5;0,1 34,0,1;0,1 34,1,3;0,1 34,2,5;0,1 35,0,1;0,1 35,1,3;0,1 35,2,5;0,1 36,0,1;0,1 36,1,3;0,1 36,2,5;0,1 37,0,1;0,1 37,1,3;0,1 37,2,5;0,1 38,0,1;0,1 38,1,3;0,1 38,2,5;0,1 39,0,1;0,1 39,1,3;0,1 39,2,5;0,1 40,0,1;0,1 40,1,3;0,1 40,2,5;0,2 3,0,1;0,2 3,1,3;0,2 3,2,5;0,2 4,0,1;0,2 4,1,3;0,2 4,2,5;0,2 5,0,1;0,2 5,1,3;0,2 5,2,5;0,2 6,0,1;0,2 6,1,3;0,2 6,2,5;0,2 7,0,1;0,2 7,1,3;0,2 7,2,5;0,2 8,0,1;0,2 8,1,3;0,2 8,2,5;0,2 9,0,1;0,2 9,1,3;0,2 9,2,5;0,2 10,0,1;0,2 10,1,3;0,2 10,2,5;0,2 11,0,1;0,2 11,1,3;0,2 11,2,5;0,2 12,0,1;0,2 12,1,3;0,2 12,2,5;0,2 13,0,1;0,2 13,1,3;0,2 13,2,5;0,2 14,0,1;0,2 14,1,3;0,2 14,2,5;0,2 15,0,1;0,2 15,1,3;0,2 15,2,5;0,2 16,0,1;0,2 16,1,3;0,2 16,2,5;0,2 17,0,1;0,2 17,1,3;0,2 17,2,5;0,2 18,0,1;0,2 18,1,3;0,2 18,2,5;0,2 19,0,1;0,2 19,1,3;0,2 19,2,5;0,2 20,0,1;0,2 20,1,3;0,2 20,2,5;0,2 21,0,1;0,2 21,1,3;0,2 21,2,5;0,2 22,0,1;0,2 22,1,3;0,2 22,2,5;0,2 23,0,1;0,2 23,1,3;0,2 23,2,5;0,2 24,0,1;0,2 24,1,3;0,2 24,2,5;0,2 25,0,1;0,2 25,1,3;0,2 25,2,5;0,2 26,0,1;0,2 26,1,3;0,2 26,2,5;0,2 27,0,1;0,2 27,1,3;0,2 27,2,5;0,2 28,0,1;0,2 28,1,3;0,2 28,2,5;0,2 29,0,1;0,2 29,1,3;0,2 29,2,5;0,2 30,0,1;0,2 30,1,3;0,2 30,2,5;0,2 31,0,1;0,2 31,1,3;0,2 31,2,5;0,2 32,0,1;0,2 32,1,3;0,2 32,2,5;0,2 33,0,1;0,2 33,1,3;0,2 33,2,5;0,2 34,0,1;0,2 34,1,3;0,2 34,2,5;0,2 35,0,1;0,2 35,1,3;0,2 35,2,5;0,2 36,0,1;0,2 36,1,3;0,2 36,2,5;0,2 37,0,1;0,2 37,1,3;0,2 37,2,5;0,2 38,0,1;0,2 38,1,3;0,2 38,2,5;0,2 39,0,1;0,2 39,1,3;0,2 39,2,5;0,2 40,0,1;0,2 40,1,3;0,2 40,2,5;0,3 4,0,1;0,3 4,1,3;0,3 4,2,5;0,3 5,0,1;0,3 5,1,3;0,3 5,2,5;0,3 6,0,1;0,3 6,1,3;0,3 6,2,5;0,3 7,0,1;0,3 7,1,3;0,3 7,2,5;0,3 8,0,1;0,3 8,1,3;0,3 8,2,5;0,3 9,0,1;0,3 9,1,3;0,3 9,2,5;0,3 10,0,1;0,3 10,1,3;0,3 10,2,5;0,3 11,0,1;0,3 11,1,3;0,3 11,2,5;0,3 12,0,1;0,3 12,1,3;0,3 12,2,5;0,3 13,0,1;0,3 13,1,3;0,3 13,2,5;0,3 14,0,1;0,3 14,1,3;0,3 14,2,5;0,3 15,0,1;0,3 15,1,3;0,3 15,2,5;0,3 16,0,1;0,3 16,1,3;0,3 16,2,5;0,3 17,0,1;0,3 17,1,3;0,3 17,2,5;0,3 18,0,1;0,3 18,1,3;0,3 18,2,5;0,3 19,0,1;0,3 19,1,3;0,3 19,2,5;0,3 20,0,1;0,3 20,1,3;0,3 20,2,5;0,3 21,0,1;0,3 21,1,3;0,3 21,2,5;0,3 22,0,1;0,3 22,1,3;0,3 22,2,5;0,3 23,0,1;0,3 23,1,3;0,3 23,2,5;0,3 24,0,1;0,3 24,1,3;0,3 24,2,5;0,3 25,0,1;0,3 25,1,3;0,3 25,2,5;0,3 26,0,1;0,3 26,1,3;0,3 26,2,5;0,3 27,0,1;0,3 27,1,3;0,3 27,2,5;0,3 28,0,1;0,3 28,1,3;0,3 28,2,5;0,3 29,0,1;0,3 29,1,3;0,3 29,2,5;0,3 30,0,1;0,3 30,1,3;0,3 30,2,5;0,3 31,0,1;0,3 31,1,3;0,3 31,2,5;0,3 32,0,1;0,3 32,1,3;0,3 32,2,5;0,3 33,0,1;0,3 33,1,3;0,3 33,2,5;0,3 34,0,1;0,3 34,1,3;0,3 34,2,5;0,3 35,0,1;0,3 35,1,3;0,3 35,2,5;0,3 36,0,1;0,3 36,1,3;0,3 36,2,5;0,3 37,0,1;0,3 37,1,3;0,3 37,2,5;0,3 38,0,1;0,3 38,1,3;0,3 38,2,5;0,3 39,0,1;0,3 39,1,3;0,3 39,2,5;0,3 40,0,1;0,3 40,1,3;0,3 40,2,5;0,4 5,0,1;0,4 5,1,3;0,4 5,2,5;0,4 6,0,1;0,4 6,1,3;0,4 6,2,5;0,4 7,0,1;0,4 7,1,3;0,4 7,2,5;0,4 8,0,1;0,4 8,1,3;0,4 8,2,5;0,4 9,0,1;0,4 9,1,3;0,4 9,2,5;0,4 10,0,1;0,4 10,1,3;0,4 10,2,5;0,4 11,0,1;0,4 11,1,3;0,4 11,2,5;0,4 12,0,1;0,4 12,1,3;0,4 12,2,5;0,4 13,0,1;0,4 13,1,3;0,4 13,2,5;0,4 14,0,1;0,4 14,1,3;0,4 14,2,5;0,4 15,0,1;0,4 15,1,3;0,4 15,2,5;0,4 16,0,1;0,4 16,1,3;0,4 16,2,5;0,4 17,0,1;0,4 17,1,3;0,4 17,2,5;0,4 18,0,1;0,4 18,1,3;0,4 18,2,5;0,4 19,0,1;0,4 19,1,3;0,4 19,2,5;0,4 20,0,1;0,4 20,1,3;0,4 20,2,5;0,4 21,0,1;0,4 21,1,3;0,4 21,2,5;0,4 22,0,1;0,4 22,1,3;0,4 22,2,5;0,4 23,0,1;0,4 23,1,3;0,4 23,2,5;0,4 24,0,1;0,4 24,1,3;0,4 24,2,5;0,4 25,0,1;0,4 25,1,3;0,4 25,2,5;0,4 26,0,1;0,4 26,1,3;0,4 26,2,5;0,4 27,0,1;0,4 27,1,3;0,4 27,2,5;0,4 28,0,1;0,4 28,1,3;0,4 28,2,5;0,4 29,0,1;0,4 29,1,3;0,4 29,2,5;0,4 30,0,1;0,4 30,1,3;0,4 30,2,5;0,4 31,0,1;0,4 31,1,3;0,4 31,2,5;0,4 32,0,1;0,4 32,1,3;0,4 32,2,5;0,4 33,0,1;0,4 33,1,3;0,4 33,2,5;0,4 34,0,1;0,4 34,1,3;0,4 34,2,5;0,4 35,0,1;0,4 35,1,3;0,4 35,2,5;0,4 36,0,1;0,4 36,1,3;0,4 36,2,5;0,4 37,0,1;0,4 37,1,3;0,4 37,2,5;0,4 38,0,1;0,4 38,1,3;0,4 38,2,5;0,4 39,0,1;0,4 39,1,3;0,4 39,2,5;0,4 40,0,1;0,4 40,1,3;0,4 40,2,5;0,5 6,0,1;0,5 6,1,3;0,5 6,2,5;0,5 7,0,1;0,5 7,1,3;0,5 7,2,5;0,5 8,0,1;0,5 8,1,3;0,5 8,2,5;0,5 9,0,1;0,5 9,1,3;0,5 9,2,5;0,5 10,0,1;0,5 10,1,3;0,5 10,2,5;0,5 11,0,1;0,5 11,1,3;0,5 11,2,5;0,5 12,0,1;0,5 12,1,3;0,5 12,2,5;0,5 13,0,1;0,5 13,1,3;0,5 13,2,5;0,5 14,0,1;0,5 14,1,3;0,5 14,2,5;0,5 15,0,1;0,5 15,1,3;0,5 15,2,5;0,5 16,0,1;0,5 16,1,3;0,5 16,2,5;0,5 17,0,1;0,5 17,1,3;0,5 17,2,5;0,5 18,0,1;0,5 18,1,3;0,5 18,2,5;0,5 19,0,1;0,5 19,1,3;0,5 19,2,5;0,5 20,0,1;0,5 20,1,3;0,5 20,2,5;0,5 21,0,1;0,5 21,1,3;0,5 21,2,5;0,5 22,0,1;0,5 22,1,3;0,5 22,2,5;0,5 23,0,1;0,5 23,1,3;0,5 23,2,5;0,5 24,0,1;0,5 24,1,3;0,5 24,2,5;0,5 25,0,1;0,5 25,1,3;0,5 25,2,5;0,5 26,0,1;0,5 26,1,3;0,5 26,2,5;0,5 27,0,1;0,5 27,1,3;0,5 27,2,5;0,5 28,0,1;0,5 28,1,3;0,5 28,2,5;0,5 29,0,1;0,5 29,1,3;0,5 29,2,5;0,5 30,0,1;0,5 30,1,3;0,5 30,2,5;0,5 31,0,1;0,5 31,1,3;0,5 31,2,5;0,5 32,0,1;0,5 32,1,3;0,5 32,2,5;0,5 33,0,1;0,5 33,1,3;0,5 33,2,5;0,5 34,0,1;0,5 34,1,3;0,5 34,2,5;0,5 35,0,1;0,5 35,1,3;0,5 35,2,5;0,5 36,0,1;0,5 36,1,3;0,5 36,2,5;0,5 37,0,1;0,5 37,1,3;0,5 37,2,5;0,5 38,0,1;0,5 38,1,3;0,5 38,2,5;0,5 39,0,1;0,5 39,1,3;0,5 39,2,5;0,5 40,0,1;0,5 40,1,3;0,5 40,2,5;0,6 7,0,1;0,6 7,1,3;0,6 7,2,5;0,6 8,0,1;0,6 8,1,3;0,6 8,2,5;0,6 9,0,1;0,6 9,1,3;0,6 9,2,5;0,6 10,0,1;0,6 10,1,3;0,6 10,2,5;0,6 11,0,1;0,6 11,1,3;0,6 11,2,5;0,6 12,0,1;0,6 12,1,3;0,6 12,2,5;0,6 13,0,1;0,6 13,1,3;0,6 13,2,5;0,6 14,0,1;0,6 14,1,3;0,6 14,2,5;0,6 15,0,1;0,6 15,1,3;0,6 15,2,5;0,6 16,0,1;0,6 16,1,3;0,6 16,2,5;0,6 17,0,1;0,6 17,1,3;0,6 17,2,5;0,6 18,0,1;0,6 18,1,3;0,6 18,2,5;0,6 19,0,1;0,6 19,1,3;0,6 19,2,5;0,6 20,0,1;0,6 20,1,3;0,6 20,2,5;0,6 21,0,1;0,6 21,1,3;0,6 21,2,5;0,6 22,0,1;0,6 22,1,3;0,6 22,2,5;0,6 23,0,1;0,6 23,1,3;0,6 23,2,5;0,6 24,0,1;0,6 24,1,3;0,6 24,2,5;0,6 25,0,1;0,6 25,1,3;0,6 25,2,5;0,6 26,0,1;0,6 26,1,3;0,6 26,2,5;0,6 27,0,1;0,6 27,1,3;0,6 27,2,5;0,6 28,0,1;0,6 28,1,3;0,6 28,2,5;0,6 29,0,1;0,6 29,1,3;0,6 29,2,5;0,6 30,0,1;0,6 30,1,3;0,6 30,2,5;0,6 31,0,1;0,6 31,1,3;0,6 31,2,5;0,6 32,0,1;0,6 32,1,3;0,6 32,2,5;0,6 33,0,1;0,6 33,1,3;0,6 33,2,5;0,6 34,0,1;0,6 34,1,3;0,6 34,2,5;0,6 35,0,1;0,6 35,1,3;0,6 35,2,5;0,6 36,0,1;0,6 36,1,3;0,6 36,2,5;0,6 37,0,1;0,6 37,1,3;0,6 37,2,5;0,6 38,0,1;0,6 38,1,3;0,6 38,2,5;0,6 39,0,1;0,6 39,1,3;0,6 39,2,5;0,6 40,0,1;0,6 40,1,3;0,6 40,2,5;0,7 8,0,1;0,7 8,1,3;0,7 8,2,5;0,7 9,0,1;0,7 9,1,3;0,7 9,2,5;0,7 10,0,1;0,7 10,1,3;0,7 10,2,5;0,7 11,0,1;0,7 11,1,3;0,7 11,2,5;0,7 12,0,1;0,7 12,1,3;0,7 12,2,5;0,7 13,0,1;0,7 13,1,3;0,7 13,2,5;0,7 14,0,1;0,7 14,1,3;0,7 14,2,5;0,7 15,0,1;0,7 15,1,3;0,7 15,2,5;0,7 16,0,1;0,7 16,1,3;0,7 16,2,5;0,7 17,0,1;0,7 17,1,3;0,7 17,2,5;0,7 18,0,1;0,7 18,1,3;0,7 18,2,5;0,7 19,0,1;0,7 19,1,3;0,7 19,2,5;0,7 20,0,1;0,7 20,1,3;0,7 20,2,5;0,7 21,0,1;0,7 21,1,3;0,7 21,2,5;0,7 22,0,1;0,7 22,1,3;0,7 22,2,5;0,7 23,0,1;0,7 23,1,3;0,7 23,2,5;0,7 24,0,1;0,7 24,1,3;0,7 24,2,5;0,7 25,0,1;0,7 25,1,3;0,7 25,2,5;0,7 26,0,1;0,7 26,1,3;0,7 26,2,5;0,7 27,0,1;0,7 27,1,3;0,7 27,2,5;0,7 28,0,1;0,7 28,1,3;0,7 28,2,5;0,7 29,0,1;0,7 29,1,3;0,7 29,2,5;0,7 30,0,1;0,7 30,1,3;0,7 30,2,5;0,7 31,0,1;0,7 31,1,3;0,7 31,2,5;0,7 32,0,1;0,7 32,1,3;0,7 32,2,5;0,7 33,0,1;0,7 33,1,3;0,7 33,2,5;0,7 34,0,1;0,7 34,1,3;0,7 34,2,5;0,7 35,0,1;0,7 35,1,3;0,7 35,2,5;0,7 36,0,1;0,7 36,1,3;0,7 36,2,5;0,7 37,0,1;0,7 37,1,3;0,7 37,2,5;0,7 38,0,1;0,7 38,1,3;0,7 38,2,5;0,7 39,0,1;0,7 39,1,3;0,7 39,2,5;0,7 40,0,1;0,7 40,1,3;0,7 40,2,5;0,8 9,0,1;0,8 9,1,3;0,8 9,2,5;0,8 10,0,1;0,8 10,1,3;0,8 10,2,5;0,8 11,0,1;0,8 11,1,3;0,8 11,2,5;0,8 12,0,1;0,8 12,1,3;0,8 12,2,5;0,8 13,0,1;0,8 13,1,3;0,8 13,2,5;0,8 14,0,1;0,8 14,1,3;0,8 14,2,5;0,8 15,0,1;0,8 15,1,3;0,8 15,2,5;0,8 16,0,1;0,8 16,1,3;0,8 16,2,5;0,8 17,0,1;0,8 17,1,3;0,8 17,2,5;0,8 18,0,1;0,8 18,1,3;0,8 18,2,5;0,8 19,0,1;0,8 19,1,3;0,8 19,2,5;0,8 20,0,1;0,8 20,1,3;0,8 20,2,5;0,8 21,0,1;0,8 21,1,3;0,8 21,2,5;0,8 22,0,1;0,8 22,1,3;0,8 22,2,5;0,8 23,0,1;0,8 23,1,3;0,8 23,2,5;0,8 24,0,1;0,8 24,1,3;0,8 24,2,5;0,8 25,0,1;0,8 25,1,3;0,8 25,2,5;0,8 26,0,1;0,8 26,1,3;0,8 26,2,5;0,8 27,0,1;0,8 27,1,3;0,8 27,2,5;0,8 28,0,1;0,8 28,1,3;0,8 28,2,5;0,8 29,0,1;0,8 29,1,3;0,8 29,2,5;0,8 30,0,1;0,8 30,1,3;0,8 30,2,5;0,8 31,0,1;0,8 31,1,3;0,8 31,2,5;0,8 32,0,1;0,8 32,1,3;0,8 32,2,5;0,8 33,0,1;0,8 33,1,3;0,8 33,2,5;0,8 34,0,1;0,8 34,1,3;0,8 34,2,5;0,8 35,0,1;0,8 35,1,3;0,8 35,2,5;0,8 36,0,1;0,8 36,1,3;0,8 36,2,5"
        #"
        # wp = "0,0,0,1;0,0,1,1;0,0,2,1;0,1,0,1;0,1,1,1;0,1,2,1;0,2,0,1;0,2,1,1;0,2,2,1;0,3,0,1;0,3,1,1;0,3,2,1;0,4,0,1;0,4,1,1;0,4,2,1;0,5,0,1;0,5,1,1;0,5,2,1;0,6,0,1;0,6,1,1;0,6,2,1;0,7,0,1;0,7,1,1;0,7,2,1;0,8,0,1;0,8,1,1;0,8,2,1;0,9,0,1;0,9,1,1;0,9,2,1;0,10,0,1;0,10,1,1;0,10,2,1;0,11,0,1;0,11,1,1;0,11,2,1;0,12,0,1;0,12,1,1;0,12,2,1;0,13,0,1;0,13,1,1;0,13,2,1;0,14,0,1;0,14,1,1;0,14,2,1;0,15,0,1;0,15,1,1;0,15,2,1;0,16,0,1;0,16,1,1;0,16,2,1;0,17,0,1;0,17,1,1;0,17,2,1;0,18,0,1;0,18,1,1;0,18,2,1;0,19,0,1;0,19,1,1;0,19,2,1;0,20,0,1;0,20,1,1;0,20,2,1;0,21,0,1;0,21,1,1;0,21,2,1;0,22,0,1;0,22,1,1;0,22,2,1;0,23,0,1;0,23,1,1;0,23,2,1;0,24,0,1;0,24,1,1;0,24,2,1;0,25,0,1;0,25,1,1;0,25,2,1;0,26,0,1;0,26,1,1;0,26,2,1;0,27,0,1;0,27,1,1;0,27,2,1;0,28,0,1;0,28,1,1;0,28,2,1;0,29,0,1;0,29,1,1;0,29,2,1;0,30,0,1;0,30,1,1;0,30,2,1;0,31,0,1;0,31,1,1;0,31,2,1;0,32,0,1;0,32,1,1;0,32,2,1;0,33,0,1;0,33,1,1;0,33,2,1;0,34,0,1;0,34,1,1;0,34,2,1;0,35,0,1;0,35,1,1;0,35,2,1;0,36,0,1;0,36,1,1;0,36,2,1;0,37,0,1;0,37,1,1;0,37,2,1;0,38,0,1;0,38,1,1;0,38,2,1;0,39,0,1;0,39,1,1;0,39,2,1;0,40,0,1;0,40,1,1;0,40,2,1;0,0 1,0,1;0,8 30,1,3;0,8 30,2,5;0,8 31,0,1;0,8 31,1,3;0,8 31,2,5;0,8 32,0,1;0,8 32,1,3;0,8 32,2,5;0,8 33,0,1;0,8 33,1,3;0,8 33,2,5;0,8 34,0,1;0,8 34,1,3;0,8 34,2,5;0,8 35,0,1;0,8 35,1,3;0,8 35,2,5;0,8 36,0,1;0,8 36,1,3;0,8 36,2,5"

        wp += ";" + "0,8 37,0,1;0,8 37,1,3;0,8 37,2,5;0,8 38,0,1;0,8 38,1,3;0,8 38,2,5;0,8 39,0,1;0,8 39,1,3;0,8 39,2,5;0,8 40,0,1;0,8 40,1,3;0,8 40,2,5;0,9 10,0,1;0,9 10,1,3;0,9 10,2,5;0,9 11,0,1;0,9 11,1,3;0,9 11,2,5;0,9 12,0,1;0,9 12,1,3;0,9 12,2,5;0,9 13,0,1;0,9 13,1,3;0,9 13,2,5;0,9 14,0,1;0,9 14,1,3;0,9 14,2,5;0,9 15,0,1;0,9 15,1,3;0,9 15,2,5;0,9 16,0,1;0,9 16,1,3;0,9 16,2,5;0,9 17,0,1;0,9 17,1,3;0,9 17,2,5;0,9 18,0,1;0,9 18,1,3;0,9 18,2,5;0,9 19,0,1;0,9 19,1,3;0,9 19,2,5;0,9 20,0,1;0,9 20,1,3;0,9 20,2,5;0,9 21,0,1;0,9 21,1,3;0,9 21,2,5;0,9 22,0,1;0,9 22,1,3;0,9 22,2,5;0,9 23,0,1;0,9 23,1,3;0,9 23,2,5;0,9 24,0,1;0,9 24,1,3;0,9 24,2,5;0,9 25,0,1;0,9 25,1,3;0,9 25,2,5;0,9 26,0,1;0,9 26,1,3;0,9 26,2,5;0,9 27,0,1;0,9 27,1,3;0,9 27,2,5;0,9 28,0,1;0,9 28,1,3;0,9 28,2,5;0,9 29,0,1;0,9 29,1,3;0,9 29,2,5;0,9 30,0,1;0,9 30,1,3;0,9 30,2,5;0,9 31,0,1;0,9 31,1,3;0,9 31,2,5;0,9 32,0,1;0,9 32,1,3;0,9 32,2,5;0,9 33,0,1;0,9 33,1,3;0,9 33,2,5;0,9 34,0,1;0,9 34,1,3;0,9 34,2,5;0,9 35,0,1;0,9 35,1,3;0,9 35,2,5;0,9 36,0,1;0,9 36,1,3;0,9 36,2,5;0,9 37,0,1;0,9 37,1,3;0,9 37,2,5;0,9 38,0,1;0,9 38,1,3;0,9 38,2,5;0,9 39,0,1;0,9 39,1,3;0,9 39,2,5;0,9 40,0,1;0,9 40,1,3;0,9 40,2,5;0,10 11,0,1;0,10 11,1,3;0,10 11,2,5;0,10 12,0,1;0,10 12,1,3;0,10 12,2,5;0,10 13,0,1;0,10 13,1,3;0,10 13,2,5;0,10 14,0,1;0,10 14,1,3;0,10 14,2,5;0,10 15,0,1;0,10 15,1,3;0,10 15,2,5;0,10 16,0,1;0,10 16,1,3;0,10 16,2,5;0,10 17,0,1;0,10 17,1,3;0,10 17,2,5;0,10 18,0,1;0,10 18,1,3;0,10 18,2,5;0,10 19,0,1;0,10 19,1,3;0,10 19,2,5;0,10 20,0,1;0,10 20,1,3;0,10 20,2,5;0,10 21,0,1;0,10 21,1,3;0,10 21,2,5;0,10 22,0,1;0,10 22,1,3;0,10 22,2,5;0,10 23,0,1;0,10 23,1,3;0,10 23,2,5;0,10 24,0,1;0,10 24,1,3;0,10 24,2,5;0,10 25,0,1;0,10 25,1,3;0,10 25,2,5;0,10 26,0,1;0,10 26,1,3;0,10 26,2,5;0,10 27,0,1;0,10 27,1,3;0,10 27,2,5;0,10 28,0,1;0,10 28,1,3;0,10 28,2,5;0,10 29,0,1;0,10 29,1,3;0,10 29,2,5;0,10 30,0,1;0,10 30,1,3;0,10 30,2,5;0,10 31,0,1;0,10 31,1,3;0,10 31,2,5;0,10 32,0,1;0,10 32,1,3;0,10 32,2,5;0,10 33,0,1;0,10 33,1,3;0,10 33,2,5;0,10 34,0,1;0,10 34,1,3;0,10 34,2,5;0,10 35,0,1;0,10 35,1,3;0,10 35,2,5;0,10 36,0,1;0,10 36,1,3;0,10 36,2,5;0,10 37,0,1;0,10 37,1,3;0,10 37,2,5;0,10 38,0,1;0,10 38,1,3;0,10 38,2,5;0,10 39,0,1;0,10 39,1,3;0,10 39,2,5;0,10 40,0,1;0,10 40,1,3;0,10 40,2,5;0,11 12,0,1;0,11 12,1,3;0,11 12,2,5;0,11 13,0,1;0,11 13,1,3;0,11 13,2,5;0,11 14,0,1;0,11 14,1,3;0,11 14,2,5;0,11 15,0,1;0,11 15,1,3;0,11 15,2,5;0,11 16,0,1;0,11 16,1,3;0,11 16,2,5;0,11 17,0,1;0,11 17,1,3;0,11 17,2,5;0,11 18,0,1;0,11 18,1,3;0,11 18,2,5;0,11 19,0,1;0,11 19,1,3;0,11 19,2,5;0,11 20,0,1;0,11 20,1,3;0,11 20,2,5;0,11 21,0,1;0,11 21,1,3;0,11 21,2,5;0,11 22,0,1;0,11 22,1,3;0,11 22,2,5;0,11 23,0,1;0,11 23,1,3;0,11 23,2,5;0,11 24,0,1;0,11 24,1,3;0,11 24,2,5;0,11 25,0,1;0,11 25,1,3;0,11 25,2,5;0,11 26,0,1;0,11 26,1,3;0,11 26,2,5;0,11 27,0,1;0,11 27,1,3;0,11 27,2,5;0,11 28,0,1;0,11 28,1,3;0,11 28,2,5;0,11 29,0,1;0,11 29,1,3;0,11 29,2,5;0,11 30,0,1;0,11 30,1,3;0,11 30,2,5;0,11 31,0,1;0,11 31,1,3;0,11 31,2,5;0,11 32,0,1;0,11 32,1,3;0,11 32,2,5;0,11 33,0,1;0,11 33,1,3;0,11 33,2,5;0,11 34,0,1;0,11 34,1,3;0,11 34,2,5;0,11 35,0,1;0,11 35,1,3;0,11 35,2,5;0,11 36,0,1;0,11 36,1,3;0,11 36,2,5;0,11 37,0,1;0,11 37,1,3;0,11 37,2,5;0,11 38,0,1;0,11 38,1,3;0,11 38,2,5;0,11 39,0,1;0,11 39,1,3;0,11 39,2,5;0,11 40,0,1;0,11 40,1,3;0,11 40,2,5;0,12 13,0,1;0,12 13,1,3;0,12 13,2,5;0,12 14,0,1;0,12 14,1,3;0,12 14,2,5;0,12 15,0,1;0,12 15,1,3;0,12 15,2,5;0,12 16,0,1;0,12 16,1,3;0,12 16,2,5;0,12 17,0,1;0,12 17,1,3;0,12 17,2,5;0,12 18,0,1;0,12 18,1,3;0,12 18,2,5;0,12 19,0,1;0,12 19,1,3;0,12 19,2,5;0,12 20,0,1;0,12 20,1,3;0,12 20,2,5;0,12 21,0,1;0,12 21,1,3;0,12 21,2,5;0,12 22,0,1;0,12 22,1,3;0,12 22,2,5;0,12 23,0,1;0,12 23,1,3;0,12 23,2,5;0,12 24,0,1;0,12 24,1,3;0,12 24,2,5;0,12 25,0,1;0,12 25,1,3;0,12 25,2,5;0,12 26,0,1;0,12 26,1,3;0,12 26,2,5;0,12 27,0,1;0,12 27,1,3;0,12 27,2,5;0,12 28,0,1;0,12 28,1,3;0,12 28,2,5;0,12 29,0,1;0,12 29,1,3;0,12 29,2,5;0,12 30,0,1;0,12 30,1,3;0,12 30,2,5;0,12 31,0,1;0,12 31,1,3;0,12 31,2,5;0,12 32,0,1;0,12 32,1,3;0,12 32,2,5;0,12 33,0,1;0,12 33,1,3;0,12 33,2,5;0,12 34,0,1;0,12 34,1,3;0,12 34,2,5;0,12 35,0,1;0,12 35,1,3;0,12 35,2,5;0,12 36,0,1;0,12 36,1,3;0,12 36,2,5;0,12 37,0,1;0,12 37,1,3;0,12 37,2,5;0,12 38,0,1;0,12 38,1,3;0,12 38,2,5;0,12 39,0,1;0,12 39,1,3;0,12 39,2,5;0,12 40,0,1;0,12 40,1,3;0,12 40,2,5;0,13 14,0,1;0,13 14,1,3;0,13 14,2,5;0,13 15,0,1;0,13 15,1,3;0,13 15,2,5;0,13 16,0,1;0,13 16,1,3;0,13 16,2,5;0,13 17,0,1;0,13 17,1,3;0,13 17,2,5;0,13 18,0,1;0,13 18,1,3;0,13 18,2,5;0,13 19,0,1;0,13 19,1,3;0,13 19,2,5;0,13 20,0,1;0,13 20,1,3;0,13 20,2,5;0,13 21,0,1;0,13 21,1,3;0,13 21,2,5;0,13 22,0,1;0,13 22,1,3;0,13 22,2,5;0,13 23,0,1;0,13 23,1,3;0,13 23,2,5;0,13 24,0,1;0,13 24,1,3;0,13 24,2,5;0,13 25,0,1;0,13 25,1,3;0,13 25,2,5;0,13 26,0,1;0,13 26,1,3;0,13 26,2,5;0,13 27,0,1;0,13 27,1,3;0,13 27,2,5;0,13 28,0,1;0,13 28,1,3;0,13 28,2,5;0,13 29,0,1;0,13 29,1,3;0,13 29,2,5;0,13 30,0,1;0,13 30,1,3;0,13 30,2,5;0,13 31,0,1;0,13 31,1,3;0,13 31,2,5;0,13 32,0,1;0,13 32,1,3;0,13 32,2,5;0,13 33,0,1;0,13 33,1,3;0,13 33,2,5;0,13 34,0,1;0,13 34,1,3;0,13 34,2,5;0,13 35,0,1;0,13 35,1,3;0,13 35,2,5;0,13 36,0,1;0,13 36,1,3;0,13 36,2,5;0,13 37,0,1;0,13 37,1,3;0,13 37,2,5;0,13 38,0,1;0,13 38,1,3;0,13 38,2,5;0,13 39,0,1;0,13 39,1,3;0,13 39,2,5;0,13 40,0,1;0,13 40,1,3;0,13 40,2,5;0,14 15,0,1;0,14 15,1,3;0,14 15,2,5;0,14 16,0,1;0,14 16,1,3;0,14 16,2,5;0,14 17,0,1;0,14 17,1,3;0,14 17,2,5;0,14 18,0,1;0,14 18,1,3;0,14 18,2,5;0,14 19,0,1;0,14 19,1,3;0,14 19,2,5;0,14 20,0,1;0,14 20,1,3;0,14 20,2,5;0,14 21,0,1;0,14 21,1,3;0,14 21,2,5;0,14 22,0,1;0,14 22,1,3;0,14 22,2,5;0,14 23,0,1;0,14 23,1,3;0,14 23,2,5;0,14 24,0,1;0,14 24,1,3;0,14 24,2,5;0,14 25,0,1;0,14 25,1,3;0,14 25,2,5;0,14 26,0,1;0,14 26,1,3;0,14 26,2,5;0,14 27,0,1;0,14 27,1,3;0,14 27,2,5;0,14 28,0,1;0,14 28,1,3;0,14 28,2,5;0,14 29,0,1;0,14 29,1,3;0,14 29,2,5;0,14 30,0,1;0,14 30,1,3;0,14 30,2,5;0,14 31,0,1;0,14 31,1,3;0,14 31,2,5;0,14 32,0,1;0,14 32,1,3;0,14 32,2,5;0,14 33,0,1;0,14 33,1,3;0,14 33,2,5;0,14 34,0,1;0,14 34,1,3;0,14 34,2,5;0,14 35,0,1;0,14 35,1,3;0,14 35,2,5;0,14 36,0,1;0,14 36,1,3;0,14 36,2,5;0,14 37,0,1;0,14 37,1,3;0,14 37,2,5;0,14 38,0,1;0,14 38,1,3;0,14 38,2,5;0,14 39,0,1;0,14 39,1,3;0,14 39,2,5;0,14 40,0,1;0,14 40,1,3;0,14 40,2,5;0,15 16,0,1;0,15 16,1,3;0,15 16,2,5;0,15 17,0,1;0,15 17,1,3;0,15 17,2,5;0,15 18,0,1;0,15 18,1,3;0,15 18,2,5;0,15 19,0,1;0,15 19,1,3;0,15 19,2,5;0,15 20,0,1;0,15 20,1,3;0,15 20,2,5;0,15 21,0,1;0,15 21,1,3;0,15 21,2,5;0,15 22,0,1;0,15 22,1,3;0,15 22,2,5;0,15 23,0,1;0,15 23,1,3;0,15 23,2,5;0,15 24,0,1;0,15 24,1,3;0,15 24,2,5;0,15 25,0,1;0,15 25,1,3;0,15 25,2,5;0,15 26,0,1;0,15 26,1,3;0,15 26,2,5;0,15 27,0,1;0,15 27,1,3;0,15 27,2,5;0,15 28,0,1;0,15 28,1,3;0,15 28,2,5;0,15 29,0,1;0,15 29,1,3;0,15 29,2,5;0,15 30,0,1;0,15 30,1,3;0,15 30,2,5;0,15 31,0,1;0,15 31,1,3;0,15 31,2,5;0,15 32,0,1;0,15 32,1,3;0,15 32,2,5;0,15 33,0,1;0,15 33,1,3;0,15 33,2,5;0,15 34,0,1;0,15 34,1,3;0,15 34,2,5;0,15 35,0,1;0,15 35,1,3;0,15 35,2,5;0,15 36,0,1;0,15 36,1,3;0,15 36,2,5;0,15 37,0,1;0,15 37,1,3;0,15 37,2,5;0,15 38,0,1;0,15 38,1,3;0,15 38,2,5;0,15 39,0,1;0,15 39,1,3;0,15 39,2,5;0,15 40,0,1;0,15 40,1,3;0,15 40,2,5;0,16 17,0,1;0,16 17,1,3;0,16 17,2,5;0,16 18,0,1;0,16 18,1,3;0,16 18,2,5;0,16 19,0,1;0,16 19,1,3;0,16 19,2,5;0,16 20,0,1;0,16 20,1,3;0,16 20,2,5;0,16 21,0,1;0,16 21,1,3;0,16 21,2,5;0,16 22,0,1;0,16 22,1,3;0,16 22,2,5;0,16 23,0,1;0,16 23,1,3;0,16 23,2,5;0,16 24,0,1;0,16 24,1,3;0,16 24,2,5;0,16 25,0,1;0,16 25,1,3;0,16 25,2,5;0,16 26,0,1;0,16 26,1,3;0,16 26,2,5;0,16 27,0,1;0,16 27,1,3;0,16 27,2,5;0,16 28,0,1;0,16 28,1,3;0,16 28,2,5;0,16 29,0,1;0,16 29,1,3;0,16 29,2,5;0,16 30,0,1;0,16 30,1,3;0,16 30,2,5;0,16 31,0,1;0,16 31,1,3;0,16 31,2,5;0,16 32,0,1;0,16 32,1,3;0,16 32,2,5;0,16 33,0,1;0,16 33,1,3;0,16 33,2,5;0,16 34,0,1;0,16 34,1,3;0,16 34,2,5;0,16 35,0,1;0,16 35,1,3;0,16 35,2,5;0,16 36,0,1;0,16 36,1,3;0,16 36,2,5;0,16 37,0,1;0,16 37,1,3;0,16 37,2,5;0,16 38,0,1;0,16 38,1,3;0,16 38,2,5;0,16 39,0,1;0,16 39,1,3;0,16 39,2,5;0,16 40,0,1;0,16 40,1,3;0,16 40,2,5;0,17 18,0,1;0,17 18,1,3;0,17 18,2,5;0,17 19,0,1;0,17 19,1,3;0,17 19,2,5;0,17 20,0,1;0,17 20,1,3;0,17 20,2,5;0,17 21,0,1;0,17 21,1,3;0,17 21,2,5;0,17 22,0,1;0,17 22,1,3;0,17 22,2,5;0,17 23,0,1;0,17 23,1,3;0,17 23,2,5;0,17 24,0,1;0,17 24,1,3;0,17 24,2,5;0,17 25,0,1;0,17 25,1,3;0,17 25,2,5;0,17 26,0,1;0,17 26,1,3;0,17 26,2,5;0,17 27,0,1;0,17 27,1,3;0,17 27,2,5;0,17 28,0,1;0,17 28,1,3;0,17 28,2,5;0,17 29,0,1;0,17 29,1,3;0,17 29,2,5;0,17 30,0,1;0,17 30,1,3;0,17 30,2,5;0,17 31,0,1;0,17 31,1,3;0,17 31,2,5;0,17 32,0,1;0,17 32,1,3;0,17 32,2,5;0,17 33,0,1;0,17 33,1,3;0,17 33,2,5;0,17 34,0,1;0,17 34,1,3;0,17 34,2,5;0,17 35,0,1;0,17 35,1,3;0,17 35,2,5;0,17 36,0,1;0,17 36,1,3;0,17 36,2,5;0,17 37,0,1;0,17 37,1,3;0,17 37,2,5;0,17 38,0,1;0,17 38,1,3;0,17 38,2,5;0,17 39,0,1;0,17 39,1,3;0,17 39,2,5;0,17 40,0,1;0,17 40,1,3;0,17 40,2,5;0,18 19,0,1;0,18 19,1,3;0,18 19,2,5;0,18 20,0,1;0,18 20,1,3;0,18 20,2,5;0,18 21,0,1;0,18 21,1,3;0,18 21,2,5;0,18 22,0,1;0,18 22,1,3;0,18 22,2,5;0,18 23,0,1;0,18 23,1,3;0,18 23,2,5;0,18 24,0,1;0,18 24,1,3;0,18 24,2,5;0,18 25,0,1;0,18 25,1,3;0,18 25,2,5;0,18 26,0,1;0,18 26,1,3;0,18 26,2,5;0,18 27,0,1;0,18 27,1,3;0,18 27,2,5;0,18 28,0,1;0,18 28,1,3;0,18 28,2,5;0,18 29,0,1;0,18 29,1,3;0,18 29,2,5;0,18 30,0,1;0,18 30,1,3;0,18 30,2,5;0,18 31,0,1;0,18 31,1,3;0,18 31,2,5;0,18 32,0,1;0,18 32,1,3;0,18 32,2,5;0,18 33,0,1;0,18 33,1,3;0,18 33,2,5;0,18 34,0,1;0,18 34,1,3;0,18 34,2,5;0,18 35,0,1;0,18 35,1,3;0,18 35,2,5;0,18 36,0,1;0,18 36,1,3;0,18 36,2,5;0,18 37,0,1;0,18 37,1,3;0,18 37,2,5;0,18 38,0,1;0,18 38,1,3;0,18 38,2,5;0,18 39,0,1;0,18 39,1,3;0,18 39,2,5;0,18 40,0,1;0,18 40,1,3;0,18 40,2,5;0,19 20,0,1;0,19 20,1,3;0,19 20,2,5;0,19 21,0,1;0,19 21,1,3;0,19 21,2,5;0,19 22,0,1;0,19 22,1,3;0,19 22,2,5;0,19 23,0,1;0,19 23,1,3;0,19 23,2,5;0,19 24,0,1;0,19 24,1,3;0,19 24,2,5;0,19 25,0,1;0,19 25,1,3;0,19 25,2,5;0,19 26,0,1;0,19 26,1,3;0,19 26,2,5;0,19 27,0,1;0,19 27,1,3;0,19 27,2,5;0,19 28,0,1;0,19 28,1,3;0,19 28,2,5;0,19 29,0,1;0,19 29,1,3;0,19 29,2,5;0,19 30,0,1;0,19 30,1,3;0,19 30,2,5;0,19 31,0,1;0,19 31,1,3;0,19 31,2,5;0,19 32,0,1;0,19 32,1,3;0,19 32,2,5;0,19 33,0,1;0,19 33,1,3;0,19 33,2,5;0,19 34,0,1;0,19 34,1,3;0,19 34,2,5;0,19 35,0,1;0,19 35,1,3;0,19 35,2,5;0,19 36,0,1;0,19 36,1,3;0,19 36,2,5;0,19 37,0,1;0,19 37,1,3;0,19 37,2,5;0,19 38,0,1;0,19 38,1,3;0,19 38,2,5;0,19 39,0,1;0,19 39,1,3;0,19 39,2,5;0,19 40,0,1;0,19 40,1,3;0,19 40,2,5;0,20 21,0,1;0,20 21,1,3;0,20 21,2,5;0,20 22,0,1;0,20 22,1,3;0,20 22,2,5;0,20 23,0,1;0,20 23,1,3;0,20 23,2,5;0,20 24,0,1;0,20 24,1,3;0,20 24,2,5;0,20 25,0,1;0,20 25,1,3;0,20 25,2,5;0,20 26,0,1;0,20 26,1,3;0,20 26,2,5;0,20 27,0,1;0,20 27,1,3;0,20 27,2,5;0,20 28,0,1;0,20 28,1,3;0,20 28,2,5;0,20 29,0,1;0,20 29,1,3;0,20 29,2,5;0,20 30,0,1;0,20 30,1,3;0,20 30,2,5;0,20 31,0,1;0,20 31,1,3;0,20 31,2,5;0,20 32,0,1;0,20 32,1,3;0,20 32,2,5;0,20 33,0,1;0,20 33,1,3;0,20 33,2,5;0,20 34,0,1;0,20 34,1,3;0,20 34,2,5;0,20 35,0,1;0,20 35,1,3;0,20 35,2,5;0,20 36,0,1;0,20 36,1,3;0,20 36,2,5;0,20 37,0,1;0,20 37,1,3;0,20 37,2,5;0,20 38,0,1;0,20 38,1,3;0,20 38,2,5;0,20 39,0,1;0,20 39,1,3;0,20 39,2,5;0,20 40,0,1;0,20 40,1,3;0,20 40,2,5;0,21 22,0,1;0,21 22,1,3;0,21 22,2,5;0,21 23,0,1;0,21 23,1,3;0,21 23,2,5;0,21 24,0,1;0,21 24,1,3;0,21 24,2,5;0,21 25,0,1;0,21 25,1,3;0,21 25,2,5;0,21 26,0,1;0,21 26,1,3;0,21 26,2,5;0,21 27,0,1;0,21 27,1,3;0,21 27,2,5;0,21 28,0,1;0,21 28,1,3;0,21 28,2,5;0,21 29,0,1;0,21 29,1,3;0,21 29,2,5;0,21 30,0,1;0,21 30,1,3;0,21 30,2,5;0,21 31,0,1;0,21 31,1,3;0,21 31,2,5;0,21 32,0,1;0,21 32,1,3;0,21 32,2,5;0,21 33,0,1;0,21 33,1,3;0,21 33,2,5;0,21 34,0,1;0,21 34,1,3;0,21 34,2,5;0,21 35,0,1;0,21 35,1,3;0,21 35,2,5;0,21 36,0,1;0,21 36,1,3;0,21 36,2,5;0,21 37,0,1;0,21 37,1,3;0,21 37,2,5;0,21 38,0,1;0,21 38,1,3;0,21 38,2,5;0,21 39,0,1;0,21 39,1,3;0,21 39,2,5;0,21 40,0,1;0,21 40,1,3;0,21 40,2,5;0,22 23,0,1;0,22 23,1,3;0,22 23,2,5;0,22 24,0,1;0,22 24,1,3;0,22 24,2,5;0,22 25,0,1;0,22 25,1,3;0,22 25,2,5;0,22 26,0,1;0,22 26,1,3;0,22 26,2,5;0,22 27,0,1;0,22 27,1,3"
        wp += ";" + "0,22 27,2,5;0,22 28,0,1;0,22 28,1,3;0,22 28,2,5;0,22 29,0,1;0,22 29,1,3;0,22 29,2,5;0,22 30,0,1;0,22 30,1,3;0,22 30,2,5;0,22 31,0,1;0,22 31,1,3;0,22 31,2,5;0,22 32,0,1;0,22 32,1,3;0,22 32,2,5;0,22 33,0,1;0,22 33,1,3;0,22 33,2,5;0,22 34,0,1;0,22 34,1,3;0,22 34,2,5;0,22 35,0,1;0,22 35,1,3;0,22 35,2,5;0,22 36,0,1;0,22 36,1,3;0,22 36,2,5;0,22 37,0,1;0,22 37,1,3;0,22 37,2,5;0,22 38,0,1;0,22 38,1,3;0,22 38,2,5;0,22 39,0,1;0,22 39,1,3;0,22 39,2,5;0,22 40,0,1;0,22 40,1,3;0,22 40,2,5;0,23 24,0,1;0,23 24,1,3;0,23 24,2,5;0,23 25,0,1;0,23 25,1,3;0,23 25,2,5;0,23 26,0,1;0,23 26,1,3;0,23 26,2,5;0,23 27,0,1;0,23 27,1,3;0,23 27,2,5;0,23 28,0,1;0,23 28,1,3;0,23 28,2,5;0,23 29,0,1;0,23 29,1,3;0,23 29,2,5;0,23 30,0,1;0,23 30,1,3;0,23 30,2,5;0,23 31,0,1;0,23 31,1,3;0,23 31,2,5;0,23 32,0,1;0,23 32,1,3;0,23 32,2,5;0,23 33,0,1;0,23 33,1,3;0,23 33,2,5;0,23 34,0,1;0,23 34,1,3;0,23 34,2,5;0,23 35,0,1;0,23 35,1,3;0,23 35,2,5;0,23 36,0,1;0,23 36,1,3;0,23 36,2,5;0,23 37,0,1;0,23 37,1,3;0,23 37,2,5;0,23 38,0,1;0,23 38,1,3;0,23 38,2,5;0,23 39,0,1;0,23 39,1,3;0,23 39,2,5;0,23 40,0,1;0,23 40,1,3;0,23 40,2,5;0,24 25,0,1;0,24 25,1,3;0,24 25,2,5;0,24 26,0,1;0,24 26,1,3;0,24 26,2,5;0,24 27,0,1;0,24 27,1,3;0,24 27,2,5;0,24 28,0,1;0,24 28,1,3;0,24 28,2,5;0,24 29,0,1;0,24 29,1,3;0,24 29,2,5;0,24 30,0,1;0,24 30,1,3;0,24 30,2,5;0,24 31,0,1;0,24 31,1,3;0,24 31,2,5;0,24 32,0,1;0,24 32,1,3;0,24 32,2,5;0,24 33,0,1;0,24 33,1,3;0,24 33,2,5;0,24 34,0,1;0,24 34,1,3;0,24 34,2,5;0,24 35,0,1;0,24 35,1,3;0,24 35,2,5;0,24 36,0,1;0,24 36,1,3;0,24 36,2,5;0,24 37,0,1;0,24 37,1,3;0,24 37,2,5;0,24 38,0,1;0,24 38,1,3;0,24 38,2,5;0,24 39,0,1;0,24 39,1,3;0,24 39,2,5;0,24 40,0,1;0,24 40,1,3;0,24 40,2,5;0,25 26,0,1;0,25 26,1,3;0,25 26,2,5;0,25 27,0,1;0,25 27,1,3;0,25 27,2,5;0,25 28,0,1;0,25 28,1,3;0,25 28,2,5;0,25 29,0,1;0,25 29,1,3;0,25 29,2,5;0,25 30,0,1;0,25 30,1,3;0,25 30,2,5;0,25 31,0,1;0,25 31,1,3;0,25 31,2,5;0,25 32,0,1;0,25 32,1,3;0,25 32,2,5;0,25 33,0,1;0,25 33,1,3;0,25 33,2,5;0,25 34,0,1;0,25 34,1,3;0,25 34,2,5;0,25 35,0,1;0,25 35,1,3;0,25 35,2,5;0,25 36,0,1;0,25 36,1,3;0,25 36,2,5;0,25 37,0,1;0,25 37,1,3;0,25 37,2,5;0,25 38,0,1;0,25 38,1,3;0,25 38,2,5;0,25 39,0,1;0,25 39,1,3;0,25 39,2,5;0,25 40,0,1;0,25 40,1,3;0,25 40,2,5;0,26 27,0,1;0,26 27,1,3;0,26 27,2,5;0,26 28,0,1;0,26 28,1,3;0,26 28,2,5;0,26 29,0,1;0,26 29,1,3;0,26 29,2,5;0,26 30,0,1;0,26 30,1,3;0,26 30,2,5;0,26 31,0,1;0,26 31,1,3;0,26 31,2,5;0,26 32,0,1;0,26 32,1,3;0,26 32,2,5;0,26 33,0,1;0,26 33,1,3;0,26 33,2,5;0,26 34,0,1;0,26 34,1,3;0,26 34,2,5;0,26 35,0,1;0,26 35,1,3;0,26 35,2,5;0,26 36,0,1;0,26 36,1,3;0,26 36,2,5;0,26 37,0,1;0,26 37,1,3;0,26 37,2,5;0,26 38,0,1;0,26 38,1,3;0,26 38,2,5;0,26 39,0,1;0,26 39,1,3;0,26 39,2,5;0,26 40,0,1;0,26 40,1,3;0,26 40,2,5;0,27 28,0,1;0,27 28,1,3;0,27 28,2,5;0,27 29,0,1;0,27 29,1,3;0,27 29,2,5;0,27 30,0,1;0,27 30,1,3;0,27 30,2,5;0,27 31,0,1;0,27 31,1,3;0,27 31,2,5;0,27 32,0,1;0,27 32,1,3;0,27 32,2,5;0,27 33,0,1;0,27 33,1,3;0,27 33,2,5;0,27 34,0,1;0,27 34,1,3;0,27 34,2,5;0,27 35,0,1;0,27 35,1,3;0,27 35,2,5;0,27 36,0,1;0,27 36,1,3;0,27 36,2,5;0,27 37,0,1;0,27 37,1,3;0,27 37,2,5;0,27 38,0,1;0,27 38,1,3;0,27 38,2,5;0,27 39,0,1;0,27 39,1,3;0,27 39,2,5;0,27 40,0,1;0,27 40,1,3;0,27 40,2,5;0,28 29,0,1;0,28 29,1,3;0,28 29,2,5;0,28 30,0,1;0,28 30,1,3;0,28 30,2,5;0,28 31,0,1;0,28 31,1,3;0,28 31,2,5;0,28 32,0,1;0,28 32,1,3;0,28 32,2,5;0,28 33,0,1;0,28 33,1,3;0,28 33,2,5;0,28 34,0,1;0,28 34,1,3;0,28 34,2,5;0,28 35,0,1;0,28 35,1,3;0,28 35,2,5;0,28 36,0,1;0,28 36,1,3;0,28 36,2,5;0,28 37,0,1;0,28 37,1,3;0,28 37,2,5;0,28 38,0,1;0,28 38,1,3;0,28 38,2,5;0,28 39,0,1;0,28 39,1,3;0,28 39,2,5;0,28 40,0,1;0,28 40,1,3;0,28 40,2,5;0,29 30,0,1;0,29 30,1,3;0,29 30,2,5;0,29 31,0,1;0,29 31,1,3;0,29 31,2,5;0,29 32,0,1;0,29 32,1,3;0,29 32,2,5;0,29 33,0,1;0,29 33,1,3;0,29 33,2,5;0,29 34,0,1;0,29 34,1,3;0,29 34,2,5;0,29 35,0,1;0,29 35,1,3;0,29 35,2,5;0,29 36,0,1;0,29 36,1,3;0,29 36,2,5;0,29 37,0,1;0,29 37,1,3;0,29 37,2,5;0,29 38,0,1;0,29 38,1,3;0,29 38,2,5;0,29 39,0,1;0,29 39,1,3;0,29 39,2,5;0,29 40,0,1;0,29 40,1,3;0,29 40,2,5;0,30 31,0,1;0,30 31,1,3;0,30 31,2,5;0,30 32,0,1;0,30 32,1,3;0,30 32,2,5;0,30 33,0,1;0,30 33,1,3;0,30 33,2,5;0,30 34,0,1;0,30 34,1,3;0,30 34,2,5;0,30 35,0,1;0,30 35,1,3;0,30 35,2,5;0,30 36,0,1;0,30 36,1,3;0,30 36,2,5;0,30 37,0,1;0,30 37,1,3;0,30 37,2,5;0,30 38,0,1;0,30 38,1,3;0,30 38,2,5;0,30 39,0,1;0,30 39,1,3;0,30 39,2,5;0,30 40,0,1;0,30 40,1,3;0,30 40,2,5;0,31 32,0,1;0,31 32,1,3;0,31 32,2,5;0,31 33,0,1;0,31 33,1,3;0,31 33,2,5;0,31 34,0,1;0,31 34,1,3;0,31 34,2,5;0,31 35,0,1;0,31 35,1,3;0,31 35,2,5;0,31 36,0,1;0,31 36,1,3;0,31 36,2,5;0,31 37,0,1;0,31 37,1,3;0,31 37,2,5;0,31 38,0,1;0,31 38,1,3;0,31 38,2,5;0,31 39,0,1;0,31 39,1,3;0,31 39,2,5;0,31 40,0,1;0,31 40,1,3;0,31 40,2,5;0,32 33,0,1;0,32 33,1,3;0,32 33,2,5;0,32 34,0,1;0,32 34,1,3;0,32 34,2,5;0,32 35,0,1;0,32 35,1,3;0,32 35,2,5;0,32 36,0,1;0,32 36,1,3;0,32 36,2,5;0,32 37,0,1;0,32 37,1,3;0,32 37,2,5;0,32 38,0,1;0,32 38,1,3;0,32 38,2,5;0,32 39,0,1;0,32 39,1,3;0,32 39,2,5;0,32 40,0,1;0,32 40,1,3;0,32 40,2,5;0,33 34,0,1;0,33 34,1,3;0,33 34,2,5;0,33 35,0,1;0,33 35,1,3;0,33 35,2,5;0,33 36,0,1;0,33 36,1,3;0,33 36,2,5;0,33 37,0,1;0,33 37,1,3;0,33 37,2,5;0,33 38,0,1;0,33 38,1,3;0,33 38,2,5;0,33 39,0,1;0,33 39,1,3;0,33 39,2,5;0,33 40,0,1;0,33 40,1,3;0,33 40,2,5;0,34 35,0,1;0,34 35,1,3;0,34 35,2,5;0,34 36,0,1;0,34 36,1,3;0,34 36,2,5;0,34 37,0,1;0,34 37,1,3;0,34 37,2,5;0,34 38,0,1;0,34 38,1,3;0,34 38,2,5;0,34 39,0,1;0,34 39,1,3;0,34 39,2,5;0,34 40,0,1;0,34 40,1,3;0,34 40,2,5;0,35 36,0,1;0,35 36,1,3;0,35 36,2,5;0,35 37,0,1;0,35 37,1,3;0,35 37,2,5;0,35 38,0,1;0,35 38,1,3;0,35 38,2,5;0,35 39,0,1;0,35 39,1,3;0,35 39,2,5;0,35 40,0,1;0,35 40,1,3;0,35 40,2,5;0,36 37,0,1;0,36 37,1,3;0,36 37,2,5;0,36 38,0,1;0,36 38,1,3;0,36 38,2,5;0,36 39,0,1;0,36 39,1,3;0,36 39,2,5;0,36 40,0,1;0,36 40,1,3;0,36 40,2,5;0,37 38,0,1;0,37 38,1,3;0,37 38,2,5;0,37 39,0,1;0,37 39,1,3;0,37 39,2,5;0,37 40,0,1;0,37 40,1,3;0,37 40,2,5;0,38 39,0,1;0,38 39,1,3;0,38 39,2,5;0,38 40,0,1;0,38 40,1,3;0,38 40,2,5;0,39 40,0,1;0,39 40,1,3;0,39 40,2,5"

        # "

        # wp = "4604,0,0,1;4604,0,1,1;4604,0,2,1;4604,1,0,1;4604,1,1,1;4604,1,2,1;4604,2,0,1;4604,2,1,1;4604,2,2,1;4604,3,0,1;4604,3,1,1;4604,3,2,1;4604,4,0,1;4604,4,1,1;4604,4,2,1;4604,5,0,1;4604,5,1,1;4604,5,2,1;4604,6,0,1;4604,6,1,1;4604,6,2,1;4604,7,0,1;4604,7,1,1;4604,7,2,1;4604,8,0,1;4604,8,1,1;4604,8,2,1;4604,9,0,1;4604,9,1,1;4604,9,2,1;4604,10,0,1;4604,10,1,1;4604,10,2,1;4604,11,0,1;4604,11,1,1;4604,11,2,1;4604,12,0,1;4604,12,1,1;4604,12,2,1;4604,13,0,1;4604,13,1,1;4604,13,2,1;4604,14,0,1;4604,14,1,1;4604,14,2,1;4604,15,0,1;4604,15,1,1;4604,15,2,1;4604,16,0,1;4604,16,1,1;4604,16,2,1;4604,17,0,1;4604,17,1,1;4604,17,2,1;4604,18,0,1;4604,18,1,1;4604,18,2,1;4604,19,0,1;4604,19,1,1;4604,19,2,1;4604,20,0,1;4604,20,1,1;4604,20,2,1;4604,21,0,1;4604,21,1,1;4604,21,2,1;4604,22,0,1;4604,22,1,1;4604,22,2,1;4604,23,0,1;4604,23,1,1;4604,23,2,1;4604,24,0,1;4604,24,1,1;4604,24,2,1;4604,25,0,1;4604,25,1,1;4604,25,2,1;4604,26,0,1;4604,26,1,1;4604,26,2,1;4604,27,0,1;4604,27,1,1;4604,27,2,1;4604,28,0,1;4604,28,1,1;4604,28,2,1;4604,29,0,1;4604,29,1,1;4604,29,2,1;4604,30,0,1;4604,30,1,1;4604,30,2,1;4604,0 1,0,1;4604,0 1,1,3;4604,0 1,2,5;4604,0 2,0,1;4604,0 2,1,3;4604,0 2,2,5;4604,0 3,0,1;4604,0 3,1,3;4604,0 3,2,5;4604,0 4,0,1;4604,0 4,1,3;4604,0 4,2,5;4604,0 5,0,1;4604,0 5,1,3;4604,0 5,2,5;4604,0 6,0,1;4604,0 6,1,3;4604,0 6,2,5;4604,0 7,0,1;4604,0 7,1,3;4604,0 7,2,5;4604,0 8,0,1;4604,0 8,1,3;4604,0 8,2,5;4604,0 9,0,1;4604,0 9,1,3;4604,0 9,2,5;4604,0 10,0,1;4604,0 10,1,3;4604,0 10,2,5;4604,0 11,0,1;4604,0 11,1,3;4604,0 11,2,5;4604,0 12,0,1;4604,0 12,1,3;4604,0 12,2,5;4604,0 13,0,1;4604,0 13,1,3;4604,0 13,2,5;4604,0 14,0,1;4604,0 14,1,3;4604,0 14,2,5;4604,0 15,0,1;4604,0 15,1,3;4604,0 15,2,5;4604,0 16,0,1;4604,0 16,1,3;4604,0 16,2,5;4604,0 17,0,1;4604,0 17,1,3;4604,0 17,2,5;4604,0 18,0,1;4604,0 18,1,3;4604,0 18,2,5;4604,0 19,0,1;4604,0 19,1,3;4604,0 19,2,5;4604,0 20,0,1;4604,0 20,1,3;4604,0 20,2,5;4604,0 21,0,1;4604,0 21,1,3;4604,0 21,2,5;4604,0 22,0,1;4604,0 22,1,3;4604,0 22,2,5;4604,0 23,0,1;4604,0 23,1,3;4604,0 23,2,5;4604,0 24,0,1;4604,0 24,1,3;4604,0 24,2,5;4604,0 25,0,1;4604,0 25,1,3;4604,0 25,2,5;4604,0 26,0,1;4604,0 26,1,3;4604,0 26,2,5;4604,0 27,0,1;4604,0 27,1,3;4604,0 27,2,5;4604,0 28,0,1;4604,0 28,1,3;4604,0 28,2,5;4604,0 29,0,1;4604,0 29,1,3;4604,0 29,2,5;4604,0 30,0,1;4604,0 30,1,3;4604,0 30,2,5;4604,1 2,0,1;4604,1 2,1,3;4604,1 2,2,5;4604,1 3,0,1;4604,1 3,1,3;4604,1 3,2,5;4604,1 4,0,1;4604,1 4,1,3;4604,1 4,2,5;4604,1 5,0,1;4604,1 5,1,3;4604,1 5,2,5;4604,1 6,0,1;4604,1 6,1,3;4604,1 6,2,5;4604,1 7,0,1;4604,1 7,1,3;4604,1 7,2,5;4604,1 8,0,1;4604,1 8,1,3;4604,1 8,2,5;4604,1 9,0,1;4604,1 9,1,3;4604,1 9,2,5;4604,1 10,0,1;4604,1 10,1,3;4604,1 10,2,5;4604,1 11,0,1;4604,1 11,1,3;4604,1 11,2,5;4604,1 12,0,1;4604,1 12,1,3;4604,1 12,2,5;4604,1 13,0,1;4604,1 13,1,3;4604,1 13,2,5;4604,1 14,0,1;4604,1 14,1,3;4604,1 14,2,5;4604,1 15,0,1;4604,1 15,1,3;4604,1 15,2,5;4604,1 16,0,1;4604,1 16,1,3;4604,1 16,2,5;4604,1 17,0,1;4604,1 17,1,3;4604,1 17,2,5;4604,1 18,0,1;4604,1 18,1,3;4604,1 18,2,5;4604,1 19,0,1;4604,1 19,1,3;4604,1 19,2,5;4604,1 20,0,1;4604,1 20,1,3;4604,1 20,2,5;4604,1 21,0,1;4604,1 21,1,3;4604,1 21,2,5;4604,1 22,0,1;4604,1 22,1,3;4604,1 22,2,5;4604,1 23,0,1;4604,1 23,1,3;4604,1 23,2,5;4604,1 24,0,1;4604,1 24,1,3;4604,1 24,2,5;4604,1 25,0,1;4604,1 25,1,3;4604,1 25,2,5;4604,1 26,0,1;4604,1 26,1,3;4604,1 26,2,5;4604,1 27,0,1;4604,1 27,1,3;4604,1 27,2,5;4604,1 28,0,1;4604,1 28,1,3;4604,1 28,2,5;4604,1 29,0,1;4604,1 29,1,3;4604,1 29,2,5;4604,1 30,0,1;4604,1 30,1,3;4604,1 30,2,5;4604,2 3,0,1;4604,2 3,1,3;4604,2 3,2,5;4604,2 4,0,1;4604,2 4,1,3;4604,2 4,2,5;4604,2 5,0,1;4604,2 5,1,3;4604,2 5,2,5;4604,2 6,0,1;4604,2 6,1,3;4604,2 6,2,5;4604,2 7,0,1;4604,2 7,1,3;4604,2 7,2,5;4604,2 8,0,1;4604,2 8,1,3;4604,2 8,2,5;4604,2 9,0,1;4604,2 9,1,3;4604,2 9,2,5;4604,2 10,0,1;4604,2 10,1,3;4604,2 10,2,5;4604,2 11,0,1;4604,2 11,1,3;4604,2 11,2,5;4604,2 12,0,1;4604,2 12,1,3;4604,2 12,2,5;4604,2 13,0,1;4604,2 13,1,3;4604,2 13,2,5;4604,2 14,0,1;4604,2 14,1,3;4604,2 14,2,5;4604,2 15,0,1;4604,2 15,1,3;4604,2 15,2,5;4604,2 16,0,1;4604,2 16,1,3;4604,2 16,2,5;4604,2 17,0,1;4604,2 17,1,3;4604,2 17,2,5;4604,2 18,0,1;4604,2 18,1,3;4604,2 18,2,5;4604,2 19,0,1;4604,2 19,1,3;4604,2 19,2,5;4604,2 20,0,1;4604,2 20,1,3;4604,2 20,2,5;4604,2 21,0,1;4604,2 21,1,3;4604,2 21,2,5;4604,2 22,0,1;4604,2 22,1,3;4604,2 22,2,5;4604,2 23,0,1;4604,2 23,1,3;4604,2 23,2,5;4604,2 24,0,1;4604,2 24,1,3;4604,2 24,2,5;4604,2 25,0,1;4604,2 25,1,3;4604,2 25,2,5;4604,2 26,0,1;4604,2 26,1,3;4604,2 26,2,5;4604,2 27,0,1;4604,2 27,1,3;4604,2 27,2,5;4604,2 28,0,1;4604,2 28,1,3;4604,2 28,2,5;4604,2 29,0,1;4604,2 29,1,3;4604,2 29,2,5;4604,2 30,0,1;4604,2 30,1,3;4604,2 30,2,5;4604,3 4,0,1;4604,3 4,1,3;4604,3 4,2,5;4604,3 5,0,1;4604,3 5,1,3;4604,3 5,2,5;4604,3 6,0,1;4604,3 6,1,3;4604,3 6,2,5;4604,3 7,0,1;4604,3 7,1,3;4604,3 7,2,5;4604,3 8,0,1;4604,3 8,1,3;4604,3 8,2,5;4604,3 9,0,1;4604,3 9,1,3;4604,3 9,2,5;4604,3 10,0,1;4604,3 10,1,3;4604,3 10,2,5;4604,3 11,0,1;4604,3 11,1,3;4604,3 11,2,5;4604,3 12,0,1;4604,3 12,1,3;4604,3 12,2,5;4604,3 13,0,1;4604,3 13,1,3;4604,3 13,2,5;4604,3 14,0,1;4604,3 14,1,3;4604,3 14,2,5;4604,3 15,0,1;4604,3 15,1,3;4604,3 15,2,5;4604,3 16,0,1;4604,3 16,1,3;4604,3 16,2,5;4604,3 17,0,1;4604,3 17,1,3;4604,3 17,2,5;4604,3 18,0,1;4604,3 18,1,3;4604,3 18,2,5;4604,3 19,0,1;4604,3 19,1,3;4604,3 19,2,5;4604,3 20,0,1;4604,3 20,1,3;4604,3 20,2,5;4604,3 21,0,1;4604,3 21,1,3;4604,3 21,2,5;4604,3 22,0,1;4604,3 22,1,3;4604,3 22,2,5;4604,3 23,0,1;4604,3 23,1,3;4604,3 23,2,5;4604,3 24,0,1;4604,3 24,1,3;4604,3 24,2,5;4604,3 25,0,1;4604,3 25,1,3;4604,3 25,2,5;4604,3 26,0,1;4604,3 26,1,3;4604,3 26,2,5;4604,3 27,0,1;4604,3 27,1,3;4604,3 27,2,5;4604,3 28,0,1;4604,3 28,1,3;4604,3 28,2,5;4604,3 29,0,1;4604,3 29,1,3;4604,3 29,2,5;4604,3 30,0,1;4604,3 30,1,3;4604,3 30,2,5;4604,4 5,0,1;4604,4 5,1,3;4604,4 5,2,5;4604,4 6,0,1;4604,4 6,1,3;4604,4 6,2,5;4604,4 7,0,1;4604,4 7,1,3;4604,4 7,2,5;4604,4 8,0,1;4604,4 8,1,3;4604,4 8,2,5;4604,4 9,0,1;4604,4 9,1,3;4604,4 9,2,5;4604,4 10,0,1;4604,4 10,1,3;4604,4 10,2,5;4604,4 11,0,1;4604,4 11,1,3;4604,4 11,2,5;4604,4 12,0,1;4604,4 12,1,3;4604,4 12,2,5;4604,4 13,0,1;4604,4 13,1,3;4604,4 13,2,5;4604,4 14,0,1;4604,4 14,1,3;4604,4 14,2,5;4604,4 15,0,1;4604,4 15,1,3;4604,4 15,2,5;4604,4 16,0,1;4604,4 16,1,3;4604,4 16,2,5;4604,4 17,0,1;4604,4 17,1,3;4604,4 17,2,5;4604,4 18,0,1;4604,4 18,1,3;4604,4 18,2,5;4604,4 19,0,1;4604,4 19,1,3;4604,4 19,2,5;4604,4 20,0,1;4604,4 20,1,3;4604,4 20,2,5;4604,4 21,0,1;4604,4 21,1,3;4604,4 21,2,5;4604,4 22,0,1;4604,4 22,1,3;4604,4 22,2,5;4604,4 23,0,1;4604,4 23,1,3;4604,4 23,2,5;4604,4 24,0,1;4604,4 24,1,3;4604,4 24,2,5;4604,4 25,0,1;4604,4 25,1,3;4604,4 25,2,5;4604,4 26,0,1;4604,4 26,1,3;4604,4 26,2,5;4604,4 27,0,1;4604,4 27,1,3;4604,4 27,2,5;4604,4 28,0,1;4604,4 28,1,3;4604,4 28,2,5;4604,4 29,0,1;4604,4 29,1,3;4604,4 29,2,5;4604,4 30,0,1;4604,4 30,1,3;4604,4 30,2,5;4604,5 6,0,1;4604,5 6,1,3;4604,5 6,2,5;4604,5 7,0,1;4604,5 7,1,3;4604,5 7,2,5;4604,5 8,0,1;4604,5 8,1,3;4604,5 8,2,5;4604,5 9,0,1;4604,5 9,1,3;4604,5 9,2,5;4604,5 10,0,1;4604,5 10,1,3;4604,5 10,2,5;4604,5 11,0,1;4604,5 11,1,3;4604,5 11,2,5;4604,5 12,0,1;4604,5 12,1,3;4604,5 12,2,5;4604,5 13,0,1;4604,5 13,1,3;4604,5 13,2,5;4604,5 14,0,1;4604,5 14,1,3;4604,5 14,2,5;4604,5 15,0,1;4604,5 15,1,3;4604,5 15,2,5;4604,5 16,0,1;4604,5 16,1,3;4604,5 16,2,5;4604,5 17,0,1;4604,5 17,1,3;4604,5 17,2,5;4604,5 18,0,1;4604,5 18,1,3;4604,5 18,2,5;4604,5 19,0,1;4604,5 19,1,3;4604,5 19,2,5;4604,5 20,0,1;4604,5 20,1,3;4604,5 20,2,5;4604,5 21,0,1;4604,5 21,1,3;4604,5 21,2,5;4604,5 22,0,1;4604,5 22,1,3;4604,5 22,2,5;4604,5 23,0,1;4604,5 23,1,3;4604,5 23,2,5;4604,5 24,0,1;4604,5 24,1,3;4604,5 24,2,5;4604,5 25,0,1;4604,5 25,1,3;4604,5 25,2,5;4604,5 26,0,1;4604,5 26,1,3;4604,5 26,2,5;4604,5 27,0,1;4604,5 27,1,3;4604,5 27,2,5;4604,5 28,0,1;4604,5 28,1,3;4604,5 28,2,5;4604,5 29,0,1;4604,5 29,1,3;4604,5 29,2,5;4604,5 30,0,1;4604,5 30,1,3;4604,5 30,2,5;4604,6 7,0,1;4604,6 7,1,3;4604,6 7,2,5;4604,6 8,0,1;4604,6 8,1,3;4604,6 8,2,5;4604,6 9,0,1;4604,6 9,1,3;4604,6 9,2,5;4604,6 10,0,1;4604,6 10,1,3;4604,6 10,2,5;4604,6 11,0,1;4604,6 11,1,3;4604,6 11,2,5;4604,6 12,0,1;4604,6 12,1,3;4604,6 12,2,5;4604,6 13,0,1;4604,6 13,1,3;4604,6 13,2,5;4604,6 14,0,1;4604,6 14,1,3;4604,6 14,2,5;4604,6 15,0,1;4604,6 15,1,3;4604,6 15,2,5;4604,6 16,0,1;4604,6 16,1,3;4604,6 16,2,5;4604,6 17,0,1;4604,6 17,1,3;4604,6 17,2,5;4604,6 18,0,1;4604,6 18,1,3;4604,6 18,2,5;4604,6 19,0,1;4604,6 19,1,3;4604,6 19,2,5;4604,6 20,0,1;4604,6 20,1,3;4604,6 20,2,5;4604,6 21,0,1;4604,6 21,1,3;4604,6 21,2,5;4604,6 22,0,1;4604,6 22,1,3;4604,6 22,2,5;4604,6 23,0,1;4604,6 23,1,3;4604,6 23,2,5;4604,6 24,0,1;4604,6 24,1,3;4604,6 24,2,5;4604,6 25,0,1;4604,6 25,1,3;4604,6 25,2,5;4604,6 26,0,1;4604,6 26,1,3;4604,6 26,2,5;4604,6 27,0,1;4604,6 27,1,3;4604,6 27,2,5;4604,6 28,0,1;4604,6 28,1,3;4604,6 28,2,5;4604,6 29,0,1;4604,6 29,1,3;4604,6 29,2,5;4604,6 30,0,1;4604,6 30,1,3;4604,6 30,2,5;4604,7 8,0,1;4604,7 8,1,3;4604,7 8,2,5;4604,7 9,0,1;4604,7 9,1,3;4604,7 9,2,5;4604,7 10,0,1;4604,7 10,1,3;4604,7 10,2,5;4604,7 11,0,1;4604,7 11,1,3;4604,7 11,2,5;4604,7 12,0,1;4604,7 12,1,3;4604,7 12,2,5;4604,7 13,0,1;4604,7 13,1,3;4604,7 13,2,5;4604,7 14,0,1;4604,7 14,1,3;4604,7 14,2,5;4604,7 15,0,1;4604,7 15,1,3;4604,7 15,2,5;4604,7 16,0,1;4604,7 16,1,3;4604,7 16,2,5;4604,7 17,0,1;4604,7 17,1,3;4604,7 17,2,5;4604,7 18,0,1;4604,7 18,1,3;4604,7 18,2,5;4604,7 19,0,1;4604,7 19,1,3;4604,7 19,2,5;4604,7 20,0,1;4604,7 20,1,3;4604,7 20,2,5;4604,7 21,0,1;4604,7 21,1,3;4604,7 21,2,5;4604,7 22,0,1;4604,7 22,1,3;4604,7 22,2,5;4604,7 23,0,1;4604,7 23,1,3;4604,7 23,2,5;4604,7 24,0,1;4604,7 24,1,3;4604,7 24,2,5;4604,7 25,0,1;4604,7 25,1,3;4604,7 25,2,5;4604,7 26,0,1;4604,7 26,1,3;4604,7 26,2,5;4604,7 27,0,1;4604,7 27,1,3;4604,7 27,2,5;4604,7 28,0,1;4604,7 28,1,3;4604,7 28,2,5;4604,7 29,0,1;4604,7 29,1,3;4604,7 29,2,5;4604,7 30,0,1;4604,7 30,1,3;4604,7 30,2,5;4604,8 9,0,1;4604,8 9,1,3;4604,8 9,2,5;4604,8 10,0,1;4604,8 10,1,3;4604,8 10,2,5;4604,8 11,0,1;4604,8 11,1,3;4604,8 11,2,5;4604,8 12,0,1;4604,8 12,1,3;4604,8 12,2,5;4604,8 13,0,1;4604,8 13,1,3;4604,8 13,2,5;4604,8 14,0,1;4604,8 14,1,3;4604,8 14,2,5;4604,8 15,0,1;4604,8 15,1,3;4604,8 15,2,5;4604,8 16,0,1;4604,8 16,1,3;4604,8 16,2,5;4604,8 17,0,1;4604,8 17,1,3;4604,8 17,2,5;4604,8 18,0,1;4604,8 18,1,3;4604,8 18,2,5;4604,8 19,0,1;4604,8 19,1,3;4604,8 19,2,5;4604,8 20,0,1;4604,8 20,1,3;4604,8 20,2,5;4604,8 21,0,1;4604,8 21,1,3;4604,8 21,2,5;4604,8 22,0,1;4604,8 22,1,3;4604,8 22,2,5;4604,8 23,0,1;4604,8 23,1,3;4604,8 23,2,5;4604,8 24,0,1;4604,8 24,1,3;4604,8 24,2,5;4604,8 25,0,1;4604,8 25,1,3;4604,8 25,2,5;4604,8 26,0,1;4604,8 26,1,3;4604,8 26,2,5;4604,8 27,0,1;4604,8 27,1,3;4604,8 27,2,5;4604,8 28,0,1;4604,8 28,1,3;4604,8 28,2,5;4604,8 29,0,1;4604,8 29,1,3;4604,8 29,2,5;4604,8 30,0,1;4604,8 30,1,3;4604,8 30,2,5;4604,9 10,0,1;4604,9 10,1,3;4604,9 10,2,5;4604,9 11,0,1;4604,9 11,1,3;4604,9 11,2,5;4604,9 12,0,1;4604,9 12,1,3;4604,9 12,2,5;4604,9 13,0,1;4604,9 13,1,3;4604,9 13,2,5;4604,9 14,0,1;4604,9 14,1,3;4604,9 14,2,5;4604,9 15,0,1;4604,9 15,1,3;4604,9 15,2,5;4604,9 16,0,1;4604,9 16,1,3;4604,9 16,2,5;4604,9 17,0,1;4604,9 17,1,3;4604,9 17,2,5;4604,9 18,0,1;4604,9 18,1,3;4604,9 18,2,5;4604,9 19,0,1;4604,9 19,1,3;4604,9 19,2,5;4604,9 20,0,1;4604,9 20,1,3;4604,9 20,2,5;4604,9 21,0,1;4604,9 21,1,3;4604,9 21,2,5;4604,9 22,0,1;4604,9 22,1,3;4604,9 22,2,5;4604,9 23,0,1;4604,9 23,1,3;4604,9 23,2,5;4604,9 24,0,1;4604,9 24,1,3;4604,9 24,2,5;4604,9 25,0,1;4604,9 25,1,3;4604,9 25,2,5;4604,9 26,0,1;4604,9 26,1,3;4604,9 26,2,5;4604,9 27,0,1;4604,9 27,1,3;4604,9 27,2,5;4604,9 28,0,1;4604,9 28,1,3;4604,9 28,2,5;4604,9 29,0,1;4604,9 29,1,3;4604,9 29,2,5;4604,9 30,0,1;4604,9 30,1,3;4604,9 30,2,5;4604,10 11,0,1;4604,10 11,1,3;4604,10 11,2,5;4604,10 12,0,1;4604,10 12,1,3;4604,10 12,2,5;4604,10 13,0,1;4604,10 13,1,3;4604,10 13,2,5;4604,10 14,0,1;4604,10 14,1,3;4604,10 14,2,5;4604,10 15,0,1;4604,10 15,1,3;4604,10 15,2,5;4604,10 16,0,1;4604,10 16,1,3;4604,10 16,2,5;4604,10 17,0,1;4604,10 17,1,3;4604,10 17,2,5;4604,10 18,0,1;4604,10 18,1,3;4604,10 18,2,5;4604,10 19,0,1;4604,10 19,1,3;4604,10 19,2,5;4604,10 20,0,1;4604,10 20,1,3;4604,10 20,2,5;4604,10 21,0,1;4604,10 21,1,3;4604,10 21,2,5;4604,10 22,0,1;4604,10 22,1,3;4604,10 22,2,5;4604,10 23,0,1;4604,10 23,1,3;4604,10 23,2,5;4604,10 24,0,1;4604,10 24,1,3;4604,10 24,2,5;4604,10 25,0,1;4604,10 25,1,3;4604,10 25,2,5;4604,10 26,0,1;4604,10 26,1,3;4604,10 26,2,5;4604,10 27,0,1;4604,10 27,1,3;4604,10 27,2,5;4604,10 28,0,1;4604,10 28,1,3;4604,10 28,2,5;4604,10 29,0,1;4604,10 29,1,3;4604,10 29,2,5;4604,10 30,0,1;4604,10 30,1,3;4604,10 30,2,5;4604,11 12,0,1;4604,11 12,1,3;4604,11 12,2,5;4604,11 13,0,1;4604,11 13,1,3;4604,11 13,2,5;4604,11 14,0,1;4604,11 14,1,3;4604,11 14,2,5;4604,11 15,0,1;4604,11 15,1,3;4604,11 15,2,5;4604,11 16,0,1;4604,11 16,1,3;4604,11 16,2,5;4604,11 17,0,1;4604,11 17,1,3;4604,11 17,2,5;4604,11 18,0,1;4604,11 18,1,3;4604,11 18,2,5;4604,11 19,0,1;4604,11 19,1,3;4604,11 19,2,5;4604,11 20,0,1;4604,11 20,1,3;4604,11 20,2,5;4604,11 21,0,1;4604,11 21,1,3;4604,11 21,2,5;4604,11 22,0,1;4604,11 22,1,3;4604,11 22,2,5;4604,11 23,0,1;4604,11 23,1,3;4604,11 23,2,5;4604,11 24,0,1;4604,11 24,1,3;4604,11 24,2,5;4604,11 25,0,1;4604,11 25,1,3;4604,11 25,2,5;4604,11 26,0,1;4604,11 26,1,3;4604,11 26,2,5;4604,11 27,0,1;4604,11 27,1,3;4604,11 27,2,5;4604,11 28,0,1;4604,11 28,1,3;4604,11 28,2,5;4604,11 29,0,1;4604,11 29,1,3;4604,11 29,2,5;4604,11 30,0,1;4604,11 30,1,3;4604,11 30,2,5;4604,12 13,0,1;4604,12 13,1,3;4604,12 13,2,5;4604,12 14,0,1;4604,12 14,1,3;4604,12 14,2,5;4604,12 15,0,1;4604,12 15,1,3;4604,12 15,2,5;4604,12 16,0,1;4604,12 16,1,3;4604,12 16,2,5;4604,12 17,0,1;4604,12 17,1,3;4604,12 17,2,5;4604,12 18,0,1;4604,12 18,1,3;4604,12 18,2,5;4604,12 19,0,1;4604,12 19,1,3;4604,12 19,2,5;4604,12 20,0,1;4604,12 20,1,3;4604,12 20,2,5;4604,12 21,0,1;4604,12 21,1,3;4604,12 21,2,5;4604,12 22,0,1;4604,12 22,1,3;4604,12 22,2,5;4604,12 23,0,1;4604,12 23,1,3;4604,12 23,2,5;4604,12 24,0,1;4604,12 24,1,3;4604,12 24,2,5;4604,12 25,0,1;4604,12 25,1,3;4604,12 25,2,5;4604,12 26,0,1;4604,12 26,1,3;4604,12 26,2,5;4604,12 27,0,1;4604,12 27,1,3;4604,12 27,2,5;4604,12 28,0,1;4604,12 28,1,3;4604,12 28,2,5;4604,12 29,0,1;4604,12 29,1,3;4604,12 29,2,5;4604,12 30,0,1;4604,12 30,1,3;4604,12 30,2,5;4604,13 14,0,1;4604,13 14,1,3;4604,13 14,2,5;4604,13 15,0,1;4604,13 15,1,3;4604,13 15,2,5;4604,13 16,0,1;4604,13 16,1,3;4604,13 16,2,5;4604,13 17,0,1;4604,13 17,1,3;4604,13 17,2,5;4604,13 18,0,1;4604,13 18,1,3;4604,13 18,2,5;4604,13 19,0,1;4604,13 19,1,3;4604,13 19,2,5;4604,13 20,0,1;4604,13 20,1,3;4604,13 20,2,5;4604,13 21,0,1;4604,13 21,1,3;4604,13 21,2,5;4604,13 22,0,1;4604,13 22,1,3;4604,13 22,2,5;4604,13 23,0,1;4604,13 23,1,3;4604,13 23,2,5;4604,13 24,0,1;4604,13 24,1,3;4604,13 24,2,5;4604,13 25,0,1"
        # #"

        # wp = "2257,9 15,1,3;2257,9 15,2,5;2257,9 16,0,1;2257,9 16,1,3;2257,9 16,2,5;2257,9 17,0,1;2257,9 17,1,3;2257,9 17,2,5;2257,9 18,0,1;2257,9 18,1,3;2257,9 18,2,5;2257,9 19,0,1;2257,9 19,1,3;2257,9 19,2,5;2257,9 20,0,1;2257,9 20,1,3;2257,9 20,2,5;2257,9 21,0,1;2257,9 21,1,3;2257,9 21,2,5;2257,9 22,0,1;2257,9 22,1,3;2257,9 22,2,5;2257,9 23,0,1;2257,9 23,1,3;2257,9 23,2,5;2257,9 24,0,1;2257,9 24,1,3;2257,9 24,2,5;2257,9 25,0,1;2257,9 25,1,3;2257,9 25,2,5;2257,9 26,0,1;2257,9 26,1,3;2257,9 26,2,5;2257,9 27,0,1;2257,9 27,1,3;2257,9 27,2,5;2257,9 28,0,1;2257,9 28,1,3;2257,9 28,2,5;2257,9 29,0,1;2257,9 29,1,3;2257,9 29,2,5;2257,9 30,0,1;2257,9 30,1,3;2257,9 30,2,5;2257,9 31,0,1;2257,9 31,1,3;2257,9 31,2,5;2257,9 32,0,1;2257,9 32,1,3;2257,9 32,2,5;2257,9 33,0,1;2257,9 33,1,3;2257,9 33,2,5;2257,9 34,0,1;2257,9 34,1,3;2257,9 34,2,5;2257,9 35,0,1;2257,9 35,1,3;2257,9 35,2,5;2257,9 36,0,1;2257,9 36,1,3;2257,9 36,2,5;2257,9 37,0,1;2257,9 37,1,3;2257,9 37,2,5;2257,9 38,0,1;2257,9 38,1,3;2257,9 38,2,5;2257,9 39,0,1;2257,9 39,1,3;2257,9 39,2,5;2257,10 11,0,1;2257,10 11,1,3;2257,10 11,2,5;2257,10 12,0,1;2257,10 12,1,3;2257,10 12,2,5;2257,10 13,0,1;2257,10 13,1,3;2257,10 13,2,5;2257,10 14,0,1;2257,10 14,1,3;2257,10 14,2,5;2257,10 15,0,1;2257,10 15,1,3;2257,10 15,2,5;2257,10 16,0,1;2257,10 16,1,3;2257,10 16,2,5;2257,10 17,0,1;2257,10 17,1,3;2257,10 17,2,5;2257,10 18,0,1;2257,10 18,1,3;2257,10 18,2,5;2257,10 19,0,1;2257,10 19,1,3;2257,10 19,2,5;2257,10 20,0,1;2257,10 20,1,3;2257,10 20,2,5;2257,10 21,0,1;2257,10 21,1,3;2257,10 21,2,5;2257,10 22,0,1;2257,10 22,1,3;2257,10 22,2,5;2257,10 23,0,1;2257,10 23,1,3;2257,10 23,2,5;2257,10 24,0,1;2257,10 24,1,3;2257,10 24,2,5;2257,10 25,0,1;2257,10 25,1,3;2257,10 25,2,5;2257,10 26,0,1;2257,10 26,1,3;2257,10 26,2,5;2257,10 27,0,1;2257,10 27,1,3;2257,10 27,2,5;2257,10 28,0,1;2257,10 28,1,3;2257,10 28,2,5;2257,10 29,0,1;2257,10 29,1,3;2257,10 29,2,5;2257,10 30,0,1;2257,10 30,1,3;2257,10 30,2,5;2257,10 31,0,1;2257,10 31,1,3;2257,10 31,2,5;2257,10 32,0,1;2257,10 32,1,3;2257,10 32,2,5;2257,10 33,0,1;2257,10 33,1,3;2257,10 33,2,5;2257,10 34,0,1;2257,10 34,1,3;2257,10 34,2,5;2257,10 35,0,1;2257,10 35,1,3;2257,10 35,2,5;2257,10 36,0,1;2257,10 36,1,3;2257,10 36,2,5;2257,10 37,0,1;2257,10 37,1,3;2257,10 37,2,5;2257,10 38,0,1;2257,10 38,1,3;2257,10 38,2,5;2257,10 39,0,1;2257,10 39,1,3;2257,10 39,2,5;2257,11 12,0,1;2257,11 12,1,3;2257,11 12,2,5;2257,11 13,0,1;2257,11 13,1,3;2257,11 13,2,5;2257,11 14,0,1;2257,11 14,1,3;2257,11 14,2,5;2257,11 15,0,1;2257,11 15,1,3;2257,11 15,2,5;2257,11 16,0,1;2257,11 16,1,3;2257,11 16,2,5;2257,11 17,0,1;2257,11 17,1,3;2257,11 17,2,5;2257,11 18,0,1;2257,11 18,1,3;2257,11 18,2,5;2257,11 19,0,1;2257,11 19,1,3;2257,11 19,2,5;2257,11 20,0,1;2257,11 20,1,3;2257,11 20,2,5;2257,11 21,0,1;2257,11 21,1,3;2257,11 21,2,5;2257,11 22,0,1;2257,11 22,1,3;2257,11 22,2,5;2257,11 23,0,1;2257,11 23,1,3;2257,11 23,2,5;2257,11 24,0,1;2257,11 24,1,3;2257,11 24,2,5;2257,11 25,0,1;2257,11 25,1,3;2257,11 25,2,5;2257,11 26,0,1;2257,11 26,1,3;2257,11 26,2,5;2257,11 27,0,1;2257,11 27,1,3;2257,11 27,2,5;2257,11 28,0,1;2257,11 28,1,3;2257,11 28,2,5;2257,11 29,0,1;2257,11 29,1,3;2257,11 29,2,5;2257,11 30,0,1;2257,11 30,1,3;2257,11 30,2,5;2257,11 31,0,1;2257,11 31,1,3;2257,11 31,2,5;2257,11 32,0,1;2257,11 32,1,3;2257,11 32,2,5;2257,11 33,0,1;2257,11 33,1,3;2257,11 33,2,5;2257,11 34,0,1;2257,11 34,1,3;2257,11 34,2,5;2257,11 35,0,1;2257,11 35,1,3;2257,11 35,2,5;2257,11 36,0,1;2257,11 36,1,3;2257,11 36,2,5;2257,11 37,0,1;2257,11 37,1,3;2257,11 37,2,5;2257,11 38,0,1;2257,11 38,1,3;2257,11 38,2,5;2257,11 39,0,1;2257,11 39,1,3;2257,11 39,2,5;2257,12 13,0,1;2257,12 13,1,3;2257,12 13,2,5;2257,12 14,0,1;2257,12 14,1,3;2257,12 14,2,5;2257,12 15,0,1;2257,12 15,1,3;2257,12 15,2,5;2257,12 16,0,1;2257,12 16,1,3;2257,12 16,2,5;2257,12 17,0,1;2257,12 17,1,3;2257,12 17,2,5;2257,12 18,0,1;2257,12 18,1,3;2257,12 18,2,5;2257,12 19,0,1;2257,12 19,1,3;2257,12 19,2,5;2257,12 20,0,1;2257,12 20,1,3;2257,12 20,2,5;2257,12 21,0,1;2257,12 21,1,3;2257,12 21,2,5;2257,12 22,0,1;2257,12 22,1,3;2257,12 22,2,5;2257,12 23,0,1;2257,12 23,1,3;2257,12 23,2,5;2257,12 24,0,1;2257,12 24,1,3;2257,12 24,2,5;2257,12 25,0,1;2257,12 25,1,3;2257,12 25,2,5;2257,12 26,0,1;2257,12 26,1,3;2257,12 26,2,5;2257,12 27,0,1;2257,12 27,1,3;2257,12 27,2,5;2257,12 28,0,1;2257,12 28,1,3;2257,12 28,2,5;2257,12 29,0,1;2257,12 29,1,3;2257,12 29,2,5;2257,12 30,0,1;2257,12 30,1,3;2257,12 30,2,5;2257,12 31,0,1;2257,12 31,1,3;2257,12 31,2,5;2257,12 32,0,1;2257,12 32,1,3;2257,12 32,2,5;2257,12 33,0,1;2257,12 33,1,3;2257,12 33,2,5;2257,12 34,0,1;2257,12 34,1,3;2257,12 34,2,5;2257,12 35,0,1;2257,12 35,1,3;2257,12 35,2,5;2257,12 36,0,1;2257,12 36,1,3;2257,12 36,2,5;2257,12 37,0,1;2257,12 37,1,3;2257,12 37,2,5;2257,12 38,0,1;2257,12 38,1,3;2257,12 38,2,5;2257,12 39,0,1;2257,12 39,1,3;2257,12 39,2,5;2257,13 14,0,1;2257,13 14,1,3;2257,13 14,2,5;2257,13 15,0,1;2257,13 15,1,3;2257,13 15,2,5;2257,13 16,0,1;2257,13 16,1,3;2257,13 16,2,5;2257,13 17,0,1;2257,13 17,1,3;2257,13 17,2,5;2257,13 18,0,1;2257,13 18,1,3;2257,13 18,2,5;2257,13 19,0,1;2257,13 19,1,3;2257,13 19,2,5;2257,13 20,0,1;2257,13 20,1,3;2257,13 20,2,5;2257,13 21,0,1;2257,13 21,1,3;2257,13 21,2,5;2257,13 22,0,1;2257,13 22,1,3;2257,13 22,2,5;2257,13 23,0,1;2257,13 23,1,3;2257,13 23,2,5;2257,13 24,0,1;2257,13 24,1,3;2257,13 24,2,5;2257,13 25,0,1;2257,13 25,1,3;2257,13 25,2,5;2257,13 26,0,1;2257,13 26,1,3;2257,13 26,2,5;2257,13 27,0,1;2257,13 27,1,3;2257,13 27,2,5;2257,13 28,0,1;2257,13 28,1,3;2257,13 28,2,5;2257,13 29,0,1;2257,13 29,1,3;2257,13 29,2,5;2257,13 30,0,1;2257,13 30,1,3;2257,13 30,2,5;2257,13 31,0,1;2257,13 31,1,3;2257,13 31,2,5;2257,13 32,0,1;2257,13 32,1,3;2257,13 32,2,5;2257,13 33,0,1;2257,13 33,1,3;2257,13 33,2,5;2257,13 34,0,1;2257,13 34,1,3;2257,13 34,2,5;2257,13 35,0,1;2257,13 35,1,3;2257,13 35,2,5;2257,13 36,0,1;2257,13 36,1,3;2257,13 36,2,5;2257,13 37,0,1;2257,13 37,1,3;2257,13 37,2,5;2257,13 38,0,1;2257,13 38,1,3;2257,13 38,2,5;2257,13 39,0,1;2257,13 39,1,3;2257,13 39,2,5;2257,14 15,0,1;2257,14 15,1,3;2257,14 15,2,5;2257,14 16,0,1;2257,14 16,1,3;2257,14 16,2,5;2257,14 17,0,1;2257,14 17,1,3;2257,14 17,2,5;2257,14 18,0,1;2257,14 18,1,3;2257,14 18,2,5;2257,14 19,0,1;2257,14 19,1,3;2257,14 19,2,5;2257,14 20,0,1;2257,14 20,1,3;2257,14 20,2,5;2257,14 21,0,1;2257,14 21,1,3;2257,14 21,2,5;2257,14 22,0,1;2257,14 22,1,3;2257,14 22,2,5;2257,14 23,0,1;2257,14 23,1,3;2257,14 23,2,5;2257,14 24,0,1;2257,14 24,1,3;2257,14 24,2,5;2257,14 25,0,1;2257,14 25,1,3;2257,14 25,2,5;2257,14 26,0,1;2257,14 26,1,3;2257,14 26,2,5;2257,14 27,0,1;2257,14 27,1,3;2257,14 27,2,5;2257,14 28,0,1;2257,14 28,1,3;2257,14 28,2,5;2257,14 29,0,1;2257,14 29,1,3;2257,14 29,2,5;2257,14 30,0,1;2257,14 30,1,3;2257,14 30,2,5;2257,14 31,0,1;2257,14 31,1,3;2257,14 31,2,5;2257,14 32,0,1;2257,14 32,1,3;2257,14 32,2,5;2257,14 33,0,1;2257,14 33,1,3;2257,14 33,2,5;2257,14 34,0,1;2257,14 34,1,3;2257,14 34,2,5;2257,14 35,0,1;2257,14 35,1,3;2257,14 35,2,5;2257,14 36,0,1;2257,14 36,1,3;2257,14 36,2,5;2257,14 37,0,1;2257,14 37,1,3;2257,14 37,2,5;2257,14 38,0,1;2257,14 38,1,3;2257,14 38,2,5;2257,14 39,0,1;2257,14 39,1,3;2257,14 39,2,5;2257,15 16,0,1;2257,15 16,1,3;2257,15 16,2,5;2257,15 17,0,1;2257,15 17,1,3;2257,15 17,2,5;2257,15 18,0,1;2257,15 18,1,3;2257,15 18,2,5;2257,15 19,0,1;2257,15 19,1,3;2257,15 19,2,5;2257,15 20,0,1;2257,15 20,1,3;2257,15 20,2,5;2257,15 21,0,1;2257,15 21,1,3;2257,15 21,2,5;2257,15 22,0,1;2257,15 22,1,3;2257,15 22,2,5;2257,15 23,0,1;2257,15 23,1,3;2257,15 23,2,5;2257,15 24,0,1;2257,15 24,1,3;2257,15 24,2,5;2257,15 25,0,1;2257,15 25,1,3;2257,15 25,2,5;2257,15 26,0,1;2257,15 26,1,3;2257,15 26,2,5;2257,15 27,0,1;2257,15 27,1,3;2257,15 27,2,5;2257,15 28,0,1;2257,15 28,1,3;2257,15 28,2,5;2257,15 29,0,1;2257,15 29,1,3;2257,15 29,2,5;2257,15 30,0,1;2257,15 30,1,3;2257,15 30,2,5;2257,15 31,0,1;2257,15 31,1,3;2257,15 31,2,5;2257,15 32,0,1;2257,15 32,1,3;2257,15 32,2,5;2257,15 33,0,1;2257,15 33,1,3;2257,15 33,2,5;2257,15 34,0,1;2257,15 34,1,3;2257,15 34,2,5;2257,15 35,0,1;2257,15 35,1,3;2257,15 35,2,5;2257,15 36,0,1;2257,15 36,1,3;2257,15 36,2,5;2257,15 37,0,1;2257,15 37,1,3;2257,15 37,2,5;2257,15 38,0,1;2257,15 38,1,3;2257,15 38,2,5;2257,15 39,0,1;2257,15 39,1,3;2257,15 39,2,5;2257,16 17,0,1;2257,16 17,1,3;2257,16 17,2,5;2257,16 18,0,1;2257,16 18,1,3;2257,16 18,2,5;2257,16 19,0,1;2257,16 19,1,3;2257,16 19,2,5;2257,16 20,0,1;2257,16 20,1,3;2257,16 20,2,5;2257,16 21,0,1;2257,16 21,1,3;2257,16 21,2,5;2257,16 22,0,1;2257,16 22,1,3;2257,16 22,2,5;2257,16 23,0,1;2257,16 23,1,3;2257,16 23,2,5;2257,16 24,0,1;2257,16 24,1,3;2257,16 24,2,5;2257,16 25,0,1;2257,16 25,1,3;2257,16 25,2,5;2257,16 26,0,1;2257,16 26,1,3;2257,16 26,2,5;2257,16 27,0,1;2257,16 27,1,3;2257,16 27,2,5;2257,16 28,0,1;2257,16 28,1,3;2257,16 28,2,5;2257,16 29,0,1;2257,16 29,1,3;2257,16 29,2,5;2257,16 30,0,1;2257,16 30,1,3;2257,16 30,2,5;2257,16 31,0,1;2257,16 31,1,3;2257,16 31,2,5;2257,16 32,0,1;2257,16 32,1,3;2257,16 32,2,5;2257,16 33,0,1;2257,16 33,1,3;2257,16 33,2,5;2257,16 34,0,1;2257,16 34,1,3;2257,16 34,2,5;2257,16 35,0,1;2257,16 35,1,3;2257,16 35,2,5;2257,16 36,0,1;2257,16 36,1,3;2257,16 36,2,5;2257,16 37,0,1;2257,16 37,1,3;2257,16 37,2,5;2257,16 38,0,1;2257,16 38,1,3;2257,16 38,2,5;2257,16 39,0,1;2257,16 39,1,3;2257,16 39,2,5;2257,17 18,0,1;2257,17 18,1,3;2257,17 18,2,5;2257,17 19,0,1;2257,17 19,1,3;2257,17 19,2,5;2257,17 20,0,1;2257,17 20,1,3;2257,17 20,2,5;2257,17 21,0,1;2257,17 21,1,3;2257,17 21,2,5;2257,17 22,0,1;2257,17 22,1,3;2257,17 22,2,5;2257,17 23,0,1;2257,17 23,1,3;2257,17 23,2,5;2257,17 24,0,1;2257,17 24,1,3;2257,17 24,2,5;2257,17 25,0,1;2257,17 25,1,3;2257,17 25,2,5;2257,17 26,0,1;2257,17 26,1,3;2257,17 26,2,5;2257,17 27,0,1;2257,17 27,1,3;2257,17 27,2,5;2257,17 28,0,1;2257,17 28,1,3;2257,17 28,2,5;2257,17 29,0,1;2257,17 29,1,3;2257,17 29,2,5;2257,17 30,0,1;2257,17 30,1,3;2257,17 30,2,5;2257,17 31,0,1;2257,17 31,1,3;2257,17 31,2,5;2257,17 32,0,1;2257,17 32,1,3;2257,17 32,2,5;2257,17 33,0,1;2257,17 33,1,3;2257,17 33,2,5;2257,17 34,0,1;2257,17 34,1,3;2257,17 34,2,5;2257,17 35,0,1;2257,17 35,1,3;2257,17 35,2,5;2257,17 36,0,1;2257,17 36,1,3;2257,17 36,2,5;2257,17 37,0,1;2257,17 37,1,3;2257,17 37,2,5;2257,17 38,0,1;2257,17 38,1,3;2257,17 38,2,5;2257,17 39,0,1;2257,17 39,1,3;2257,17 39,2,5;2257,18 19,0,1;2257,18 19,1,3;2257,18 19,2,5;2257,18 20,0,1;2257,18 20,1,3;2257,18 20,2,5;2257,18 21,0,1;2257,18 21,1,3;2257,18 21,2,5;2257,18 22,0,1;2257,18 22,1,3;2257,18 22,2,5;2257,18 23,0,1;2257,18 23,1,3;2257,18 23,2,5;2257,18 24,0,1;2257,18 24,1,3;2257,18 24,2,5;2257,18 25,0,1;2257,18 25,1,3;2257,18 25,2,5;2257,18 26,0,1;2257,18 26,1,3;2257,18 26,2,5;2257,18 27,0,1;2257,18 27,1,3;2257,18 27,2,5;2257,18 28,0,1;2257,18 28,1,3;2257,18 28,2,5;2257,18 29,0,1;2257,18 29,1,3;2257,18 29,2,5;2257,18 30,0,1;2257,18 30,1,3;2257,18 30,2,5;2257,18 31,0,1;2257,18 31,1,3;2257,18 31,2,5;2257,18 32,0,1;2257,18 32,1,3;2257,18 32,2,5;2257,18 33,0,1;2257,18 33,1,3;2257,18 33,2,5;2257,18 34,0,1;2257,18 34,1,3;2257,18 34,2,5;2257,18 35,0,1;2257,18 35,1,3;2257,18 35,2,5;2257,18 36,0,1;2257,18 36,1,3;2257,18 36,2,5;2257,18 37,0,1;2257,18 37,1,3;2257,18 37,2,5;2257,18 38,0,1;2257,18 38,1,3;2257,18 38,2,5;2257,18 39,0,1;2257,18 39,1,3;2257,18 39,2,5;2257,19 20,0,1;2257,19 20,1,3;2257,19 20,2,5;2257,19 21,0,1;2257,19 21,1,3;2257,19 21,2,5;2257,19 22,0,1;2257,19 22,1,3;2257,19 22,2,5;2257,19 23,0,1;2257,19 23,1,3;2257,19 23,2,5;2257,19 24,0,1;2257,19 24,1,3;2257,19 24,2,5;2257,19 25,0,1;2257,19 25,1,3;2257,19 25,2,5;2257,19 26,0,1;2257,19 26,1,3;2257,19 26,2,5;2257,19 27,0,1;2257,19 27,1,3;2257,19 27,2,5;2257,19 28,0,1;2257,19 28,1,3;2257,19 28,2,5;2257,19 29,0,1;2257,19 29,1,3;2257,19 29,2,5;2257,19 30,0,1;2257,19 30,1,3;2257,19 30,2,5;2257,19 31,0,1;2257,19 31,1,3;2257,19 31,2,5;2257,19 32,0,1;2257,19 32,1,3;2257,19 32,2,5;2257,19 33,0,1;2257,19 33,1,3;2257,19 33,2,5;2257,19 34,0,1;2257,19 34,1,3;2257,19 34,2,5;2257,19 35,0,1;2257,19 35,1,3;2257,19 35,2,5;2257,19 36,0,1;2257,19 36,1,3;2257,19 36,2,5;2257,19 37,0,1;2257,19 37,1,3;2257,19 37,2,5;2257,19 38,0,1;2257,19 38,1,3;2257,19 38,2,5;2257,19 39,0,1;2257,19 39,1,3;2257,19 39,2,5;2257,20 21,0,1;2257,20 21,1,3;2257,20 21,2,5;2257,20 22,0,1;2257,20 22,1,3;2257,20 22,2,5;2257,20 23,0,1;2257,20 23,1,3;2257,20 23,2,5;2257,20 24,0,1;2257,20 24,1,3;2257,20 24,2,5;2257,20 25,0,1;2257,20 25,1,3;2257,20 25,2,5;2257,20 26,0,1;2257,20 26,1,3;2257,20 26,2,5;2257,20 27,0,1;2257,20 27,1,3;2257,20 27,2,5;2257,20 28,0,1;2257,20 28,1,3;2257,20 28,2,5;2257,20 29,0,1;2257,20 29,1,3;2257,20 29,2,5;2257,20 30,0,1;2257,20 30,1,3;2257,20 30,2,5;2257,20 31,0,1;2257,20 31,1,3;2257,20 31,2,5;2257,20 32,0,1;2257,20 32,1,3;2257,20 32,2,5;2257,20 33,0,1;2257,20 33,1,3;2257,20 33,2,5;2257,20 34,0,1;2257,20 34,1,3;2257,20 34,2,5;2257,20 35,0,1;2257,20 35,1,3;2257,20 35,2,5;2257,20 36,0,1;2257,20 36,1,3;2257,20 36,2,5;2257,20 37,0,1;2257,20 37,1,3;2257,20 37,2,5;2257,20 38,0,1;2257,20 38,1,3;2257,20 38,2,5;2257,20 39,0,1;2257,20 39,1,3;2257,20 39,2,5;2257,21 22,0,1;2257,21 22,1,3;2257,21 22,2,5;2257,21 23,0,1;2257,21 23,1,3;2257,21 23,2,5;2257,21 24,0,1;2257,21 24,1,3;2257,21 24,2,5;2257,21 25,0,1;2257,21 25,1,3;2257,21 25,2,5;2257,21 26,0,1;2257,21 26,1,3;2257,21 26,2,5;2257,21 27,0,1;2257,21 27,1,3;2257,21 27,2,5;2257,21 28,0,1;2257,21 28,1,3;2257,21 28,2,5;2257,21 29,0,1;2257,21 29,1,3;2257,21 29,2,5;2257,21 30,0,1;2257,21 30,1,3;2257,21 30,2,5;2257,21 31,0,1;2257,21 31,1,3;2257,21 31,2,5;2257,21 32,0,1;2257,21 32,1,3;2257,21 32,2,5;2257,21 33,0,1;2257,21 33,1,3;2257,21 33,2,5;2257,21 34,0,1;2257,21 34,1,3;2257,21 34,2,5;2257,21 35,0,1;2257,21 35,1,3;2257,21 35,2,5;2257,21 36,0,1;2257,21 36,1,3;2257,21 36,2,5;2257,21 37,0,1;2257,21 37,1,3;2257,21 37,2,5;2257,21 38,0,1;2257,21 38,1,3;2257,21 38,2,5;2257,21 39,0,1;2257,21 39,1,3;2257,21 39,2,5;2257,22 23,0,1;2257,22 23,1,3;2257,22 23,2,5;2257,22 24,0,1;2257,22 24,1,3;2257,22 24,2,5;2257,22 25,0,1;2257,22 25,1,3;2257,22 25,2,5;2257,22 26,0,1;2257,22 26,1,3;2257,22 26,2,5;2257,22 27,0,1;2257,22 27,1,3;2257,22 27,2,5;2257,22 28,0,1;2257,22 28,1,3;2257,22 28,2,5;2257,22 29,0,1;2257,22 29,1,3;2257,22 29,2,5;2257,22 30,0,1;2257,22 30,1,3;2257,22 30,2,5;2257,22 31,0,1;2257,22 31,1,3;2257,22 31,2,5;2257,22 32,0,1;2257,22 32,1,3;2257,22 32,2,5;2257,22 33,0,1;2257,22 33,1,3;2257,22 33,2,5;2257,22 34,0,1;2257,22 34,1,3;2257,22 34,2,5;2257,22 35,0,1;2257,22 35,1,3;2257,22 35,2,5;2257,22 36,0,1;2257,22 36,1,3;2257,22 36,2,5;2257,22 37,0,1;2257,22 37,1,3;2257,22 37,2,5;2257,22 38,0,1;2257,22 38,1,3;2257,22 38,2,5;2257,22 39,0,1;2257,22 39,1,3;2257,22 39,2,5;2257,23 24,0,1;2257,23 24,1,3;2257,23 24,2,5;2257,23 25,0,1;2257,23 25,1,3;2257,23 25,2,5;2257,23 26,0,1;2257,23 26,1,3;2257,23 26,2,5;2257,23 27,0,1;2257,23 27,1,3;2257,23 27,2,5;2257,23 28,0,1;2257,23 28,1,3;2257,23 28,2,5;2257,23 29,0,1;2257,23 29,1,3;2257,23 29,2,5;2257,23 30,0,1;2257,23 30,1,3;2257,23 30,2,5;2257,23 31,0,1;2257,23 31,1,3;2257,23 31,2,5;2257,23 32,0,1;2257,23 32,1,3;2257,23 32,2,5;2257,23 33,0,1;2257,23 33,1,3;2257,23 33,2,5"
        # #"

        # wp = "2257,23 34,0,1;2257,23 34,1,3;2257,23 34,2,5;2257,23 35,0,1;2257,23 35,1,3;2257,23 35,2,5;2257,23 36,0,1;2257,23 36,1,3;2257,23 36,2,5;2257,23 37,0,1;2257,23 37,1,3;2257,23 37,2,5;2257,23 38,0,1;2257,23 38,1,3;2257,23 38,2,5;2257,23 39,0,1;2257,23 39,1,3;2257,23 39,2,5;2257,24 25,0,1;2257,24 25,1,3;2257,24 25,2,5;2257,24 26,0,1;2257,24 26,1,3;2257,24 26,2,5;2257,24 27,0,1;2257,24 27,1,3;2257,24 27,2,5;2257,24 28,0,1;2257,24 28,1,3;2257,24 28,2,5;2257,24 29,0,1;2257,24 29,1,3;2257,24 29,2,5;2257,24 30,0,1;2257,24 30,1,3;2257,24 30,2,5;2257,24 31,0,1;2257,24 31,1,3;2257,24 31,2,5;2257,24 32,0,1;2257,24 32,1,3;2257,24 32,2,5;2257,24 33,0,1;2257,24 33,1,3;2257,24 33,2,5;2257,24 34,0,1;2257,24 34,1,3;2257,24 34,2,5;2257,24 35,0,1;2257,24 35,1,3;2257,24 35,2,5;2257,24 36,0,1;2257,24 36,1,3;2257,24 36,2,5;2257,24 37,0,1;2257,24 37,1,3;2257,24 37,2,5;2257,24 38,0,1;2257,24 38,1,3;2257,24 38,2,5;2257,24 39,0,1;2257,24 39,1,3;2257,24 39,2,5;2257,25 26,0,1;2257,25 26,1,3;2257,25 26,2,5;2257,25 27,0,1;2257,25 27,1,3;2257,25 27,2,5;2257,25 28,0,1;2257,25 28,1,3;2257,25 28,2,5;2257,25 29,0,1;2257,25 29,1,3;2257,25 29,2,5;2257,25 30,0,1;2257,25 30,1,3;2257,25 30,2,5;2257,25 31,0,1;2257,25 31,1,3;2257,25 31,2,5;2257,25 32,0,1;2257,25 32,1,3;2257,25 32,2,5;2257,25 33,0,1;2257,25 33,1,3;2257,25 33,2,5;2257,25 34,0,1;2257,25 34,1,3;2257,25 34,2,5;2257,25 35,0,1;2257,25 35,1,3;2257,25 35,2,5;2257,25 36,0,1;2257,25 36,1,3;2257,25 36,2,5;2257,25 37,0,1;2257,25 37,1,3;2257,25 37,2,5;2257,25 38,0,1;2257,25 38,1,3;2257,25 38,2,5;2257,25 39,0,1;2257,25 39,1,3;2257,25 39,2,5;2257,26 27,0,1;2257,26 27,1,3;2257,26 27,2,5;2257,26 28,0,1;2257,26 28,1,3;2257,26 28,2,5;2257,26 29,0,1;2257,26 29,1,3;2257,26 29,2,5;2257,26 30,0,1;2257,26 30,1,3;2257,26 30,2,5;2257,26 31,0,1;2257,26 31,1,3;2257,26 31,2,5;2257,26 32,0,1;2257,26 32,1,3;2257,26 32,2,5;2257,26 33,0,1;2257,26 33,1,3;2257,26 33,2,5;2257,26 34,0,1;2257,26 34,1,3;2257,26 34,2,5;2257,26 35,0,1;2257,26 35,1,3;2257,26 35,2,5;2257,26 36,0,1;2257,26 36,1,3;2257,26 36,2,5;2257,26 37,0,1;2257,26 37,1,3;2257,26 37,2,5;2257,26 38,0,1;2257,26 38,1,3;2257,26 38,2,5;2257,26 39,0,1;2257,26 39,1,3;2257,26 39,2,5;2257,27 28,0,1;2257,27 28,1,3;2257,27 28,2,5;2257,27 29,0,1;2257,27 29,1,3;2257,27 29,2,5;2257,27 30,0,1;2257,27 30,1,3;2257,27 30,2,5;2257,27 31,0,1;2257,27 31,1,3;2257,27 31,2,5;2257,27 32,0,1;2257,27 32,1,3;2257,27 32,2,5;2257,27 33,0,1;2257,27 33,1,3;2257,27 33,2,5;2257,27 34,0,1;2257,27 34,1,3;2257,27 34,2,5;2257,27 35,0,1;2257,27 35,1,3;2257,27 35,2,5;2257,27 36,0,1;2257,27 36,1,3;2257,27 36,2,5;2257,27 37,0,1;2257,27 37,1,3;2257,27 37,2,5;2257,27 38,0,1;2257,27 38,1,3;2257,27 38,2,5;2257,27 39,0,1;2257,27 39,1,3;2257,27 39,2,5;2257,28 29,0,1;2257,28 29,1,3;2257,28 29,2,5;2257,28 30,0,1;2257,28 30,1,3;2257,28 30,2,5;2257,28 31,0,1;2257,28 31,1,3;2257,28 31,2,5;2257,28 32,0,1;2257,28 32,1,3;2257,28 32,2,5;2257,28 33,0,1;2257,28 33,1,3;2257,28 33,2,5;2257,28 34,0,1;2257,28 34,1,3;2257,28 34,2,5;2257,28 35,0,1;2257,28 35,1,3;2257,28 35,2,5;2257,28 36,0,1;2257,28 36,1,3;2257,28 36,2,5;2257,28 37,0,1;2257,28 37,1,3;2257,28 37,2,5;2257,28 38,0,1;2257,28 38,1,3;2257,28 38,2,5;2257,28 39,0,1;2257,28 39,1,3;2257,28 39,2,5;2257,29 30,0,1;2257,29 30,1,3;2257,29 30,2,5;2257,29 31,0,1;2257,29 31,1,3;2257,29 31,2,5;2257,29 32,0,1;2257,29 32,1,3;2257,29 32,2,5;2257,29 33,0,1;2257,29 33,1,3;2257,29 33,2,5;2257,29 34,0,1;2257,29 34,1,3;2257,29 34,2,5;2257,29 35,0,1;2257,29 35,1,3;2257,29 35,2,5;2257,29 36,0,1;2257,29 36,1,3;2257,29 36,2,5;2257,29 37,0,1;2257,29 37,1,3;2257,29 37,2,5;2257,29 38,0,1;2257,29 38,1,3;2257,29 38,2,5;2257,29 39,0,1;2257,29 39,1,3;2257,29 39,2,5;2257,30 31,0,1;2257,30 31,1,3;2257,30 31,2,5;2257,30 32,0,1;2257,30 32,1,3;2257,30 32,2,5;2257,30 33,0,1;2257,30 33,1,3;2257,30 33,2,5;2257,30 34,0,1;2257,30 34,1,3;2257,30 34,2,5;2257,30 35,0,1;2257,30 35,1,3;2257,30 35,2,5;2257,30 36,0,1;2257,30 36,1,3;2257,30 36,2,5;2257,30 37,0,1;2257,30 37,1,3;2257,30 37,2,5;2257,30 38,0,1;2257,30 38,1,3;2257,30 38,2,5;2257,30 39,0,1;2257,30 39,1,3;2257,30 39,2,5;2257,31 32,0,1;2257,31 32,1,3;2257,31 32,2,5;2257,31 33,0,1;2257,31 33,1,3;2257,31 33,2,5;2257,31 34,0,1;2257,31 34,1,3;2257,31 34,2,5;2257,31 35,0,1;2257,31 35,1,3;2257,31 35,2,5;2257,31 36,0,1;2257,31 36,1,3;2257,31 36,2,5;2257,31 37,0,1;2257,31 37,1,3;2257,31 37,2,5;2257,31 38,0,1;2257,31 38,1,3;2257,31 38,2,5;2257,31 39,0,1;2257,31 39,1,3;2257,31 39,2,5;2257,32 33,0,1;2257,32 33,1,3;2257,32 33,2,5;2257,32 34,0,1;2257,32 34,1,3;2257,32 34,2,5;2257,32 35,0,1;2257,32 35,1,3;2257,32 35,2,5;2257,32 36,0,1;2257,32 36,1,3;2257,32 36,2,5;2257,32 37,0,1;2257,32 37,1,3;2257,32 37,2,5;2257,32 38,0,1;2257,32 38,1,3;2257,32 38,2,5;2257,32 39,0,1;2257,32 39,1,3;2257,32 39,2,5;2257,33 34,0,1;2257,33 34,1,3;2257,33 34,2,5;2257,33 35,0,1;2257,33 35,1,3;2257,33 35,2,5;2257,33 36,0,1;2257,33 36,1,3;2257,33 36,2,5;2257,33 37,0,1;2257,33 37,1,3;2257,33 37,2,5;2257,33 38,0,1;2257,33 38,1,3;2257,33 38,2,5;2257,33 39,0,1;2257,33 39,1,3;2257,33 39,2,5;2257,34 35,0,1;2257,34 35,1,3;2257,34 35,2,5;2257,34 36,0,1;2257,34 36,1,3;2257,34 36,2,5;2257,34 37,0,1;2257,34 37,1,3;2257,34 37,2,5;2257,34 38,0,1;2257,34 38,1,3;2257,34 38,2,5;2257,34 39,0,1;2257,34 39,1,3;2257,34 39,2,5;2257,35 36,0,1;2257,35 36,1,3;2257,35 36,2,5;2257,35 37,0,1;2257,35 37,1,3;2257,35 37,2,5;2257,35 38,0,1;2257,35 38,1,3;2257,35 38,2,5;2257,35 39,0,1;2257,35 39,1,3;2257,35 39,2,5;2257,36 37,0,1;2257,36 37,1,3;2257,36 37,2,5;2257,36 38,0,1;2257,36 38,1,3;2257,36 38,2,5;2257,36 39,0,1;2257,36 39,1,3;2257,36 39,2,5;2257,37 38,0,1;2257,37 38,1,3;2257,37 38,2,5;2257,37 39,0,1;2257,37 39,1,3;2257,37 39,2,5;2257,38 39,0,1;2257,38 39,1,3;2257,38 39,2,5"
        # #"

        # wp = "940,7 32,1,3;940,7 32,2,5;940,7 33,0,1;940,7 33,1,3;940,7 33,2,5;940,7 34,0,1;940,7 34,1,3;940,7 34,2,5;940,7 35,0,1;940,7 35,1,3;940,7 35,2,5;940,7 36,0,1;940,7 36,1,3;940,7 36,2,5;940,7 37,0,1;940,7 37,1,3;940,7 37,2,5;940,7 38,0,1;940,7 38,1,3;940,7 38,2,5;940,7 39,0,1;940,7 39,1,3;940,7 39,2,5;940,7 40,0,1;940,7 40,1,3;940,7 40,2,5;940,7 41,0,1;940,7 41,1,3;940,7 41,2,5;940,7 42,0,1;940,7 42,1,3;940,7 42,2,5;940,7 43,0,1;940,7 43,1,3;940,7 43,2,5;940,7 44,0,1;940,7 44,1,3;940,7 44,2,5;940,7 45,0,1;940,7 45,1,3;940,7 45,2,5;940,8 9,0,1;940,8 9,1,3;940,8 9,2,5;940,8 10,0,1;940,8 10,1,3;940,8 10,2,5;940,8 11,0,1;940,8 11,1,3;940,8 11,2,5;940,8 12,0,1;940,8 12,1,3;940,8 12,2,5;940,8 13,0,1;940,8 13,1,3;940,8 13,2,5;940,8 14,0,1;940,8 14,1,3;940,8 14,2,5;940,8 15,0,1;940,8 15,1,3;940,8 15,2,5;940,8 16,0,1;940,8 16,1,3;940,8 16,2,5;940,8 17,0,1;940,8 17,1,3;940,8 17,2,5;940,8 18,0,1;940,8 18,1,3;940,8 18,2,5;940,8 19,0,1;940,8 19,1,3;940,8 19,2,5;940,8 20,0,1;940,8 20,1,3;940,8 20,2,5;940,8 21,0,1;940,8 21,1,3;940,8 21,2,5;940,8 22,0,1;940,8 22,1,3;940,8 22,2,5;940,8 23,0,1;940,8 23,1,3;940,8 23,2,5;940,8 24,0,1;940,8 24,1,3;940,8 24,2,5;940,8 25,0,1;940,8 25,1,3;940,8 25,2,5;940,8 26,0,1;940,8 26,1,3;940,8 26,2,5;940,8 27,0,1;940,8 27,1,3;940,8 27,2,5;940,8 28,0,1;940,8 28,1,3;940,8 28,2,5;940,8 29,0,1;940,8 29,1,3;940,8 29,2,5;940,8 30,0,1;940,8 30,1,3;940,8 30,2,5;940,8 31,0,1;940,8 31,1,3;940,8 31,2,5;940,8 32,0,1;940,8 32,1,3;940,8 32,2,5;940,8 33,0,1;940,8 33,1,3;940,8 33,2,5;940,8 34,0,1;940,8 34,1,3;940,8 34,2,5;940,8 35,0,1;940,8 35,1,3;940,8 35,2,5;940,8 36,0,1;940,8 36,1,3;940,8 36,2,5;940,8 37,0,1;940,8 37,1,3;940,8 37,2,5;940,8 38,0,1;940,8 38,1,3;940,8 38,2,5;940,8 39,0,1;940,8 39,1,3;940,8 39,2,5;940,8 40,0,1;940,8 40,1,3;940,8 40,2,5;940,8 41,0,1;940,8 41,1,3;940,8 41,2,5;940,8 42,0,1;940,8 42,1,3;940,8 42,2,5;940,8 43,0,1;940,8 43,1,3;940,8 43,2,5;940,8 44,0,1;940,8 44,1,3;940,8 44,2,5;940,8 45,0,1;940,8 45,1,3;940,8 45,2,5;940,9 10,0,1;940,9 10,1,3;940,9 10,2,5;940,9 11,0,1;940,9 11,1,3;940,9 11,2,5;940,9 12,0,1;940,9 12,1,3;940,9 12,2,5;940,9 13,0,1;940,9 13,1,3;940,9 13,2,5;940,9 14,0,1;940,9 14,1,3;940,9 14,2,5;940,9 15,0,1;940,9 15,1,3;940,9 15,2,5;940,9 16,0,1;940,9 16,1,3;940,9 16,2,5;940,9 17,0,1;940,9 17,1,3;940,9 17,2,5;940,9 18,0,1;940,9 18,1,3;940,9 18,2,5;940,9 19,0,1;940,9 19,1,3;940,9 19,2,5;940,9 20,0,1;940,9 20,1,3;940,9 20,2,5;940,9 21,0,1;940,9 21,1,3;940,9 21,2,5;940,9 22,0,1;940,9 22,1,3;940,9 22,2,5;940,9 23,0,1;940,9 23,1,3;940,9 23,2,5;940,9 24,0,1;940,9 24,1,3;940,9 24,2,5;940,9 25,0,1;940,9 25,1,3;940,9 25,2,5;940,9 26,0,1;940,9 26,1,3;940,9 26,2,5;940,9 27,0,1;940,9 27,1,3;940,9 27,2,5;940,9 28,0,1;940,9 28,1,3;940,9 28,2,5;940,9 29,0,1;940,9 29,1,3;940,9 29,2,5;940,9 30,0,1;940,9 30,1,3;940,9 30,2,5;940,9 31,0,1;940,9 31,1,3;940,9 31,2,5;940,9 32,0,1;940,9 32,1,3;940,9 32,2,5;940,9 33,0,1;940,9 33,1,3;940,9 33,2,5;940,9 34,0,1;940,9 34,1,3;940,9 34,2,5;940,9 35,0,1;940,9 35,1,3;940,9 35,2,5;940,9 36,0,1;940,9 36,1,3;940,9 36,2,5;940,9 37,0,1;940,9 37,1,3;940,9 37,2,5;940,9 38,0,1;940,9 38,1,3;940,9 38,2,5;940,9 39,0,1;940,9 39,1,3;940,9 39,2,5;940,9 40,0,1;940,9 40,1,3;940,9 40,2,5;940,9 41,0,1;940,9 41,1,3;940,9 41,2,5;940,9 42,0,1;940,9 42,1,3;940,9 42,2,5;940,9 43,0,1;940,9 43,1,3;940,9 43,2,5;940,9 44,0,1;940,9 44,1,3;940,9 44,2,5;940,9 45,0,1;940,9 45,1,3;940,9 45,2,5;940,10 11,0,1;940,10 11,1,3;940,10 11,2,5;940,10 12,0,1;940,10 12,1,3;940,10 12,2,5;940,10 13,0,1;940,10 13,1,3;940,10 13,2,5;940,10 14,0,1;940,10 14,1,3;940,10 14,2,5;940,10 15,0,1;940,10 15,1,3;940,10 15,2,5;940,10 16,0,1;940,10 16,1,3;940,10 16,2,5;940,10 17,0,1;940,10 17,1,3;940,10 17,2,5;940,10 18,0,1;940,10 18,1,3;940,10 18,2,5;940,10 19,0,1;940,10 19,1,3;940,10 19,2,5;940,10 20,0,1;940,10 20,1,3;940,10 20,2,5;940,10 21,0,1;940,10 21,1,3;940,10 21,2,5;940,10 22,0,1;940,10 22,1,3;940,10 22,2,5;940,10 23,0,1;940,10 23,1,3;940,10 23,2,5;940,10 24,0,1;940,10 24,1,3;940,10 24,2,5;940,10 25,0,1;940,10 25,1,3;940,10 25,2,5;940,10 26,0,1;940,10 26,1,3;940,10 26,2,5;940,10 27,0,1;940,10 27,1,3;940,10 27,2,5;940,10 28,0,1;940,10 28,1,3;940,10 28,2,5;940,10 29,0,1;940,10 29,1,3;940,10 29,2,5;940,10 30,0,1;940,10 30,1,3;940,10 30,2,5;940,10 31,0,1;940,10 31,1,3;940,10 31,2,5;940,10 32,0,1;940,10 32,1,3;940,10 32,2,5;940,10 33,0,1;940,10 33,1,3;940,10 33,2,5;940,10 34,0,1;940,10 34,1,3;940,10 34,2,5;940,10 35,0,1;940,10 35,1,3;940,10 35,2,5;940,10 36,0,1;940,10 36,1,3;940,10 36,2,5;940,10 37,0,1;940,10 37,1,3;940,10 37,2,5;940,10 38,0,1;940,10 38,1,3;940,10 38,2,5;940,10 39,0,1;940,10 39,1,3;940,10 39,2,5;940,10 40,0,1;940,10 40,1,3;940,10 40,2,5;940,10 41,0,1;940,10 41,1,3;940,10 41,2,5;940,10 42,0,1;940,10 42,1,3;940,10 42,2,5;940,10 43,0,1;940,10 43,1,3;940,10 43,2,5;940,10 44,0,1;940,10 44,1,3;940,10 44,2,5;940,10 45,0,1;940,10 45,1,3;940,10 45,2,5;940,11 12,0,1;940,11 12,1,3;940,11 12,2,5;940,11 13,0,1;940,11 13,1,3;940,11 13,2,5;940,11 14,0,1;940,11 14,1,3;940,11 14,2,5;940,11 15,0,1;940,11 15,1,3;940,11 15,2,5;940,11 16,0,1;940,11 16,1,3;940,11 16,2,5;940,11 17,0,1;940,11 17,1,3;940,11 17,2,5;940,11 18,0,1;940,11 18,1,3;940,11 18,2,5;940,11 19,0,1;940,11 19,1,3;940,11 19,2,5;940,11 20,0,1;940,11 20,1,3;940,11 20,2,5;940,11 21,0,1;940,11 21,1,3;940,11 21,2,5;940,11 22,0,1;940,11 22,1,3;940,11 22,2,5;940,11 23,0,1;940,11 23,1,3;940,11 23,2,5;940,11 24,0,1;940,11 24,1,3;940,11 24,2,5;940,11 25,0,1;940,11 25,1,3;940,11 25,2,5;940,11 26,0,1;940,11 26,1,3;940,11 26,2,5;940,11 27,0,1;940,11 27,1,3;940,11 27,2,5;940,11 28,0,1;940,11 28,1,3;940,11 28,2,5;940,11 29,0,1;940,11 29,1,3;940,11 29,2,5;940,11 30,0,1;940,11 30,1,3;940,11 30,2,5;940,11 31,0,1;940,11 31,1,3;940,11 31,2,5;940,11 32,0,1;940,11 32,1,3;940,11 32,2,5;940,11 33,0,1;940,11 33,1,3;940,11 33,2,5;940,11 34,0,1;940,11 34,1,3;940,11 34,2,5;940,11 35,0,1;940,11 35,1,3;940,11 35,2,5;940,11 36,0,1;940,11 36,1,3;940,11 36,2,5;940,11 37,0,1;940,11 37,1,3;940,11 37,2,5;940,11 38,0,1;940,11 38,1,3;940,11 38,2,5;940,11 39,0,1;940,11 39,1,3;940,11 39,2,5;940,11 40,0,1;940,11 40,1,3;940,11 40,2,5;940,11 41,0,1;940,11 41,1,3;940,11 41,2,5;940,11 42,0,1;940,11 42,1,3;940,11 42,2,5;940,11 43,0,1;940,11 43,1,3;940,11 43,2,5;940,11 44,0,1;940,11 44,1,3;940,11 44,2,5;940,11 45,0,1;940,11 45,1,3;940,11 45,2,5;940,12 13,0,1;940,12 13,1,3;940,12 13,2,5;940,12 14,0,1;940,12 14,1,3;940,12 14,2,5;940,12 15,0,1;940,12 15,1,3;940,12 15,2,5;940,12 16,0,1;940,12 16,1,3;940,12 16,2,5;940,12 17,0,1;940,12 17,1,3;940,12 17,2,5;940,12 18,0,1;940,12 18,1,3;940,12 18,2,5;940,12 19,0,1;940,12 19,1,3;940,12 19,2,5;940,12 20,0,1;940,12 20,1,3;940,12 20,2,5;940,12 21,0,1;940,12 21,1,3;940,12 21,2,5;940,12 22,0,1;940,12 22,1,3;940,12 22,2,5;940,12 23,0,1;940,12 23,1,3;940,12 23,2,5;940,12 24,0,1;940,12 24,1,3;940,12 24,2,5;940,12 25,0,1;940,12 25,1,3;940,12 25,2,5;940,12 26,0,1;940,12 26,1,3;940,12 26,2,5;940,12 27,0,1;940,12 27,1,3;940,12 27,2,5;940,12 28,0,1;940,12 28,1,3;940,12 28,2,5;940,12 29,0,1;940,12 29,1,3;940,12 29,2,5;940,12 30,0,1;940,12 30,1,3;940,12 30,2,5;940,12 31,0,1;940,12 31,1,3;940,12 31,2,5;940,12 32,0,1;940,12 32,1,3;940,12 32,2,5;940,12 33,0,1;940,12 33,1,3;940,12 33,2,5;940,12 34,0,1;940,12 34,1,3;940,12 34,2,5;940,12 35,0,1;940,12 35,1,3;940,12 35,2,5;940,12 36,0,1;940,12 36,1,3;940,12 36,2,5;940,12 37,0,1;940,12 37,1,3;940,12 37,2,5;940,12 38,0,1;940,12 38,1,3;940,12 38,2,5;940,12 39,0,1;940,12 39,1,3;940,12 39,2,5;940,12 40,0,1;940,12 40,1,3;940,12 40,2,5;940,12 41,0,1;940,12 41,1,3;940,12 41,2,5;940,12 42,0,1;940,12 42,1,3;940,12 42,2,5;940,12 43,0,1;940,12 43,1,3;940,12 43,2,5;940,12 44,0,1;940,12 44,1,3;940,12 44,2,5;940,12 45,0,1;940,12 45,1,3;940,12 45,2,5;940,13 14,0,1;940,13 14,1,3;940,13 14,2,5;940,13 15,0,1;940,13 15,1,3;940,13 15,2,5;940,13 16,0,1;940,13 16,1,3;940,13 16,2,5;940,13 17,0,1;940,13 17,1,3;940,13 17,2,5;940,13 18,0,1;940,13 18,1,3;940,13 18,2,5;940,13 19,0,1;940,13 19,1,3;940,13 19,2,5;940,13 20,0,1;940,13 20,1,3;940,13 20,2,5;940,13 21,0,1;940,13 21,1,3;940,13 21,2,5;940,13 22,0,1;940,13 22,1,3;940,13 22,2,5;940,13 23,0,1;940,13 23,1,3;940,13 23,2,5;940,13 24,0,1;940,13 24,1,3;940,13 24,2,5;940,13 25,0,1;940,13 25,1,3;940,13 25,2,5;940,13 26,0,1;940,13 26,1,3;940,13 26,2,5;940,13 27,0,1;940,13 27,1,3;940,13 27,2,5;940,13 28,0,1;940,13 28,1,3;940,13 28,2,5;940,13 29,0,1;940,13 29,1,3;940,13 29,2,5;940,13 30,0,1;940,13 30,1,3;940,13 30,2,5;940,13 31,0,1;940,13 31,1,3;940,13 31,2,5;940,13 32,0,1;940,13 32,1,3;940,13 32,2,5;940,13 33,0,1;940,13 33,1,3;940,13 33,2,5;940,13 34,0,1;940,13 34,1,3;940,13 34,2,5;940,13 35,0,1;940,13 35,1,3;940,13 35,2,5;940,13 36,0,1;940,13 36,1,3;940,13 36,2,5;940,13 37,0,1;940,13 37,1,3;940,13 37,2,5;940,13 38,0,1;940,13 38,1,3;940,13 38,2,5;940,13 39,0,1;940,13 39,1,3;940,13 39,2,5;940,13 40,0,1;940,13 40,1,3;940,13 40,2,5;940,13 41,0,1;940,13 41,1,3;940,13 41,2,5;940,13 42,0,1;940,13 42,1,3;940,13 42,2,5;940,13 43,0,1;940,13 43,1,3;940,13 43,2,5;940,13 44,0,1;940,13 44,1,3;940,13 44,2,5;940,13 45,0,1;940,13 45,1,3;940,13 45,2,5;940,14 15,0,1;940,14 15,1,3;940,14 15,2,5;940,14 16,0,1;940,14 16,1,3;940,14 16,2,5;940,14 17,0,1;940,14 17,1,3;940,14 17,2,5;940,14 18,0,1;940,14 18,1,3;940,14 18,2,5;940,14 19,0,1;940,14 19,1,3;940,14 19,2,5;940,14 20,0,1;940,14 20,1,3;940,14 20,2,5;940,14 21,0,1;940,14 21,1,3;940,14 21,2,5;940,14 22,0,1;940,14 22,1,3;940,14 22,2,5;940,14 23,0,1;940,14 23,1,3;940,14 23,2,5;940,14 24,0,1;940,14 24,1,3;940,14 24,2,5;940,14 25,0,1;940,14 25,1,3;940,14 25,2,5;940,14 26,0,1;940,14 26,1,3;940,14 26,2,5;940,14 27,0,1;940,14 27,1,3;940,14 27,2,5;940,14 28,0,1;940,14 28,1,3;940,14 28,2,5;940,14 29,0,1;940,14 29,1,3;940,14 29,2,5;940,14 30,0,1;940,14 30,1,3;940,14 30,2,5;940,14 31,0,1;940,14 31,1,3;940,14 31,2,5;940,14 32,0,1;940,14 32,1,3;940,14 32,2,5;940,14 33,0,1;940,14 33,1,3;940,14 33,2,5;940,14 34,0,1;940,14 34,1,3;940,14 34,2,5;940,14 35,0,1;940,14 35,1,3;940,14 35,2,5;940,14 36,0,1;940,14 36,1,3;940,14 36,2,5;940,14 37,0,1;940,14 37,1,3;940,14 37,2,5;940,14 38,0,1;940,14 38,1,3;940,14 38,2,5;940,14 39,0,1;940,14 39,1,3;940,14 39,2,5;940,14 40,0,1;940,14 40,1,3;940,14 40,2,5;940,14 41,0,1;940,14 41,1,3;940,14 41,2,5;940,14 42,0,1;940,14 42,1,3;940,14 42,2,5;940,14 43,0,1;940,14 43,1,3;940,14 43,2,5;940,14 44,0,1;940,14 44,1,3;940,14 44,2,5;940,14 45,0,1;940,14 45,1,3;940,14 45,2,5;940,15 16,0,1;940,15 16,1,3;940,15 16,2,5;940,15 17,0,1;940,15 17,1,3;940,15 17,2,5;940,15 18,0,1;940,15 18,1,3;940,15 18,2,5;940,15 19,0,1;940,15 19,1,3;940,15 19,2,5;940,15 20,0,1;940,15 20,1,3;940,15 20,2,5;940,15 21,0,1;940,15 21,1,3;940,15 21,2,5;940,15 22,0,1;940,15 22,1,3;940,15 22,2,5;940,15 23,0,1;940,15 23,1,3;940,15 23,2,5;940,15 24,0,1;940,15 24,1,3;940,15 24,2,5;940,15 25,0,1;940,15 25,1,3;940,15 25,2,5;940,15 26,0,1;940,15 26,1,3;940,15 26,2,5;940,15 27,0,1;940,15 27,1,3;940,15 27,2,5;940,15 28,0,1;940,15 28,1,3;940,15 28,2,5;940,15 29,0,1;940,15 29,1,3;940,15 29,2,5;940,15 30,0,1;940,15 30,1,3;940,15 30,2,5;940,15 31,0,1;940,15 31,1,3;940,15 31,2,5;940,15 32,0,1;940,15 32,1,3;940,15 32,2,5;940,15 33,0,1;940,15 33,1,3;940,15 33,2,5;940,15 34,0,1;940,15 34,1,3;940,15 34,2,5;940,15 35,0,1;940,15 35,1,3;940,15 35,2,5;940,15 36,0,1;940,15 36,1,3;940,15 36,2,5;940,15 37,0,1;940,15 37,1,3;940,15 37,2,5;940,15 38,0,1;940,15 38,1,3;940,15 38,2,5;940,15 39,0,1;940,15 39,1,3;940,15 39,2,5;940,15 40,0,1;940,15 40,1,3;940,15 40,2,5;940,15 41,0,1;940,15 41,1,3;940,15 41,2,5;940,15 42,0,1;940,15 42,1,3;940,15 42,2,5;940,15 43,0,1;940,15 43,1,3;940,15 43,2,5;940,15 44,0,1;940,15 44,1,3;940,15 44,2,5;940,15 45,0,1;940,15 45,1,3;940,15 45,2,5;940,16 17,0,1;940,16 17,1,3;940,16 17,2,5;940,16 18,0,1;940,16 18,1,3;940,16 18,2,5;940,16 19,0,1;940,16 19,1,3;940,16 19,2,5;940,16 20,0,1;940,16 20,1,3;940,16 20,2,5;940,16 21,0,1;940,16 21,1,3;940,16 21,2,5;940,16 22,0,1;940,16 22,1,3;940,16 22,2,5;940,16 23,0,1;940,16 23,1,3;940,16 23,2,5;940,16 24,0,1;940,16 24,1,3;940,16 24,2,5;940,16 25,0,1;940,16 25,1,3;940,16 25,2,5;940,16 26,0,1;940,16 26,1,3;940,16 26,2,5;940,16 27,0,1;940,16 27,1,3;940,16 27,2,5;940,16 28,0,1;940,16 28,1,3;940,16 28,2,5;940,16 29,0,1;940,16 29,1,3;940,16 29,2,5;940,16 30,0,1;940,16 30,1,3;940,16 30,2,5;940,16 31,0,1;940,16 31,1,3;940,16 31,2,5;940,16 32,0,1;940,16 32,1,3;940,16 32,2,5;940,16 33,0,1;940,16 33,1,3;940,16 33,2,5;940,16 34,0,1;940,16 34,1,3;940,16 34,2,5;940,16 35,0,1;940,16 35,1,3;940,16 35,2,5;940,16 36,0,1;940,16 36,1,3;940,16 36,2,5;940,16 37,0,1;940,16 37,1,3;940,16 37,2,5;940,16 38,0,1;940,16 38,1,3;940,16 38,2,5;940,16 39,0,1;940,16 39,1,3;940,16 39,2,5;940,16 40,0,1;940,16 40,1,3;940,16 40,2,5;940,16 41,0,1;940,16 41,1,3;940,16 41,2,5;940,16 42,0,1;940,16 42,1,3;940,16 42,2,5;940,16 43,0,1;940,16 43,1,3;940,16 43,2,5;940,16 44,0,1;940,16 44,1,3;940,16 44,2,5;940,16 45,0,1;940,16 45,1,3;940,16 45,2,5;940,17 18,0,1;940,17 18,1,3;940,17 18,2,5;940,17 19,0,1;940,17 19,1,3;940,17 19,2,5;940,17 20,0,1;940,17 20,1,3;940,17 20,2,5;940,17 21,0,1;940,17 21,1,3;940,17 21,2,5;940,17 22,0,1;940,17 22,1,3;940,17 22,2,5;940,17 23,0,1;940,17 23,1,3;940,17 23,2,5;940,17 24,0,1;940,17 24,1,3;940,17 24,2,5;940,17 25,0,1;940,17 25,1,3;940,17 25,2,5;940,17 26,0,1;940,17 26,1,3;940,17 26,2,5;940,17 27,0,1;940,17 27,1,3;940,17 27,2,5;940,17 28,0,1;940,17 28,1,3;940,17 28,2,5;940,17 29,0,1;940,17 29,1,3;940,17 29,2,5;940,17 30,0,1;940,17 30,1,3;940,17 30,2,5;940,17 31,0,1;940,17 31,1,3;940,17 31,2,5;940,17 32,0,1;940,17 32,1,3;940,17 32,2,5;940,17 33,0,1;940,17 33,1,3;940,17 33,2,5;940,17 34,0,1;940,17 34,1,3;940,17 34,2,5;940,17 35,0,1;940,17 35,1,3;940,17 35,2,5;940,17 36,0,1;940,17 36,1,3;940,17 36,2,5;940,17 37,0,1;940,17 37,1,3;940,17 37,2,5;940,17 38,0,1;940,17 38,1,3;940,17 38,2,5;940,17 39,0,1;940,17 39,1,3;940,17 39,2,5;940,17 40,0,1;940,17 40,1,3;940,17 40,2,5"
        # #"

        # Read archive workers from stdin

        # workpackage = "2405,8 6 4,3"
        # workpackage += ";" + "2405,8 6 4,1"
        # workpackage += ";" + "2405,8 6 4,2"
        # workpackage += ";" + "2405,8 6 5,3"
        # workpackage += ";" + "2405,8 6 3,3"
        # workpackage += ";" + "2405,8 6 2,3"
        # workpackage = "2405,8 6 4,2"
        # workpackage = "0,39 40,0;0,39 40,1;0,39 40,2"

        results = run_jobs(moldb, tordb, wp, dump_sdf="test.sdf", debug=args.debug)

        print()
        print(results[0])

        quit()

    return

if __name__ == '__main__':
    main()

