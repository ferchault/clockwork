#

import json
import copy
import itertools
import multiprocessing
import os
import sys
import time
from functools import partial

import numpy as np
from numpy import linalg
from tqdm import tqdm

import clockwork
import merge
import similarity_fchl19 as sim
from chemhelp import cheminfo
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalForceFields
from rdkit.Chem import rdmolfiles

from communication import rediscomm

import joblib

# Set local cache
cachedir = '.pycache'
memory = joblib.Memory(cachedir, verbose=0)

DEFAULT_DECIMALS = 5
# DEFAULT_DECIMALS = 12

def correct_userpath(filepath):
    return os.path.expanduser(filepath)


def get_forcefield(molobj):

    ffprop = ChemicalForceFields.MMFFGetMoleculeProperties(molobj)
    forcefield = ChemicalForceFields.MMFFGetMoleculeForceField(molobj, ffprop) # 0.01 overhead

    return ffprop, forcefield


def run_forcefield(ff, steps, energy=1e-2, force=1e-3):
    """
    """

    try:
        status = ff.Minimize(maxIts=steps, energyTol=energy, forceTol=force)
    except RuntimeError:
        return 1

    return status


def run_forcefield_prime(ff, steps, energy=1e-2, force=1e-3):

    try:
        status = ff.Minimize(maxIts=steps, energyTol=energy, forceTol=force)
    except RuntimeError:
        return 1

    return status


@memory.cache
def generate_torsion_combinations(total_torsions, n_tor):

    combinations = clockwork.generate_torsion_combinations(total_torsions, n_tor)
    combinations = list(combinations)

    return combinations


def generate_torsions(total_torsions,
    min_cost=0, max_cost=15, prefix="0"):

    cost_input, cost_cost = clockwork.generate_costlist(total_torsions=total_torsions)

    for (n_tor, resolution), cost in zip(cost_input[min_cost:max_cost], cost_cost[min_cost:max_cost]):

        combinations = generate_torsion_combinations(total_torsions, n_tor)

        for combination in combinations:

            jobstr = prefix + ","
            torstr = " ".join([str(x) for x in combination])
            resstr = str(resolution)
            jobstr += torstr + "," + resstr
            print(jobstr)

    return


def generate_torsions_specific(total_torsions, n_tor, resolution, prefix="0"):

    sep = ","

    combinations = generate_torsion_combinations(total_torsions, n_tor)

    for combination in combinations:

        jobstr = prefix + sep
        torstr = " ".join([str(x) for x in combination])
        resstr = str(resolution)
        jobstr += torstr + sep + resstr
        print(jobstr)

    return


def generate_jobs(molobjs, args, tordb=None,
    min_cost=0, max_cost=15):

    # TODO Group by cost?

    combos = args.jobcombos

    n_molecules = len(molobjs)

    if tordb is None:
        tordb = [cheminfo.get_torsions(molobj) for molobj in molobjs]


    for i in range(n_molecules)[:500]:

        molobj = molobjs[i]
        torsions = tordb[i]

        total_torsions = len(torsions)

        prefix = str(i)

        if combos is None:
            generate_torsions(total_torsions, prefix=prefix, min_cost=min_cost, max_cost=max_cost)

        else:

            for combo in combos:
                combo = combo.split(",")
                combo = [int(x) for x in combo]
                generate_torsions_specific(total_torsions, combo[0], combo[1], prefix=prefix)


    # quit()
    #
    # cost_input, cost_cost = clockwork.generate_costlist(total_torsions=total_torsions)
    #
    # for (n_tor, resolution), cost in zip(cost_input[min_cost:max_cost], cost_cost[min_cost:max_cost]):
    #
    #     combinations = clockwork.generate_torsion_combinations(total_torsions, n_tor)
    #
    #     for combination in combinations:
    #
    #         # torsions = [tordb[x] for x in combination]
    #
    #         jobstr = prefix + ","
    #         torstr = " ".join([str(x) for x in combination])
    #         resstr = str(resolution)
    #         jobstr += torstr + "," + resstr
    #         print(jobstr)
    #
    # quit()
    return


def converge_clockwork(molobj, tordb, max_cost=2):
    """
    molobj
    torsions_idx
    resolution

    """

    atoms, xyz = cheminfo.molobj_to_xyz(molobj)

    total_torsions = len(tordb)
    print("total torsions", total_torsions)

    # TODO Cache this
    cost_input, cost_cost = clockwork.generate_costlist(total_torsions=total_torsions)

    # TODO cost_cost and costfunc

    offset = 6
    max_cost = 1
    offset = 1
    max_cost = 7
    # offset = 7
    # max_cost = 1

    for (n_tor, resolution), cost in zip(cost_input[offset:offset+max_cost], cost_cost[offset:offset+max_cost]):

        start = time.time()

        # Iterate over torsion combinations
        combinations = clockwork.generate_torsion_combinations(total_torsions, n_tor)

        cost_result_energies = []
        cost_result_coordinates = []

        C = 0

        for combination in combinations:

            # TODO Move this to function

            com_start = time.time()

            torsions = [tordb[i] for i in combination]

            result_energies, result_coordinates = get_clockwork_conformations(molobj, torsions, resolution)
            n_results = len(result_energies)
            result_cost = [cost]*n_results

            com_end = time.time()

            # print("new confs", len(result_energies), "{:6.2f}".format(com_end-com_start))

            # Merge
            if len(cost_result_energies) == 0:

                cost_result_energies += list(result_energies)
                cost_result_coordinates += list(result_coordinates)
                continue

            else:

                start_merge = time.time()

                # TODO Move this to function

                continue

                idxs = merge.merge_asymmetric(atoms,
                    result_energies,
                    cost_result_energies,
                    result_coordinates,
                    cost_result_coordinates, decimals=2, debug=True)

                for i, idx in enumerate(idxs):

                    C += 1

                    if len(idx) == 0:
                        cost_result_energies.append(result_energies[i])
                        cost_result_coordinates.append(result_coordinates[i])

                end_merge = time.time()

            print("total confs", len(cost_result_energies), "{:10.2f}".format(end_merge-start_merge))
            continue

        end = time.time()

        print("conv", n_tor, resolution, cost, len(cost_result_energies), "tot: {:5.2f}".format(end-start), "per sec: {:5.2f}".format(cost/(end-start)))

    quit()

    return



def get_clockwork_conformations(molobj, torsions, resolution,
    atoms=None,
    debug=False,
    timings=False):
    """

    Get all conformation for specific cost
    cost defined from torsions and resolution

    """

    n_torsions = len(torsions)

    if atoms is None:
        atoms, xyz = cheminfo.molobj_to_xyz(molobj, atom_type="int")
        del xyz


    combinations = clockwork.generate_clockwork_combinations(resolution, n_torsions)

    # Collect energies and coordinates
    end_energies = []
    end_coordinates = []
    end_representations = []

    first = True

    for resolutions in combinations:

        time_start = time.time()

        # Get all conformations
        c_energies, c_coordinates, c_states = get_conformations(molobj, torsions, resolutions)

        N = len(c_energies)

        # Filter unconverged
        success = np.argwhere(c_states == 0)
        success = success.flatten()
        c_energies = c_energies[success]
        c_coordinates = c_coordinates[success]

        N2 = len(c_energies)

        # Calculate representations
        c_representations = [sim.get_representation(atoms, coordinates) for coordinates in c_coordinates]
        c_representations = np.asarray(c_representations)

        # Clean all new conformers for energies and similarity
        idxs = clean_representations(atoms, c_energies, c_representations)

        c_energies = c_energies[idxs]
        c_coordinates = c_coordinates[idxs]
        c_representations = c_representations[idxs]

        if first:
            first = False
            end_energies += list(c_energies)
            end_coordinates += list(c_coordinates)
            end_representations += list(c_representations)
            continue

        # Asymmetrically add new conformers
        idxs = merge.merge_asymmetric(atoms,
            c_energies,
            end_energies,
            c_representations,
            end_representations)

        # Add new unique conformation to return collection
        for i, idx in enumerate(idxs):

            # if conformation already exists, continue
            if len(idx) > 0: continue

            # Add new unique conformation to collection
            end_energies.append(c_energies[i])
            end_coordinates.append(c_coordinates[i])
            end_representations.append(c_representations[i])


        time_end = time.time()

        if timings:
            timing = time_end - time_start
            print("res time {:8.2f} cnf/sec - {:8.2f} tot sec".format(N/timing, timing))

        continue

    return end_energies, end_coordinates


def clean_representations(atoms, energies, representations):
    """
    """
    N = len(energies)

    # Keep index for only unique
    # idxs = merge.merge_asymmetric(atoms,
    #     energies,
    #     energies,
    #     representations,
    #     representations)

    idxs = merge.merge(atoms,
        energies,
        representations)

    # Here all cost is the same, so just take the first conformer
    # idxs = [idx[0] for idx in idxs]
    # idxs = np.unique(idxs)

    return idxs


def clean_conformers(atoms, energies, coordinates, states=None):

    # Total count
    N = len(energies)

    if states is not None:

        # Keep only converged states
        success = np.argwhere(states == 0)
        success = success.flatten()

        # Only looked at converged states, discard rest
        energies = energies[success]
        coordinates = coordinates[success]

    # TODO what about failed states?
    # TODO Check high energies

    # TODO change to asymetric merge (cleaner code)

    # Keep index for only unique
    idxs = merge.merge(atoms, energies, coordinates)

    # Here all cost is the same, so just take the first conformer
    idxs = [idx[0] for idx in idxs]

    return idxs


def get_conformations(molobj, torsions, resolutions):

    molobj = copy.deepcopy(molobj)

    n_torsions = len(torsions)

    # init energy
    energies = []
    states = []
    coordinates = []

    # no constraints
    ffprop, forcefield = get_forcefield(molobj)

    # Forcefield generation failed
    if forcefield is None:
        return [], [], []

    # Get conformer and origin
    conformer = molobj.GetConformer()
    origin = conformer.GetPositions()

    # Origin angle
    origin_angles = []

    # HACK rdkit requires int type for index
    torsions = [[int(y) for y in x] for x in torsions]

    for idxs in torsions:
        angle = Chem.rdMolTransforms.GetDihedralDeg(conformer, *idxs)
        origin_angles.append(angle)

    # Get resolution angles
    angle_iterator = clockwork.generate_angles(resolutions, n_torsions)

    for angle in angle_iterator:

        # reset coordinates
        set_coordinates(conformer, origin)

        # Minimze with torsion angle constraint
        energy, pos, status = calculate_forcefield(molobj, conformer, torsions, origin_angles, angle,
                ffprop=ffprop,
                ff=forcefield)

        # collect
        energies += [energy]
        coordinates += [pos]
        states += [status]

    return np.asarray(energies), np.asarray(coordinates), np.asarray(states)


def get_energy(molobj):

    ffprop, ff = get_forcefield(molobj)

    # Get current energy
    energy = ff.CalcEnergy()

    return energy


def get_energies(molobj, coordinates,
    ffprop=None,
    ff=None):

    if ffprop is None or ff is None:
        ffprop, ff = get_forcefield(molobj)

    # Get conformer and origin
    conformer = molobj.GetConformer()

    for coordinate in coordinates:
        set_coordinates(conformer, coordinate)

    # Get current energy
    energy = ff.CalcEnergy()

    return


def get_sdfcontent(sdffile, rtn_atoms=False):

    coordinates = []
    energies = []

    reader = cheminfo.read_sdffile(sdffile)
    molobjs = [molobj for molobj in reader]
    atoms = ""

    for molobj in molobjs:
        atoms, coordinate = cheminfo.molobj_to_xyz(molobj)
        energy = get_energy(molobj)

        coordinates.append(coordinate)
        energies.append(energy)

    if rtn_atoms:
        return molobjs[0], atoms, energies, coordinates

    return energies, coordinates


def calculate_forcefield(molobj, conformer, torsions, origin_angles, delta_angles,
    ffprop=None,
    ff=None,
    delta=10**-7,
    coord_decimals=6):
    """


    Disclaimer: lots of hacks, sorry. Let me know if you have an alternative.

    Note: There is a artificat where if delta < 10**-16 the FF will find a
    *extremely* local minima with very high energy (un-physical)the FF will
    find a *extremely* local minima with very high energy (un-physical).
    Setting delta to 10**-6 (numerical noise) should fix this.

    Note: rdkit forcefield restrained optimization will optimized to a *very*
    local and very unphysical minima which the global optimizer cannot get out
    from. Truncating the digits of the coordinates to six is a crude but
    effective way to slight move the the molecule out of this in a reproducable
    way.


    """

    if ffprop is None or ff is None:
        ffprop, ff = get_forcefield(molobj)

    sdfstr = cheminfo.molobj_to_sdfstr(molobj)
    molobj_prime, status = cheminfo.sdfstr_to_molobj(sdfstr)
    conformer_prime = molobj_prime.GetConformer()

    # Setup constrained forcefield
    # ffprop_prime, ffc = get_forcefield(molobj_prime)
    ffc = ChemicalForceFields.MMFFGetMoleculeForceField(molobj_prime, ffprop)

    # Set angles and constrains for all torsions
    for i, angle in enumerate(delta_angles):

        set_angle = origin_angles[i] + angle

        # Set clockwork angle
        try: Chem.rdMolTransforms.SetDihedralDeg(conformer_prime, *torsions[i], set_angle)
        except: pass

        # Set forcefield constrain
        ffc.MMFFAddTorsionConstraint(*torsions[i], False,
            set_angle-delta, set_angle+delta, 1.0e10)

    # minimize constrains
    status = run_forcefield(ffc, 500)

    # Set result
    coordinates = conformer_prime.GetPositions()
    coordinates = np.round(coordinates, coord_decimals) # rdkit hack, read description
    cheminfo.conformer_set_coordinates(conformer, coordinates)

    # minimize global
    status = run_forcefield_prime(ff, 700, force=1e-4)

    # Get current energy
    energy = ff.CalcEnergy()

    if status == 0:

        grad = ff.CalcGrad()
        grad = np.array(grad)
        grad_norm = linalg.norm(grad)

        if grad_norm > 1:
            status = 1

    debug = False
    if energy > 1000 and debug:

        print(torsions, origin_angles, delta_angles)
        print(energy, status)

        print("id")
        print(id(molobj_prime))
        print(id(molobj))

        molobj_test, status = cheminfo.sdfstr_to_molobj(sdfstr)
        coordinates = conformer.GetPositions()
        cheminfo.molobj_set_coordinates(molobj_test, coordinates)
        ffprop_t, ff_t = get_forcefield(molobj)
        run_forcefield(ff_t, 500)

        print(coordinates)


        for idxs in torsions:
            angle = Chem.rdMolTransforms.GetDihedralDeg(conformer, *idxs)
            print("ANGLE 1", angle)

        f = open("_test_dumpsdf.sdf", 'w')
        sdf = cheminfo.save_molobj(molobj)
        f.write(sdf)

        # prop, ff = get_forcefield(molobj)
        # status = run_forcefield(ff, 500)
        conformer = molobj_test.GetConformer()

        for idxs in torsions:
            angle = Chem.rdMolTransforms.GetDihedralDeg(conformer, *idxs)
            print("ANGLE 2",angle)

        print(energy, status)

        sdf = cheminfo.save_molobj(molobj_test)
        f.write(sdf)

        f.close()
        quit()

    # Get current positions
    pos = conformer.GetPositions()

    return energy, pos, status


def set_coordinates(conformer, coordinates):

    for i, pos in enumerate(coordinates):
        conformer.SetAtomPosition(i, pos)

    return


def run_job(molobj, tordb, jobstr):
    sep = ","
    jobstr = jobstr.split(sep)
    molid, torsions_idx, resolution = jobstr

    molid = int(molid)
    resolution = int(resolution)

    torsions_idx = torsions_idx.split()
    torsions_idx = [int(idx) for idx in torsions_idx]

    torsions = [tordb[idx] for idx in torsions_idx]

    job_energies, job_coordinates = get_clockwork_conformations(molobj, torsions, resolution)

    return job_energies, job_coordinates

###

def run_jobfile(molobjs, tordbs, filename, threads=0):

    # Prepare molobjs to xyz

    origins = []

    for molobj in molobjs:
        atoms, xyz = cheminfo.molobj_to_xyz(molobj)
        origins.append(xyz)

    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    if threads > 0:
        run_joblines_threads(origins, molobjs, tordbs, lines, threads=threads, dump=True)

    else:
        run_joblines(origins, molobjs, tordbs, lines, dump=True)

    return True


def run_joblines_threads(origins, molobjs, tordbs, lines, threads=1, show_bar=True, dump=False):

    # TODO Collect the conformers and return them
    # list for each line

    pool = multiprocessing.Pool(threads)

    if not show_bar:
        pool.map(partial(run_jobline, origins, molobjs, tordbs, dump=dump), lines)

    else:
        pbar = tqdm(total=len(lines))
        for i, _ in enumerate(pool.imap_unordered(partial(run_jobline, origins, molobjs, tordbs, dump=dump), lines)):
            pbar.update()
        pbar.close()

    return True


def run_joblines(origins, molobjs, tordbs, lines, dump=False):

    lines_energies = []
    lines_coordinates = []

    for i, line in enumerate(tqdm(lines)):

        job_energies, job_coordinates = run_jobline(origins, molobjs, tordbs, line, prefix=i, dump=dump)

    return True


def run_jobline(origins, molobjs, tordbs, line,
    prefix=None,
    debug=False,
    dump=False):

    sep = ","

    # TODO multiple molobjs

    line = line.strip()

    # Locate molobj
    line_s = line.split(sep)
    molid = int(line_s[0])

    molobj = molobjs[molid]
    tordb = tordbs[molid]

    # deep copy
    molobj = copy.deepcopy(molobj)
    cheminfo.molobj_set_coordinates(molobj, origins[molid])

    if dump:
        if prefix is None:
            prefix = line.replace(" ", "_").replace(",", ".")

        filename = "_tmp_data/{:}.sdf".format(prefix)

        # if os.path.exists(filename):
        #     return [],[]

    job_start = time.time()

    job_energies, job_coordinates = run_job(molobj, tordb, line)

    job_end = time.time()

    if debug:
        print(line, "-", len(job_energies), "{:5.2f}".format(job_end-job_start), filename)

    if dump:
        if debug: print("saving {:} confs to".format(len(job_energies)), filename)
        fsdf = open(filename, 'w')
        for energy, coordinates in zip(job_energies, job_coordinates):
            sdfstr = cheminfo.save_molobj(molobj, coordinates)
            fsdf.write(sdfstr)

    return job_energies, job_coordinates



#####

def readstdin_sdf():

    while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:

        line = sys.stdin.readline()

        if not line:
            yield from []
            break

        line = line.strip()
        yield line


#####

def read_tordb(filename):

    with open(filename) as f:
        lines = f.readlines()

    tordb = []
    for line in lines:
        line = line.split(":")
        idx = line[0]
        torsions = line[1]
        torsions = torsions.split(",")
        torsions = [np.array(x.split(), dtype=int) for x in torsions]
        torsions = np.asarray(torsions, dtype=int)
        tordb.append(torsions)

    return tordb


def main_redis(args):

    redis_task = args.redis_task

    if args.redis_connect is not None:
        redis_connection = args.redis_connection_str

    else:

        if not os.path.exists(args.redis_connect_file):
            print("error: redis connection not set and file does not exists")
            print("error: path", args.redis_connect_file)
            quit()

        with open(args.redis_connect_file, 'r') as f:
            redis_connection = f.read().strip()

    tasks = rediscomm.Taskqueue(redis_connection, redis_task)


    # Prepare moldb
    molecules = cheminfo.read_sdffile(args.sdf)
    molecules = [molobj for molobj in molecules]

    # Prepare tordb
    if args.sdftor is None:
        tordb = [cheminfo.get_torsions(molobj) for molobj in molecules]
    else:
        tordb = read_tordb(args.sdftor)


    # Make origins
    origins = []
    for molobj in molecules:
        xyz = cheminfo.molobj_get_coordinates(molobj)
        origins.append(xyz)


    # TODO if threads is > 0 then make more redis_workers

    do_work = lambda x: redis_worker(origins, molecules, tordb, x, debug=args.debug)
    tasks.main_loop(do_work)

    return


def redis_worker(origins, moldb, tordb, lines, debug=False):
    """
    job is lines

    try
    except
        rtn = ("error "+jobs, error)
        error = traceback.format_exc()
        print(error)

    """

    # TODO Prepare for multiple lines
    line = lines

    stamp1 = time.time()

    energies, coordinates = run_jobline(origins, moldb, tordb, line, debug=debug)

    # Prepare dump
    results = prepare_redis_dump(energies, coordinates)

    stamp2 = time.time()

    print("workpackage done {:5.3f}".format(stamp2-stamp1))


    here=1
    line = line.split(",")
    line[here] = line[here].split(" ")
    line[here] = len(line[here])
    line[here] = str(line[here])

    storestring = "Results_" + "_".join(line)
    status = ""

    return results, status, storestring


def prepare_redis_dump(energies, coordinates, coord_decimals=DEFAULT_DECIMALS):

    results = []

    for energy, coord in zip(energies, coordinates):
        coord = np.round(coord, coord_decimals).flatten().tolist()
        result = [energy, coord]
        result = json.dumps(result)
        result = result.replace(" ", "")
        results.append(result)

    results = "\n".join(results)

    return results


def main_file(args):

    suppl = cheminfo.read_sdffile(args.sdf)
    molobjs = [molobj for molobj in suppl]

    if args.sdftor:
        tordb = read_tordb(args.sdftor)
    else:
        tordb = [cheminfo.get_torsions(molobj) for molobj in molobjs]

    if args.jobfile:
        run_jobfile(molobjs, tordb, args.jobfile, threads=args.threads)

    else:
        # TODO Base on tordb
        generate_jobs(molobjs, args, tordb=tordb)

    return


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version="1.0")
    parser.add_argument('--sdf', type=str, help='SDF file', metavar='file', default="~/db/qm9s.sdf.gz")
    parser.add_argument('--sdftor', type=str, help='Torsion indexes for the SDF file', metavar='file', default=None)

    parser.add_argument('-j', '--threads', type=int, default=0)

    parser.add_argument('--jobcombos', nargs="+", help="", metavar="str")
    # OR
    parser.add_argument('--jobfile', type=str, help='txt of jobs', metavar='file')
    # OR
    parser.add_argument('--redis-task', help="redis task name", default=None)
    parser.add_argument('--redis-connect', '--redis-connect-str', help="connection to str redis server", default=None)
    parser.add_argument('--redis-connect-file', help="connection to redis server", default="~/db/redis_connection")

    parser.add_argument('--debug', action="store_true", help="", default=False)

    args = parser.parse_args()

    if "~" in args.sdf:
        args.sdf = correct_userpath(args.sdf)

    is_redis = False
    is_file = False

    if args.redis_task is not None:

        if "~" in args.redis_connect_file:
            args.redis_connect_file = correct_userpath(args.redis_connect_file)

        is_redis = True

    else:
        is_file = True


    if is_file:
        main_file(args)

    if is_redis:
        main_redis(args)


    return


if __name__ == '__main__':
    main()
