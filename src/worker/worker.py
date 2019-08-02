#

import copy
from tqdm import tqdm

import itertools
import multiprocessing
import os
import sys
import time
from functools import partial

import numpy as np

import clockwork
import merge
from chemhelp import cheminfo
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalForceFields

import similarity_fchl19 as sim

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


def generate_jobs(molobj, tordb=None, min_cost=0, max_cost=15, prefix="1"):

    # TODO Group by cost?

    if tordb is None:
        tordb = cheminfo.get_torsions(molobj)

    total_torsions = len(tordb)

    cost_input, cost_cost = clockwork.generate_costlist(total_torsions=total_torsions)

    for (n_tor, resolution), cost in zip(cost_input[min_cost:max_cost], cost_cost[min_cost:max_cost]):

        combinations = clockwork.generate_torsion_combinations(total_torsions, n_tor)

        for combination in combinations:

            # torsions = [tordb[x] for x in combination]

            jobstr = prefix + ","
            torstr = " ".join([str(x) for x in combination])
            resstr = str(resolution)
            jobstr += torstr + "," + resstr
            print(jobstr)

    quit()
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
    ff=None):

    if ffprop is None or ff is None:
        ffprop, ff = get_forcefield(molobj)

    # Setup constrained forcefield
    ffc = ChemicalForceFields.MMFFGetMoleculeForceField(molobj, ffprop)

    # Set angles and constrains for all torsions
    for i, angle in enumerate(delta_angles):

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
    status = run_forcefield_prime(ff, 700, force=1e-4)

    # Get current energy
    energy = ff.CalcEnergy()

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
    for molobj in [molobjs]:
        atoms, xyz = cheminfo.molobj_to_xyz(molobj)
        origins.append(xyz)

    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    if threads > 0:
        run_joblines_threads(origins, molobjs, tordbs, lines, threads=threads)

    else:
        run_joblines(origins, molobjs, tordbs, lines)

    return True


def run_joblines_threads(origins, molobjs, tordbs, lines, threads=1, show_bar=True):

    # TODO tdqm on pool.map?
    # https://github.com/tqdm/tqdm/issues/484

    pool = multiprocessing.Pool(threads)

    if not show_bar:
        pool.map(partial(run_jobline, origins, molobjs, tordbs), lines)

    else:
        pbar = tqdm(total=len(lines))
        for i, _ in enumerate(pool.imap_unordered(partial(run_jobline, origins, molobjs, tordbs), lines)):
            pbar.update()
        pbar.close()

    return True


def run_jobline(origins, molobjs, tordbs, line,
    prefix=None,
    debug=False):

    # TODO redis options, don't dump results to hdd

    # TODO multiple molobjs

    molobj = molobjs
    tordb = tordbs

    # deep copy
    molobj = copy.deepcopy(molobj)
    cheminfo.molobj_set_coordinates(molobj, origins[0])

    line = line.strip()

    if prefix is None:
        prefix = line \
            .replace(" ", "_") \
            .replace(",", ".")

    filename = "_tmp_data/{:}.sdf".format(prefix)

    if os.path.exists(filename):
        return [],[]

    job_start = time.time()

    job_energies, job_coordinates = run_job(molobj, tordb, line)

    job_end = time.time()

    if debug:
        print(line, "-", len(job_energies), "{:5.2f}".format(job_end-job_start), filename)

    fsdf = open(filename, 'w')
    for energy, coordinates in zip(job_energies, job_coordinates):
        sdfstr = cheminfo.save_molobj(molobj, coordinates)
        fsdf.write(sdfstr)

    return job_energies, job_coordinates


def run_joblines(origins, molobjs, tordbs, lines):

    for i, line in enumerate(tqdm(lines)):

        run_jobline(molobjs, tordbs, line, prefix=i)

    return True


#####

def readstdin_sdf():

    while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:

        line = sys.stdin.readline()

        if not line:
            yield from []
            break

        line = line.strip()
        yield line


def correct_userpath(filepath):
    return os.path.expanduser(filepath)




def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version="1.0")
    parser.add_argument('--sdf', type=str, help='SDF file', metavar='file', default="~/db/qm9s.sdf.gz")
    parser.add_argument('--jobfile', type=str, help='txt of jobs', metavar='file')

    parser.add_argument('-j', '--threads', type=int, default=1)

    args = parser.parse_args()

    if "~" in args.sdf:
        args.sdf = correct_userpath(args.sdf)

    print(args.sdf)

    suppl = cheminfo.read_sdffile(args.sdf)
    molecules = [molobj for molobj in suppl]
    n_molcules = len(molecules)
    molobj = molecules[0]
    torsions = cheminfo.get_torsions(molobj)

    # torsion_idx = 0
    # torsion = torsions[torsion_idx:2]
    # resolution = 4
    # conformers = get_clockwork_conformations(molobj, torsion, resolution)

    # converge_clockwork(molobj, torsions)

    if args.jobfile:
        run_jobfile(molobj, torsions, args.jobfile, threads=args.threads)
    else:
        generate_jobs(molobj)

    return


def get_energy_test():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version="1.0")
    parser.add_argument('--sdf', type=str, help='SDF file', metavar='file', default="~/db/qm9s.sdf.gz")
    parser.add_argument('--jobfile', type=str, help='txt of jobs', metavar='file')

    parser.add_argument('-j', '--threads', type=int, default=1)

    args = parser.parse_args()

    if "~" in args.sdf:
        args.sdf = correct_userpath(args.sdf)


    sdf = open(args.sdf, 'r').read()

    # Testing weird convergence
    molobj, status = cheminfo.sdfstr_to_molobj(sdf)
    energy = get_energy(molobj)
    print(energy)
    conformer = molobj.GetConformer()
    ffprop, ff = get_forcefield(molobj)
    status = run_forcefield_prime(ff, 700, force=1e-4)
    energy = get_energy(molobj)
    print(energy)

    sdfstr = cheminfo.molobj_to_sdfstr(molobj)
    print(sdfstr)

    return


if __name__ == '__main__':
    main()
