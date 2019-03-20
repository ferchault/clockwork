"""

merge jobs

"""

import pickle
import gzip

import numpy as np
import scipy.special

import adm_fchl
import adm_molecule

def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def costfunc(clockres, n):
    """
    param:
        resolution list
        total number of torsions

    return
        cost

    """
    t = len(clockres)
    maxres = max(clockres)
    value = scipy.special.binom(n, t)
    no = 2**(maxres)
    value *= no
    return value


def info(jobstr):

    job = eval(jobstr)

    (midx, toridx, reslist), uniqueidx, energy, coord = job

    return midx, toridx, reslist, uniqueidx, energy, coord



def energyandfchl(jobs, atoms, decimals=1, threshold=0.98, reset_count=False, only_energies=None, debug=False):
    """

    - sort energy
    - for all energy there are "unique", decimal=0.1
        - do qml

    """

    atoms = np.array(atoms)
    n_atoms = len(atoms)

    if len(jobs) == 0:
        return []

    if type(jobs[0]) == str:
        jobs = [info(job) for job in jobs]

    energies = []
    costlist = []

    for job in jobs:

        energy = job[4]
        energy = float(energy)

        reslist = job[2]

        cost = costfunc(reslist, 30)

        energies.append(energy)
        costlist.append(cost)

        if debug:
            print("in", job[:5])

    keep = []
    keep_origin = []
    keep_merge = []

    costlist = np.array(costlist)
    energies = np.round(energies, decimals=decimals)

    # Only do fchl on unique energies, pass others to keep_list
    if only_energies is None:
        unique_energies = np.unique(energies)
    else:
        unique_energies = np.unique(energies)

        for energy in unique_energies:

            # exists energy in only_energies
            idx, = np.where(only_energies == energy)
            if len(idx) > 0: continue

            jobidx, = np.where(energies == energy)
            for idx in jobidx:
                keep.append(idx)
                keep_origin.append(idx)
                keep_merge.append(0)

        unique_energies = only_energies

    if debug:
        print("merge: {:} energies".format(len(unique_energies)), unique_energies)

    for energy in unique_energies:

        jobidx, = np.where(energies == energy)

        if len(jobidx) == 1:

            keep.append(jobidx[0])
            keep_origin.append(jobidx[0])
            keep_merge.append(0)

            continue

        costs = costlist[jobidx]

        if debug:
            print(energy, costs)

        positions = []
        for i in jobidx:
            position = jobs[i][-1]
            position = np.array(position)
            position = position.reshape((n_atoms, 3))
            positions.append(position)

        if debug:
            print("merge: qml for {:} @ {:}".format(len(jobidx), energy))

        # Energy are the same, compare with FCHL
        similarity = adm_fchl.get_similarity(atoms, positions)

        dealtwith = []

        for j, row in enumerate(similarity):

            if j in dealtwith: continue

            rowsim, = np.where(row >= threshold)

            # if the molecules are broken, fchl will give similarity of nan
            # just skip these molecules
            if len(rowsim) == 0: continue

            # find the lowest row cost, of all (incl dealtwith)
            rowcosts = costs[rowsim]
            costidx = np.argsort(rowcosts)[0]
            keep_sim_cost = rowsim[costidx]

            # want not dealt with index and lowest cost
            rowsim = [idx for idx in rowsim if idx not in dealtwith]
            rowcosts = costs[rowsim]
            costidx = np.argsort(rowcosts)[0]
            keep_sim_pos = rowsim[costidx]

            # convert similarity idx to jobidx
            keep_idx = jobidx[keep_sim_pos]
            keep_cost = jobidx[keep_sim_cost]

            if debug:
                print("keep", keep_idx, keep_cost, costlist[keep_cost])

            # Save the indexes
            keep.append(keep_idx)
            keep_origin.append(keep_cost)
            keep_merge.append(len(rowsim) - 1)

            dealtwith += list(rowsim)


    # correct jobs with origins
    rtnjobs = []

    for jidx, oidx, nmerge in zip(keep, keep_origin, keep_merge):

        job = jobs[jidx]
        job = list(job)

        if debug:
            print("rtn", jidx, job[:5])

        if jidx != oidx:

            ojob = jobs[oidx]
            origin = ojob[2]
            job[2] = origin

        # merge count
        if reset_count: job[3] = nmerge
        else: job[3] += nmerge

        rtnjobs.append(job)

    return rtnjobs


def merge_energy(jobs, decimals=1):

    if type(jobs[0]) == str:
        jobs = [info(job) for job in jobs]

    energies = []
    costlist = []

    for job in jobs:

        energy = job[4]
        energy = float(energy)

        reslist = job[2]

        cost = costfunc(reslist, 30)

        energies.append(energy)
        costlist.append(cost)


    costlist = np.array(costlist)
    energies = np.round(energies, decimals=decimals)
    unique_energies = np.unique(energies)

    keep = []

    for energy in unique_energies:

        idx, = np.where(energies == energy)
        costs = costlist[idx]

        idxsort = np.argsort(costs)

        j = idxsort[0]
        keep.append(idx[j])


    jobs = [jobs[idx] for idx in keep]

    return jobs


def merge_fchl(jobs):
    """

    take a list of jobs and merge them

    """


    return jobs


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help='', metavar='file')
    parser.add_argument('-s', '--sdf', type=str, help='', metavar='file')
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        lines = f.readlines()


    ext = args.sdf.split(".")[-1]

    if ext == "sdf":

        moldb = adm_molecule.load_sdf(args.sdf)

    elif ext == "gz":

        inf = gzip.open(args.sdf)
        moldb = adm_molecule.load_sdf_file(inf)

    else:

        print("unknown format")
        quit()


    lines_molid = []

    try:
        lines_molid = load_obj("tmp_molid")
    except FileNotFoundError:

        for i, line in enumerate(lines):

            line = eval(line)
            molid = line[0]
            lines_molid.append(molid)

        save_obj("tmp_molid", lines_molid)

    lines_molid = np.array(lines_molid)
    unique_molid = np.unique(lines_molid)

    for molid in unique_molid:

        # Create mol
        mol = moldb[molid]

        atoms = [atom for atom in mol.GetAtoms()]
        atoms_str = [atom.GetSymbol() for atom in atoms]
        atoms_int = [atom.GetAtomicNum() for atom in atoms]

        # find all lines with for mol
        lineidx, = np.where(lines_molid == molid)

        print(lineidx)

        resultlist = []
        for i in lineidx:
            resultlist.append(eval(lines[i]))

        # Merge results
        results = energyandfchl(resultlist, atoms_int)

        for result in results:
            print(result[:3])

        # print("merged mol", molid, len(lineidx), "->", len(results))



    quit()


    # Select mol and find atoms
    job = info(lines[0])
    molidx = job[0]
    mol = moldb[molidx]

    atoms = [atom for atom in mol.GetAtoms()]
    atoms_str = [atom.GetSymbol() for atom in atoms]
    atoms_int = [atom.GetAtomicNum() for atom in atoms]


    jobs = energyandfchl(lines, atoms_int, reset_count=True)

    for job in jobs:
        print(job)

    return


if __name__ == '__main__':
    main()

