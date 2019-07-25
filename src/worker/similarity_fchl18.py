
from tqdm import tqdm

import copy
import numpy as np
import numpy as np
import clockwork

import workkernel

from chemhelp import cheminfo
import rmsd
import worker

import os
import joblib

from qml import fchl

# Set QML fortran threads to 1
os.environ["OMP_NUM_THREADS"] = "1"

# Set local cache
cachedir = '.pycache'
memory = joblib.Memory(cachedir, verbose=0)

# Set default parameters for kernel similarity
DSIGMAS = [0.8]
DPARAMETERS = {
    'alchemy': 'off',
    "cut_distance": 10**6,
    'kernel_args': {"sigma": DSIGMAS},
}


def merge_with_cost(atoms, energies, coordinates, costs, decimals=1, threshold=0.98):
    """
    atoms list int
    energies list float
    coordinates list list float
    costs list float

    merge conformers and keep lowest cost

    """

    merged_idxs = merge(atoms, energies, coordinates, decimals=decimals, threshold=threshold)

    for idxs in merged_idxs:

        print(idxs)
        quit()

    return


def merge_two_jobs(atoms,
    energies_x,
    energies_y,
    coordinates_x,
    coordinates_y,
    costs_x,
    costs_y,
    decimals=1,
    threshold=0.98):

    # TODO

    return


def align_coordinates(coordinates):

    # Align everything to first coordinate

    coordinates = copy.deepcopy(coordinates)

    coordinate_ref = coordinates[0]
    coordinate_ref -= rmsd.centroid(coordinate_ref)
    coordinates[0] = coordinate_ref

    for i, coordinate in enumerate(coordinates[1:]):

        coordinate -= rmsd.centroid(coordinate)
        U = rmsd.kabsch(coordinate, coordinate_ref)
        coordinate = np.dot(coordinate, U)
        coordinates[i] = coordinate

    return coordinates


def merge_asymmetric(atoms, energies_x, energies_y, coordinates_x, coordinates_y,
    decimals=1,
    threshold=0.98,
    molobj=None,
    debug=False):
    """
    """

    coordinates_x = np.asarray(coordinates_x)
    coordinates_y = np.asarray(coordinates_y)

    energies_x = np.round(energies_x, decimals=decimals)
    energies_y = np.round(energies_y, decimals=decimals)

    new_unique_energies = np.unique(energies_x)

    # Return index from x, with idx of same similarity
    # if empty, it is new
    rtnidx = [[] for x in range(len(energies_x))]

    for uenergy in new_unique_energies:

        idx_x, = np.where(energies_x == uenergy)
        idx_y, = np.where(energies_y == uenergy)

        if debug:
            print("amerge", uenergy, len(idx_x), len(idx_y))

        if len(idx_y) == 0:
            # all unique, continue
            continue

        if len(idx_y) > 5 and molobj is not None:

            atoms_str = [cheminfo.convert_atom(atom) for atom in atoms]

            # f = open("_tmp_test.xyz", 'w')
            f = open("_tmp_test.sdf", 'w')

            coordinates_dump = align_coordinates(coordinates_y[idx_y])
            energies = energies_y[idx_y]

            for energy, coordinates in zip(energies, coordinates_dump):
                # xyzstr = rmsd.set_coordinates(atoms_str, coordinates, title=str(energy))
                # xyzstr += "\n"
                sdfstr = cheminfo.save_molobj(molobj, coordinates)

                f.write(sdfstr)
            f.close()


        # list of list of idx
        similar = merge_asymmetric_similarity(atoms,
            coordinates_x[idx_x],
            coordinates_y[idx_y],
            threshold=threshold)

        # convert local similarity to idx_y
        for i, sidx in enumerate(similar):
            sidx = [idx_y[j] for j in sidx]
            sidx = np.asarray(sidx)
            similar[i] = sidx

        # Create rtn idx
        for i, idx in enumerate(idx_x):
            rtnidx[idx] = similar[i]

    return rtnidx


def merge_asymmetric_objs(
    energies_x,
    energies_y,
    objs_x,
    objs_y,
    decimals=1,
    threshold=0.98,
    molobj=None,
    debug=False,
    **kwargs):
    """
    merge_asymmetric_similarity_objs

    """

    objs_x = np.asarray(objs_x)
    objs_y = np.asarray(objs_y)

    energies_x = np.round(energies_x, decimals=decimals)
    energies_y = np.round(energies_y, decimals=decimals)

    new_unique_energies = np.unique(energies_x)

    # Return index from x, with idx of same similarity
    # if empty, it is new
    rtnidx = [[] for x in range(len(energies_x))]

    for uenergy in new_unique_energies:

        idx_x, = np.where(energies_x == uenergy)
        idx_y, = np.where(energies_y == uenergy)

        if debug:
            print("amerge", uenergy, len(idx_x), len(idx_y))

        if len(idx_y) == 0:
            # all unique, continue
            continue


        # list of list of idx
        similar = merge_asymmetric_similarity_objs(
            objs_x[idx_x],
            objs_y[idx_y],
            threshold=threshold,
            **kwargs)

        # convert local similarity to idx_y
        for i, sidx in enumerate(similar):
            sidx = [idx_y[j] for j in sidx]
            sidx = np.asarray(sidx)
            similar[i] = sidx

        # Create rtn idx
        for i, idx in enumerate(idx_x):
            rtnidx[idx] = similar[i]

    return rtnidx


def merge(atoms, energies, coordinates, decimals=1, threshold=0.98):
    """
    atoms list int
    energies list float
    coordinates list list float
    """

    N = len(energies)

    coordinates = np.asarray(coordinates)

    # Find unique energies
    energies = np.round(energies, decimals=decimals)
    unique_energies = np.unique(energies)

    rtnidx = []

    for uenergy in unique_energies:

        idx, = np.where(energies == uenergy)

        if idx.shape[0] == 1:
            rtnidx.append(idx)
            continue

        # list of list of idx
        similar = merge_similarity(atoms, coordinates[idx], threshold=threshold)

        for i, sidx in enumerate(similar):
            sidx = [idx[j] for j in sidx]
            similar[i] = sidx
            rtnidx.append(sidx)

    return rtnidx


def merge_similarity(atoms, positions, threshold=0.98):
    """
    atoms list int
    positions list list float
    """

    # Energy are the same, compare with FCHL
    similarity = get_similarity(atoms, positions)

    dealtwith = []
    similar = []

    for j, row in enumerate(similarity):

        # Ignore if already merged
        if j in dealtwith: continue

        # In this row, find all similar coordinates
        rowsim, = np.where(row >= threshold)

        # if the molecules are broken, fchl will give similarity of nan
        # just skip these molecules. There should be at least 1 under threshold (itself)
        if len(rowsim) == 0: continue

        similar.append(rowsim)
        dealtwith += list(rowsim)

    return similar


def merge_asymmetric_similarity(atoms,
    positions_x,
    positions_y,
    threshold=0.98):
    """
    atoms list int
    positions list list float


    returns similarity for each X, compared to each Y.
    So basically, for the first X you will get a list of idx where similarity
    is within threshold

    """

    # Energy are the same, compare with FCHL
    similarity = get_asymmetric_similarity(atoms, positions_x, positions_y)

    similar = []

    for j, row in enumerate(similarity):

        # In this row, find all similar coordinates
        rowsim, = np.where(row >= threshold)
        similar.append(rowsim)

    return similar


def merge_asymmetric_similarity_objs(
    objs_x,
    objs_y,
    threshold=0.98,
    **kwargs):
    """
    atoms list int
    positions list list float


    returns similarity for each X, compared to each Y.
    So basically, for the first X you will get a list of idx where similarity
    is within threshold

    """

    # Energy are the same, compare with FCHL
    similarity = workkernel.get_kernel_from_objs(objs_x, objs_y, **DPARAMETERS)
    # print(similarity)

    similar = []

    for j, row in enumerate(similarity):

        # In this row, find all similar coordinates
        rowsim, = np.where(row >= threshold)
        similar.append(rowsim)

    return similar


def get_asymmetric_similarity(atoms, positions_x, positions_y):

    # Energy are the same, compare with FCHL
    representations_x = get_representations_fchl(atoms, positions_x)
    representations_y = get_representations_fchl(atoms, positions_y)
    representations_x = np.array(representations_x)
    representations_y = np.array(representations_y)

    similarity = get_kernel_fchl(representations_x, representations_y)

    return similarity


def get_similarity(atoms, positions, max_size=None):

    # Energy are the same, compare with FCHL
    representations = get_representations_fchl(atoms, positions, max_size=max_size)
    representations = np.array(representations)

    similarity = get_kernel_fchl(representations, representations)

    return similarity


def get_representations_fchl(atoms, coordinates_list, max_size=30):

    if max_size is None:
        max_size = len(atoms)

    replist = []

    for coordinates in coordinates_list:
        rep = fchl.generate_representation(coordinates, atoms, max_size=max_size, cut_distance=10**6)
        replist.append(rep)

    return replist


def get_kernel_fchl_(rep_alpha, rep_beta, debug=False):
    """
    """

    sigmas = [0.8]

    kernel, = workkernel.get_global_kernels_split(rep_alpha, rep_beta,
            kernel_args={"sigma":sigmas},
            cut_distance=10**6,
            alchemy="off")

    return kernel


def get_kernel_fchl(rep_alpha, rep_beta, debug=False):

    # Print OMP
    if debug:
        print(os.environ["OMP_NUM_THREADS"])

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


@memory.cache
def sortjobs(lines):
    """
    """

    costview = {}

    for line in lines:
        line = line.strip()
        sdffile = line.replace(",", ".").replace(" ", "_")
        line = line.split(",")
        res = line[-1]
        body = len(line[1].split())
        body = str(body)

        cost = body + "_" + res

        if cost not in costview:
            costview[cost] = []

        costview[cost].append(sdffile)

    return costview


def merge_cumulative(cost_list, coordinates_list):
    """
    """


    return


def merge_sdflist(lines, datadir=""):

    # Get first line
    sdffile = datadir + lines[0] + ".sdf"

    molobj, atoms, keep_energies, keep_coordinates = worker.get_sdfcontent(sdffile, rtn_atoms=True)
    keep_representations = [workkernel.FchlRepresentation(atoms, coordinates, **DPARAMETERS) for coordinates in keep_coordinates]


    for line in tqdm(lines[1:]):
    # for line in lines[1:]:

        sdffile = datadir + line + ".sdf"
        from_energies, from_coordinates = worker.get_sdfcontent(sdffile)
        from_representations = [workkernel.FchlRepresentation(atoms, coordinates, **DPARAMETERS) for coordinates in from_coordinates]

        # print("merge", len(from_energies))

        # Asymmetrically add new conformers
        # idxs = merge_asymmetric(atoms,
        #     from_energies,
        #     keep_energies,
        #     from_coordinates,
        #     keep_coordinates, debug=False)


        idxs = merge_asymmetric_objs(
            from_energies,
            keep_energies,
            from_representations,
            keep_representations, **DPARAMETERS)


        for i, idx in enumerate(idxs):

            # if conformation already exists, continue
            if len(idx) > 0: continue

            # Add new unique conformation to collection
            keep_energies.append(from_energies[i])
            keep_coordinates.append(from_coordinates[i])
            keep_representations.append(from_representations[i])

    sdfstr = ""
    for coordinates in keep_coordinates:
        sdfstr += cheminfo.save_molobj(molobj, coordinates)

    return sdfstr



def mergesdfs(sdflist):
    """
    """
    # Merge sdf0 with sdf1 and etc

    molobjs_list = []

    for sdf in sdflist:
        sdfs = cheminfo.read_sdffile(sdf)
        molobjs = [molobj for molobj in sdfs]
        molobjs_list.append(molobjs)



    coordinates_sdf = []

    for molobjs in molobjs_list:
        coordinates = [cheminfo.molobj_get_coordinates(molobj) for molobj in molobjs]
        coordinates_sdf.append(coordinates)


    atoms, xyz = cheminfo.molobj_to_xyz(molobjs_list[0][0])

    coordinates_x = coordinates_sdf[0]
    coordinates_y = coordinates_sdf[1]


    # JCK 
    sigmas = [0.8]
    parameters = {
        'alchemy': 'off',
        "cut_distance": 10**6,
        'kernel_args': {"sigma":sigmas},
    }

    repObjs_x = [workkernel.FchlRepresentation(atoms, coordinates, **parameters) for coordinates in coordinates_x[:3]]
    repObjs_y = [workkernel.FchlRepresentation(atoms, coordinates, **parameters) for coordinates in coordinates_y[:4]]

    similarity = workkernel.get_kernel_from_objs(repObjs_x, repObjs_y, **parameters)
    print("qml obj kernel:")
    print(similarity)

    # JCK 


    # Energy are the same, compare with FCHL
    representations_x = get_representations_fchl(atoms, coordinates_x, max_size=len(atoms))
    representations_x = np.asarray(representations_x[:3])

    representations_y = get_representations_fchl(atoms, coordinates_y, max_size=len(atoms))
    representations_y = np.asarray(representations_y[:4])

    similarity = get_kernel_fchl(representations_x, representations_y)
    print("qml kernel:")
    print(similarity)

    similarity = get_kernel_fchl_(representations_x, representations_y)
    print("qml fst kernel:")
    print(similarity)


    return



def main():
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version="1.0")
    parser.add_argument('--format', action='store', help='', metavar='FMT')
    parser.add_argument('--xyz', action='store', help='', metavar='FILE')
    parser.add_argument('--sdf', nargs="+", action='store', help='', metavar='FILE')

    args = parser.parse_args()

    # mergesdfs(args.sdf)
    #
    # quit()

    outdir = "_tmp_apentane_cost/"
    sdfdir = "_tmp_apentane/"
    joblist = "_tmp_joblist.txt"

    with open(joblist, 'r') as f:
        lines = f.readlines()

    costsdf = sortjobs(lines)

    for key in costsdf.keys():
        print(key, len(costsdf[key]))

    # keys = ["2_1"]
    # keys = ["3_2"]
    # keys = ["4_1"]
    keys = ["2_4"]

    for x in keys:
        print()
        print(x, len(costsdf[x]))

        filename = outdir + x + ".sdf"
        # if os.path.isfile(filename):
        #     continue

        lines = costsdf[x]
        sdfs = merge_sdflist(lines, datadir=sdfdir)

        f = open(filename, 'w')
        f.write(sdfs)
        f.close()


    return

if __name__ == '__main__':
    main()
