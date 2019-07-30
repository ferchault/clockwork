
from tqdm import tqdm

import copy
import numpy as np
import numpy as np
import clockwork

from chemhelp import cheminfo
import rmsd
import worker

import os
import joblib

import similarity_fchl19 as sim

import qml
from qml import fchl

# Set QML fortran threads to 1
os.environ["OMP_NUM_THREADS"] = "1"

# Set local cache
cachedir = '.pycache'
memory = joblib.Memory(cachedir, verbose=0)



def get_representations_fchl(atoms, coordinates_list, max_size=30):

    if max_size is None:
        max_size = len(atoms)

    replist = []

    for coordinates in coordinates_list:
        rep = fchl.generate_representation(coordinates, atoms, max_size=max_size, cut_distance=10**6)
        replist.append(rep)

    return replist


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


def cumulative_similarity(atoms, representations,
    threshold=0.98):
    """
    """


    u_representations = [representations[0]]
    s_idxs = [0]

    for i, representation in enumerate(representations[1:]):

        i += 1

        similar = merge_asymmetric_similarity(atoms,
            [representation],
            u_representations,
            threshold=threshold)

        # We are only looking at one representation
        similar = similar[0]

        if len(similar) > 0:
            continue

        u_representations += [representation]
        s_idxs += [i]

    return np.asarray(s_idxs)


def merge(atoms, energies, representations,
    debug=False,
    decimals=1,
    threshold=0.98):
    """

    """

    representations = np.asarray(representations)
    energies = np.round(energies, decimals=decimals)

    unique_energies = np.unique(energies)

    # Return index from x, with idx of same similarity if empty, it is new
    keepidx = []

    for uenergy in unique_energies:

        u_idxs, = np.where(energies == uenergy)
        u_representations = representations[u_idxs]

        if len(u_idxs) == 1:
            continue

        if debug:
            print(uenergy, len(u_idxs))

        unique_idxs = cumulative_similarity(atoms, u_representations)

        for simidx in unique_idxs:
            simidx = u_idxs[simidx]
            keepidx += [simidx]

    return keepidx


def merge_asymmetric(atoms, energies_x, energies_y, rep_x, rep_y,
    decimals=1,
    threshold=0.98):
    """
    """

    rep_x = np.asarray(rep_x)
    rep_y = np.asarray(rep_y)

    energies_x = np.round(energies_x, decimals=decimals)
    energies_y = np.round(energies_y, decimals=decimals)

    new_unique_energies = np.unique(energies_x)

    # Return index from x, with idx of same similarity if empty, it is new
    rtnidx = [[] for x in range(len(energies_x))]

    # Iterate over unique energies for kernel similarities
    for uenergy in new_unique_energies:

        idx_x, = np.where(energies_x == uenergy)
        idx_y, = np.where(energies_y == uenergy)

        # all unique, continue
        if len(idx_y) == 0: continue

        # print(uenergy, len(idx_x), len(idx_y))

        # list of list of idx
        similar = merge_asymmetric_similarity(atoms,
            rep_x[idx_x],
            rep_y[idx_y],
            threshold=threshold
            )

        # convert local similarity to idx_y
        for i, sidx in enumerate(similar):
            sidx = [idx_y[j] for j in sidx]
            sidx = np.asarray(sidx)
            similar[i] = sidx

        # Create rtn idx
        for i, idx in enumerate(idx_x):
            rtnidx[idx] = similar[i]

    return rtnidx


def merge_asymmetric_similarity(
    atoms,
    rep_x,
    rep_y,
    threshold=0.98,
    **kwargs):
    """
    atoms list int
    positions list list float


    returns similarity for each X, compared to each Y.
    So basically, for the first X you will get a list of idx where similarity
    is within threshold

    """

    rep_x = np.asarray(rep_x)
    rep_y = np.asarray(rep_y)

    Nx = len(rep_x)
    Ny = len(rep_y)

    # Energy are the same, compare with FCHL
    similarity = sim.get_kernel(rep_x, rep_y, [atoms]*Nx, [atoms]*Ny, **kwargs)

    similar = []

    for j, row in enumerate(similarity):

        # In this row, find all similar coordinates
        rowsim, = np.where(row >= threshold)
        similar.append(rowsim)

    return similar


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
    keep_representations = [sim.get_representation(atoms, coordinates) for coordinates in keep_coordinates]




    for line in tqdm(lines[1:]):
    # for line in lines[1:]:

        sdffile = datadir + line + ".sdf"
        from_energies, from_coordinates = worker.get_sdfcontent(sdffile)
        from_representations = [sim.get_representation(atoms, coordinates) for coordinates in from_coordinates]

        # print("merge", len(from_energies))

        # Asymmetrically add new conformers
        # idxs_ref = merge_asymmetric_fchl18(atoms,
        #     from_energies,
        #     keep_energies,
        #     from_coordinates,
        #     keep_coordinates, debug=False)


        # find
        # energies = np.round(keep_energies, 1)
        # idx, = np.where(energies == 14.8)
        # print(from_coordinates[idx[0]])
        #
        # print(idx)


        idxs = merge_asymmetric(
            atoms,
            from_energies,
            keep_energies,
            from_representations,
            keep_representations)



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

    representations19_x = [sim.get_representation(atoms, coordinates) for coordinates in coordinates_x]
    representations19_y = [sim.get_representation(atoms, coordinates) for coordinates in coordinates_y]
    representations19_x = np.asarray(representations19_x[:5])
    representations19_y = np.asarray(representations19_y[:5])

    nx = len(representations19_x)
    ny = len(representations19_y)

    similarity = sim.get_kernel(representations19_x, representations19_y, [atoms]*nx, [atoms]*ny)
    print(similarity)

    # JCK 

    # fchl18
    representations_x = get_representations_fchl(atoms, coordinates_x, max_size=len(atoms))
    representations_x = np.asarray(representations_x[:5])
    representations_y = get_representations_fchl(atoms, coordinates_y, max_size=len(atoms))
    representations_y = np.asarray(representations_y[:5])
    similarity = get_kernel_fchl(representations_x, representations_y)
    print("qml kernel:")
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

    if args.sdf is not None:
        help(qml)
        mergesdfs(args.sdf)
        quit()

    outdir = "_tmp_apentane_cost/"
    sdfdir = "_tmp_apentane/"
    joblist = "_tmp_joblist.txt"

    with open(joblist, 'r') as f:
        lines = f.readlines()

    costsdf = sortjobs(lines)

    for key in costsdf.keys():
        print(key, len(costsdf[key]))

    keys = costsdf.keys()
    # keys = ["2_1"]
    # keys = ["3_2"]
    # keys = ["4_1"]
    # keys = ["2_4"]

    for x in keys:
        print()
        print(x, len(costsdf[x]))

        filename = outdir + x + ".sdf"

        if os.path.isfile(filename):
            print("skipped", filename)
            continue

        lines = costsdf[x]
        sdfs = merge_sdflist(lines, datadir=sdfdir)

        f = open(filename, 'w')
        f.write(sdfs)
        f.close()


    return

if __name__ == '__main__':
    main()
