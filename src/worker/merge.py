
import copy
import numpy as np
from qml import fchl
import numpy as np
import clockwork

from chemhelp import cheminfo
import rmsd

import os
os.environ["OMP_NUM_THREADS"] = "1"

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

        # if debug:
        #     print("amerge", uenergy, len(idx_x), len(idx_y))

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


def get_asymmetric_similarity(atoms, positions_x, positions_y):

    # Energy are the same, compare with FCHL
    representations_x = get_representations_fchl(atoms, positions_x)
    representations_y = get_representations_fchl(atoms, positions_y)
    representations_x = np.array(representations_x)
    representations_y = np.array(representations_y)

    similarity = get_kernel_fchl(representations_x, representations_y)

    return similarity


def get_similarity(atoms, positions):

    # Energy are the same, compare with FCHL
    representations = get_representations_fchl(atoms, positions)
    representations = np.array(representations)

    similarity = get_kernel_fchl(representations, representations)

    return similarity


def get_representations_fchl(atoms, coordinates_list, max_size=30):

    replist = []

    for coordinates in coordinates_list:
        rep = fchl.generate_representation(coordinates, atoms, max_size=max_size, cut_distance=10**6)
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


def main():
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version="1.0")
    parser.add_argument('--format', action='store', help='', metavar='FMT')
    parser.add_argument('--xyz', action='store', help='', metavar='FILE')
    parser.add_argument('--sdf', action='store', help='', metavar='FILE')

    args = parser.parse_args()

    # TODO merge sdf


    return

if __name__ == '__main__':
    main()
