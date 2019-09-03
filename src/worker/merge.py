import time
from tqdm import tqdm
import sys
import select

import multiprocessing as mp

import copy
import numpy as np
import numpy as np
import clockwork

from chemhelp import cheminfo
import rmsd
import worker

import matplotlib.pyplot as plt

import os
import joblib

import json

import similarity_fchl19 as sim

import qml
from qml import fchl

# Set QML fortran threads to 1
os.environ["OMP_NUM_THREADS"] = "1"

# Set local cache
cachedir = '.pycache'
memory = joblib.Memory(cachedir, verbose=0)


DEFAULT_DECIMALS = 5

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


def cumulative_similarity_lazy(atoms, coordinates,
    threshold=0.98):
    """
    """

    u_representations = [sim.get_representation(atoms, coordinates[0])]
    s_idxs = [0]

    for i, coord in enumerate(coordinates[1:]):

        i += 1

        representation = sim.get_representation(atoms, coord)

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


    return


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


def merge_cost(atoms,
    energies_x,
    energies_y,
    representations_x,
    representations_y,
    costs_x,
    costs_y):

    

    return


def merge_coordinates(atoms, energies, coordinates,
    debug=False,
    decimals=1,
    threshold=0.98):
    """

    """

    coordinates = np.asarray(coordinates)

    # find unique energies
    energies = np.round(energies, decimals=decimals)
    unique_energies = np.unique(energies)

    # Return index from x, with idx of same similarity if empty, it is new
    keepidx = []

    for uenergy in unique_energies:

        u_idxs, = np.where(energies == uenergy)
        u_coordinates = coordinates[u_idxs]

        if len(u_idxs) == 1:
            keepidx += list(u_idxs)
            continue

        if debug:
            print(uenergy, len(u_idxs))

        # u_representations = [sim.get_representation(atoms, coord) for coord in u_coordinates]
        # u_representations = np.asarray(u_representations)

        unique_idxs = cumulative_similarity_lazy(atoms, u_coordinates)

        for simidx in unique_idxs:
            simidx = u_idxs[simidx]
            keepidx += [simidx]

    return keepidx



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
            keepidx += list(u_idxs)
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
    merge x into y (keep y)

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



def generate_sdf(filename):

    suppl = cheminfo.read_sdffile(filename)

    # start
    molobj = next(suppl)
    atoms, coord = cheminfo.molobj_to_xyz(molobj)
    energy = worker.get_energy(molobj)
    representation = sim.get_representation(atoms, coord)

    # init lists
    molobjs = [molobj]
    energies = [energy]
    coordinates = [coord]
    representations = [representation]

    # collect the rest
    for molobj in suppl:

        energy = worker.get_energy(molobj)
        coord = cheminfo.molobj_get_coordinates(molobj)
        representation = sim.get_representation(atoms, coord)

        molobjs.append(molobj)
        energies.append(energy)
        coordinates.append(coord)
        representations.append(representation)


    return molobjs, energies, coordinates, representations


def read_txt(filename, n_atoms):

    energies = []
    coordinates = []
    costs = []

    lines = open(filename, 'r').readlines()

    for line in lines:

        line = eval(line)

        energy = line[0]
        coord = line[1]
        coord = np.asarray(coord)
        coord = coord.reshape((n_atoms, 3))

        if len(line) > 2:
            cost = line[2]
            costs.append(cost)

        energies.append(energy)
        coordinates.append(coord)

    if len(costs) > 0:
        return energies, coordinates, costs

    return energies, coordinates


def read_resulttxt(atoms_list, filename, molidx=None):
    """

    """

    if molidx is None:
        molidx = filename.split("/")[-1].split("_")[0]
        molidx = int(molidx)

    atoms = atoms_list[molidx]
    n_atoms = len(atoms)

    lines = open(filename, 'r').readlines()

    energies = []
    coordinates = []

    for line in lines:

        line = eval(line)

        energy = line[0]
        coord = line[1]
        coord = np.asarray(coord)
        coord = coord.reshape((n_atoms, 3))

        energies.append(energy)
        coordinates.append(coord)

    return energies, coordinates, atoms


def merge_result(atoms, energies, coordinates):

    # cumulative merging
    idxs = merge_coordinates(atoms, energies, coordinates)

    uenergies = []
    ucoordinates = []

    for idx in idxs:

        uenergies.append(energies[idx])
        ucoordinates.append(coordinates[idx])

    return uenergies, ucoordinates


def merge_results_filenames(molobjs, filenames):

    print("filenames")

    # init
    energies = []
    coordinates = []
    representations = []
    atoms = []
    n_total = 0

    atoms_list = [cheminfo.molobj_to_atoms(molobj) for molobj in molobjs]

    for filename in filenames:
        merge_results_filename(filename, atoms_list)

    return


def merge_results_filename(filename, atoms_list, iolock=None):

    energies, coordinates, atoms = read_resulttxt(atoms_list, filename)
    energies, coordinates = merge_result(atoms, energies, coordinates)
    results = worker.prepare_redis_dump(energies, coordinates)

    dumpfile = filename + ".merged"

    with open(dumpfile, 'w') as f:
        f.write(results)

    if iolock is None:
        print("merge", filename, "and found {:}".format(len(energies)))
    else:
        with iolock:
            print("merge", filename, "and found {:}".format(len(energies)))

    return


def merge_results_cumulative_prime(sdffile, filenametemplate, debug=True, molid=0):

    # the G list
    combos = clockwork.generate_linear_costlist()

    # init
    energies = []
    coordinates = []
    representations = []
    atoms = []
    costs = []
    n_total = 0

    molobjs = cheminfo.read_sdffile(sdffile[0])
    molobjs = [molobj for molobj in molobjs]
    atoms_list = [cheminfo.molobj_to_atoms(molobj) for molobj in molobjs]

    filenametemplate = filenametemplate[0]

    if molid is None:
        print("hey, select molid")
        quit()

    molobj = molobjs[molid]
    atoms = cheminfo.molobj_to_atoms(molobj)
    n_atoms = len(atoms)

    for combo in combos:

        filename = filenametemplate.format(molid, *combo)

        print(filename, file=sys.stderr)

        energies_next, coordinates_next = read_txt(filename, n_atoms)
        representations_next = [sim.get_representation(atoms, coord) for coord in coordinates_next]

        if len(energies) == 0:
            n_new = len(energies_next)
            energies += energies_next
            coordinates += coordinates_next
            representations += representations_next
            costs += [combo]*n_new
            n_total += n_new
            continue


        idxs = merge_asymmetric(
            atoms,
            energies_next,
            energies,
            representations_next,
            representations)


        n_new = 0
        for i, idxl in enumerate(idxs):

            N = len(idxl)
            if N > 0: continue

            energies.append(energies_next[i])
            coordinates.append(coordinates_next[i])
            representations.append(representations_next[i])
            costs.append(combo)
            n_new += 1


        if debug:
            n_total += n_new
            print(" - new", n_new, file=sys.stderr)
            print("total", n_total, file=sys.stderr)

    # TODO Dump sdf and costs

    return molobj, energies, coordinates, costs



def merge_results_cumulative(sdffile, filenames, debug=True, molid=0):

    # init
    energies = []
    coordinates = []
    representations = []
    atoms = []
    n_total = 0

    molobjs = cheminfo.read_sdffile(sdffile[0])
    molobjs = [molobj for molobj in molobjs]
    atoms_list = [cheminfo.molobj_to_atoms(molobj) for molobj in molobjs]

    for filename in filenames:

        energies_next, coordinates_next, atoms = read_resulttxt(atoms_list, filename)
        representations_next = [sim.get_representation(atoms, coord) for coord in coordinates_next]


        if len(energies) == 0:
            energies += energies_next
            coordinates += coordinates_next
            representations += representations_next
            n_total += len(energies_next)
            continue


        idxs = merge_asymmetric(
            atoms,
            energies_next,
            energies,
            representations_next,
            representations)


        n_new = 0
        for i, idxl in enumerate(idxs):

            N = len(idxl)
            if N > 0: continue

            energies.append(energies_next[i])
            coordinates.append(coordinates_next[i])
            representations.append(representations_next[i])
            n_new += 1


        if debug:
            n_total += n_new
            print(" - new", n_new)
            print("total", n_total)

    return


def merge_sdfs(filenames):

    molobjs = []
    energies = []
    coordinates = []
    representations = []
    atoms = []
    n_total = 0

    for filename in filenames:

        try:
            molobjs_next, energies_next, coordinates_next, representations_next = generate_sdf(filename)
        except:
            continue

        if len(molobjs) == 0:
            atoms, coord = cheminfo.molobj_to_xyz(molobjs_next[0])
            energies += energies_next
            coordinates += coordinates_next
            representations += representations_next
            molobjs += molobjs_next
            n_total += len(molobjs_next)
            continue

        if args.debug:
            print(" {:} = {:} confs".format(filename, len(molobjs_next)))

        idxs = merge_asymmetric(
                atoms,
                energies_next,
                energies,
                representations_next,
                representations)

        n_new = 0
        for i, idxl in enumerate(idxs):

            N = len(idxl)
            if N > 0: continue

            energies.append(energies_next[i])
            coordinates.append(coordinates_next[i])
            representations.append(representations_next[i])
            molobjs.append(molobjs_next[i])
            n_new += 1


        if args.debug:
            n_total += n_new
            print(" - new", n_new)
            print("total", n_total)


    if args.dump:
        sdfstr = [cheminfo.molobj_to_sdfstr(molobj) for molobj in molobjs]
        sdfstr = "".join(sdfstr)
        print(sdfstr)


    return


def dump_txt(energies, coordinates, costs, coord_decimals=DEFAULT_DECIMALS):

    results = []

    for energy, coord, cost in zip(energies, coordinates, costs):
        coord = np.round(coord, coord_decimals).flatten().tolist()
        result = [energy, coord, cost]
        result = json.dumps(result)
        result = result.replace(" ", "")
        results.append(result)

    results = "\n".join(results)

    return results


def dump_sdf(molobj, energies, coordinates, costs):


    hel = molobj.SetProp('_Name', '')

    dumpstr = ""

    for energy, coord, cost in zip(energies, coordinates, costs):

        # Set coordinates
        cheminfo.molobj_set_coordinates(molobj, coord)

        molobj.SetProp('Energy', str(energy))
        molobj.SetProp('Cost', str(cost))

        sdfstr = cheminfo.molobj_to_sdfstr(molobj)

        dumpstr += sdfstr


    print(dumpstr)

    return


def merge_individual(molobjs, filenames, procs=0):

    if procs == 0:
        merge_results_filenames(molobjs, filenames)

    else:
        merge_individual_mp(molobjs, filenames, procs=procs)

    return


def process(q, iolock, func, args, kwargs, debug=True):
    """

    multiprocessing interface for calling

    func(x,*args, **kwargs) with coming from q

    args
        q - queue
        iolock - print lock
        func - function to be called
        args - positional argument
        kwargs - key words args

    """

    while True:

        x = q.get()

        if debug:
            with iolock: print("get", x)

        if x is None: break

        func(x, *args, **kwargs, iolock=iolock)

    return


def spawn(xlist, func, args, kwargs, procs=1):
    """
    spawn processes with func on xlist

    """

    q = mp.Queue(maxsize=procs)
    iolock = mp.Lock()
    pool = mp.Pool(procs, initializer=process, initargs=(q, iolock, func, args, kwargs))

    for x in xlist:

        q.put(x) # halts if queue is full

        if debug:
            with iolock:
                print("put", x)

    for _ in range(procs):
        q.put(None)

    pool.close()
    pool.join()

    # TODO Collect returns from pool

    return


def merge_individual_mp(molobjs, filenames, procs=1, debug=True):

    print("starting {:} procs".format(procs))

    atoms_list = [cheminfo.molobj_to_atoms(molobj) for molobj in molobjs]

    func = merge_results_filename
    args = [atoms_list]
    kwargs = {}

    q = mp.Queue(maxsize=procs)
    iolock = mp.Lock()
    pool = mp.Pool(procs, initializer=process, initargs=(q, iolock, func, args, kwargs))

    for x in filenames:
        q.put(x) # stops if queue is full

        if debug:
            with iolock:
                print("put", x)

    for _ in range(procs):
        q.put(None)

    pool.close()
    pool.join()

    return



def stdin():
    """
    Generator for reading txts from stdin
    """

    while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:

        line = sys.stdin.readline()

        if not line:
            yield from []
            break

        line = line.strip()
        yield line


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version="1.0")

    parser.add_argument('--txt', nargs="+", help='Read results from txt file (require sdf)', metavar="FILE")
    parser.add_argument('--txtfmt', help='format for cost mergeing', metavar="STR")
    parser.add_argument('--sdf', nargs="+", action='store', help='', metavar='FILE')
    parser.add_argument('--sdfstdin', action='store_true', help='Read sdf files from stdin')
    parser.add_argument('--txtstdin', action='store_true', help='Read txt files from stdin')
    parser.add_argument('--molid', action='store', help='What molid from sdf should be used for txt', metavar='int', type=int)

    parser.add_argument('--format', action='store', help='What output format? (sdf, txt)', metavar='str')
    parser.add_argument('--dump', action='store_true', help='dump sdf str to stdout')

    parser.add_argument('--debug', action='store_true', help='debug statements')
    parser.add_argument('-j', '--procs', type=int, help='Merge using multiprocessing', default=0)

    args = parser.parse_args()

    if args.sdf is None:
        print("error: actually we need sdfs to merge")
        quit()


    molobjs = [molobj for molobj in cheminfo.read_sdffile(args.sdf[0])]

    if args.txtstdin:
        filenames = stdin()
    else:
        filenames = [txt for txt in args.txt]

    merge_individual(molobjs, filenames, procs=args.procs)


    quit()

    if args.txt is None:
        merge_sdfs(args.sdf)

    else:
        #TODO Need flags
        # merge_results(args.sdf, args.txt)

        # merge_results_cumulative(args.sdf, args.txt)

        molobj, energies, coordinates, costs = merge_results_cumulative_prime(args.sdf, args.txt, molid=args.molid)

        if args.format == "sdf":
            dump_sdf(molobj, energies, coordinates, costs)
        else:
            out = dump_txt(energies, coordinates, costs)
            print(out)


    return



def main_folder():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version="1.0")
    parser.add_argument('--sdf', nargs="+", action='store', help='', metavar='FILE')
    args = parser.parse_args()

    # TODO Merge results from redis

    if args.sdf is None:
        print("error: actually we need sdfs to merge")
        quit()


    dumpdir = "_tmp_apentane_cum/"

    filename = args.sdf[0] + "{:}_{:}" + ".sdf"

    molobjs, energies, coordinates, representations = generate_sdf(filename.format(1,1))

    atoms, xyz = cheminfo.molobj_to_xyz(molobjs[0])

    # costcombos, costs = clockwork.generate_costlist(total_torsions=28)
    costcombos, costs = clockwork.generate_costlist()

    n_total = len(molobjs)
    molcosts = [(1,1)]*n_total

    print("start", n_total)

    for combo in costcombos[:15]:

        try:
            molobjs_new, energies_new, coordinates_new, representations_new = generate_sdf(filename.format(*combo))
        except:
            continue

        print(" merge", len(molobjs_new))

        idxs = merge_asymmetric(
                atoms,
                energies_new,
                energies,
                representations_new,
                representations)

        n_new = 0
        for i, idxl in enumerate(idxs):

            N = len(idxl)
            if N > 0: continue

            energies.append(energies_new[i])
            coordinates.append(coordinates_new[i])
            representations.append(representations_new[i])
            molobjs.append(molobjs_new[i])

            n_new += 1


        molcosts += [combo]*n_new

        n_total += n_new
        print(" - new", n_new)
        print("total", n_total, combo)



    sdfstr = [cheminfo.molobj_to_sdfstr(molobj) for molobj in molobjs]
    sdfstr = "".join(sdfstr)
    f = open(dumpdir+"all.sdf", 'w')
    f.write(sdfstr)
    f.close()

    hellodump = ""
    for combo in molcosts:
        hello = "{:} {:}".format(*combo)
        hellodump += hello+"\n"

    f = open(dumpdir+"costs.csv", 'w')
    f.write(hellodump)
    f.close()

    plt.plot(energies, 'k.')
    plt.yscale("log")
    plt.savefig(dumpdir+"energies")


    return



def main_test():
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

    tstart = time.time()

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

    tend = time.time()

    print("total {:5.2f} sec".format(tend-tstart))

    return

if __name__ == '__main__':
    main()
