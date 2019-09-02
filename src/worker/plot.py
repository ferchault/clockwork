import numpy as np

from chemhelp import cheminfo
import worker
import merge

import matplotlib.pyplot as plt


def get_energies(sdffile):

    suppl = cheminfo.read_sdffile(sdffile)

    molobjs = []
    energies = []

    f = open("_test_minuncov.sdf", 'w')
    for molobj in suppl:

        # energy = worker.get_energy(molobj)

        # prop, ff = worker.get_forcefield(molobj)
        # status = worker.run_forcefield(ff, 500)

        energy = worker.get_energy(molobj)
        molobjs.append(molobj)
        energies.append(energy)

        if energy > 100:
            sdf = cheminfo.save_molobj(molobj)
            f.write(sdf)

    f.close()
    return molobjs, energies


def energy_histogram(energies):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_bins = None

    # energies = np.log(energies)

    energies = np.array(energies)

    # idxs, = np.where(energies < 20)
    # energies = energies[idxs]

    # ax.set_xscale('log')
    ax.hist(energies)

    fig.savefig("_tmp_energy_his")


    return


def count_histogram(costs):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    costs = np.array(costs)
    costs = costs.T


    ghis = ax.hist2d(costs[0,:], costs[1,:], bins=[4,4],
        cmap='PuRd' )


    # fig.colorbar(ghis, cax=cax, orientation='horizontal')

    fig.savefig("_tmp_conf_his")

    print(costs)


    return


def analyse_results(molobj, txtfile):


    ffprop, forcefield = worker.get_forcefield(molobj)


    coord = cheminfo.molobj_get_coordinates(molobj)
    grad = forcefield.CalcGrad()
    print(grad)
    print(coord)
    print()

    worker.run_forcefield(forcefield, 500)
    grad = forcefield.CalcGrad()
    # print(grad)
    # print(coord)
    # print()
    # print()

    coord = cheminfo.molobj_get_coordinates(molobj)

    atoms = cheminfo.molobj_to_atoms(molobj)
    n_atoms = len(atoms)

    energies, coordinates, costs = merge.read_txt(txtfile, n_atoms)
    # costs = [0]*len(energies)


    # energies = np.round(energies, 1)
    # print(np.unique(energies))
    # quit()
    #
    # pick = 79
    #
    # print(energies[pick])
    # coord = coordinates[pick]
    #
    # cheminfo.molobj_set_coordinates(molobj, coord)
    # energy = worker.get_energy(molobj)
    # grad = forcefield.CalcGrad()
    # grad_norm = np.array(grad)
    # grad_norm = np.abs(grad_norm)
    # grad_norm = grad_norm.mean()
    # print(energy, grad_norm)
    #
    # # opt
    # status = worker.run_forcefield(forcefield, 5, force=1e-6)
    #
    # print(status)
    #
    # energy = worker.get_energy(molobj)
    # grad = forcefield.CalcGrad()
    # grad_norm = np.array(grad)
    # grad_norm = np.abs(grad_norm)
    # grad_norm = grad_norm.mean()
    #
    # print(energy, grad_norm)
    #
    # merge.dump_sdf(molobj, energies, coordinates, costs)

    energy_histogram(energies)
    count_histogram(costs)




    return


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version="1.0")
    parser.add_argument('--sdf', nargs="+", type=str, help='SDF file', metavar='file')

    parser.add_argument('--txt', type=str, help='txt result file', metavar='file')
    parser.add_argument('--molidx', type=int, help='index in sdf for result', metavar='int')

    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()

    if "~" in args.sdf:
        args.sdf = correct_userpath(args.sdf)

    if "~" in args.txt:
        args.txt = correct_userpath(args.txt)

    offset = 0

    if args.txt is None:
        for sdffile in args.sdf:

            molobjs, energies = get_energies(sdffile)
            energies = np.asarray(energies)
            x_axis = np.arange(len(energies), dtype=int)

            idx, = np.where(energies > 100)

            if len(idx) > 0:
                print("large energy", sdffile, idx, energies[idx])

            plt.plot(x_axis+offset, energies, '.')

            offset += 1

        # plt.ylim([0, 50])
        plt.savefig("_tmp_energies")
        quit()


    # do txt
    molobjs = cheminfo.read_sdffile(args.sdf[0])
    molobjs = [molobj for molobj in molobjs]
    molobj = molobjs[args.molidx]

    analyse_results(molobj, args.txt)

    return


if __name__ == '__main__':
    main()
