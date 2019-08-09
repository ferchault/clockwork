import numpy as np

from chemhelp import cheminfo
import worker

import matplotlib.pyplot as plt


def get_energies(sdffile):

    suppl = cheminfo.read_sdffile(sdffile)

    molobjs = []
    energies = []

    f = open("_test_minuncov.sdf", 'w')
    for molobj in suppl:

        # energy = worker.get_energy(molobj)

        prop, ff = worker.get_forcefield(molobj)
        status = worker.run_forcefield(ff, 500)


        energy = worker.get_energy(molobj)
        molobjs.append(molobj)
        energies.append(energy)

        if energy > 100:
            sdf = cheminfo.save_molobj(molobj)
            f.write(sdf)

    f.close()
    return molobjs, energies


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version="1.0")
    parser.add_argument('--sdf', nargs="+", type=str, help='SDF file', metavar='file')

    args = parser.parse_args()

    if "~" in args.sdf:
        args.sdf = correct_userpath(args.sdf)


    offset = 0

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


    return


if __name__ == '__main__':
    main()
