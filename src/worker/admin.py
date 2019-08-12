
from chemhelp import cheminfo



def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version="1.0")
    parser.add_argument('--sdf', type=str, help='SDF file', metavar='file', default="~/db/qm9s.sdf.gz")
    parser.add_argument('--debug', action="store_true", help="", default=False)

    args = parser.parse_args()

    max_mols = 500

    molobjs = cheminfo.read_sdffile(args.sdf)

    tordb = []
    for i, molobj in enumerate(molobjs):
        # if i > max_mols: break
        torsions = cheminfo.get_torsions(molobj)
        tordb.append(torsions)

    for i, torsions in enumerate(tordb):

        torsions_str = []
        for torsion in torsions:
            fmt = " ".join(["{:}"]*4).format(*torsion)
            torsions_str.append(fmt)


        state = ",".join(torsions_str)

        print(i, ":", state)

    return


if __name__ == '__main__':
    main()
