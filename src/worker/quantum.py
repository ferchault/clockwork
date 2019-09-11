import os

import worker
import merge
import easyusage

from chemhelp import cheminfo
from chemhelp.chemistry import mopac
from chemhelp import xyz2mol

import rdkit.Chem as Chem

def get_molobj(atoms, coord):

    # for rdkit
    atoms = [int(atom) for atom in atoms]

    quick = True
    charged_fragments = True
    mol = xyz2mol.xyz2mol(atoms, 0, coord, charged_fragments, quick)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
    m = Chem.MolFromSmiles(smiles)

    return m

def get_smiles(atoms, coord):

    m = get_molobj(atoms, coord)
    smi = Chem.MolToSmiles(m, isomericSmiles=False)

    return smi


def optmize_conformation(atoms, coord, filename=None):
    """

    """

    if filename is None:

        # TODO Get TMPDIR, if running in SLURM, otherwise _tmp_mopac_
        # TMPDIR
        filename = "_tmp_mopac_opt_"

    parameters = {
        "method": "PM6",
        "keywords": "precise"
    }

    properties = mopac.calculate(atoms, coord, parameters=parameters, write_only=False, label=filename)

    oenergy = properties["h"]
    ocoord = properties["coord"]

    del properties

    return oenergy, ocoord


def parse_results(molidx, readtemplate, molobjs, dump_results=None, debug=True, **kwargs):
    """
    """

    if debug:

        filename = dump_results.format(molidx)

        if os.path.exists(filename):
            print("exists", molidx)
            return

        print("parsing", molidx)

    filename = readtemplate.format(molidx)
    molobj = molobjs[molidx]

    reference_smiles = cheminfo.molobj_to_smiles(molobj, remove_hs=True)

    atoms = cheminfo.molobj_to_atoms(molobj)
    n_atoms = len(atoms)

    energies, coordinates, costs = merge.read_txt(filename, n_atoms)

    oenergies = []
    ocoordinates = []
    ocosts = []

    for i, energy, coord, cost in zip(range(len(energies)), energies, coordinates, costs):

        filename="_tmp_mopac_/_" + str(molidx) + "-" +str(i)+ "_"

        try:
            oenergy, ocoord = optmize_conformation(atoms, coord, filename=filename)
        except:
            print("unconverged", filename)
            continue

        m = get_molobj(atoms, ocoord)
        smiles = cheminfo.molobj_to_smiles(m)

        same_graph = (smiles == reference_smiles)

        if same_graph:
            oenergies.append(oenergy)
            ocoordinates.append(ocoord)
            ocosts.append(cost)

        # print(smiles == reference_smiles, "{:5.2f}".format(energy), "{:5.2f}".format(oenergy), cost)


    idxs = merge.merge_cost(atoms, oenergies, ocoordinates, ocosts)

    renergies = []
    rcoords = []
    rcosts = []

    for idx in idxs:

        energy = oenergies[idx]
        coord = ocoordinates[idx]
        cost = ocosts[idx]

        renergies.append(energy)
        rcoords.append(coord)
        rcosts.append(cost)


    if dump_results is not None:

        out = merge.dump_txt(renergies, rcoords, rcosts)

        filename = dump_results.format(molidx)
        f = open(filename, 'w')
        f.write(out)
        f.close()

    return renergies, rcoords, rcosts


def parallel_parse_results(readtemplate, molobjs, molidxs, writetemplate, procs=1):

    easyusage.parallel(molidxs,
        parse_results,
        [readtemplate, molobjs],
        {"dump_results": writetemplate},
        procs=procs)

    return


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version="1.0")


    parser.add_argument('--method', action='store', help='What QM program to use', metavar='str', default="PM6")

    parser.add_argument('--txtfmt', help='format for cost mergeing', metavar="STR")
    parser.add_argument('--sdf', nargs="+", action='store', help='', metavar='FILE')
    parser.add_argument('--molid', action='store', help='What molid from sdf should be used for txt', metavar='int(s)')

    parser.add_argument('--debug', action='store_true', help='debug statements')

    parser.add_argument('-j', '--procs', type=int, help='Use multiple processes (default=0)', default=0)

    args = parser.parse_args()

    if args.sdf is None:
        print("error: actually we need sdfs information")
        quit()

    molidxs = args.molid
    molidxs = molidxs.split("-")
    if len(molidxs) == 1:
        molidxs = [int(molidx) for molidx in molidxs]
    else:
        molidxs = [int(molidx) for molidx in molidxs]
        molidxs = range(molidxs[0], molidxs[1]+1)

    # molobj db
    molobjs = [molobj for molobj in cheminfo.read_sdffile(args.sdf[0])]

    dump_results = "_tmp_results_data1/{:}.results"

    if args.procs > 0:
        parallel_parse_results(args.txtfmt, molobjs, molidxs, dump_results, procs=args.procs)
        return

    for idx in molidxs:
        parse_results(idx, args.txtfmt, molobjs, dump_results="_tmp_results_data1/{:}.results".format(args.molid))

    return


if __name__ == "__main__":
    main()
