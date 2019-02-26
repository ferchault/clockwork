
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalForceFields


def load_sdf(filename):
    """

    load sdf file and return rdkit mol list

    args:
        filename sdf

    return:
        list of mol objs

    """

    suppl = Chem.SDMolSupplier(filename,
                               removeHs=False,
                               sanitize=True)

    mollist = [mol for mol in suppl]

    return mollist


def load_sdf_file(obj):

    suppl = Chem.ForwardSDMolSupplier(obj,
                               removeHs=False,
                               sanitize=True)

    mollist = [mol for mol in suppl]

    return mollist



