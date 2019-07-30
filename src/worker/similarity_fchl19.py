
import qml

from qml.kernels import get_global_kernel
from qml.representations import generate_fchl_acsf


def get_representation(atoms, coordinates, **kwargs):
    """

    atoms
    coordinates
    max_atoms

    """

    max_atoms = kwargs.get("max_atoms", len(atoms))
    rep = generate_fchl_acsf(atoms, coordinates, pad=max_atoms)

    return rep


def get_kernel(rep_x, rep_y, atoms_x, atoms_y, **kwargs):

    sigma = kwargs.get("sigma", 4.0)

    kernel = get_global_kernel(rep_x, rep_y,  atoms_x, atoms_y,  sigma)

    # TODO ouch, should be transformed
    kernel = kernel.T

    return kernel



