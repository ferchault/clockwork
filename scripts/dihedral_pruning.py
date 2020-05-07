import json
from itertools import combinations

import numpy as np


class Pruning:
    def __init__(self, fname):
        self.read_json(fname)
        self.support = {}

    def read_json(self, fname):
        with open(fname) as infile:
            merged = json.load(infile)
        self.dihedral_occ = merged['dihedral_occ']
        self.n_confs = len(merged['conformers'])

        conformer_list = []
        for i in self.dihedral_occ:
            dihed = self.dihedral_occ[i]
            if dihed:
                for calc in dihed:
                    conformer_list.append(calc)
        self.conformer_list = conformer_list

    def calc_support(self, nbody, support_thresh=None):
        if nbody == 1:
            dihedrals = [item for sublist in self.conformer_list for item in sublist]
            dihed_id, count = np.unique(dihedrals, return_counts=True)
            support = count/self.n_confs
            # you could directly filter out dihedrals based on support
            if not support_thresh:
                self.support[nbody] = np.vstack((dihed_id, support)).swapaxes(1, 0)
        else:
            if nbody - 1 == 1:
                dihedrals = self.support[nbody-1][:, 0].astype('int')
            else:
                dihedrals = np.unique([item for sublist in self.support[nbody-1][:, 0] for item in sublist])
            nbody_count = {}
            nbody_dihedrals = combinations(dihedrals, nbody)
            for dihed_combination in nbody_dihedrals:
                for dihedral_conformers in self.conformer_list:
                    if set(dihed_combination).issubset(set(dihedral_conformers)):
                        if dihed_combination not in nbody_count:
                            nbody_count[dihed_combination] = 0
                        nbody_count[dihed_combination] += 1
            self.support[nbody] = np.array([[dihedral, nbody_count[dihedral]/self.n_confs] for dihedral in nbody_count])


if __name__ == '__main__':
    p = Pruning('/Users/c0uch1/work/clockwork_data/merge_files/ci-0234.merged')
    p.calc_support(1)
    p.calc_support(2)
    p.calc_support(3)