#!/usr/bin/env python
"""
Merges two unique lists of JSON conformer search results. Uses heuristics to reduce the number of required comparisons.

Assumptions:
 - both files only contain data from the same molecule.
 - all geometries have the same element ordering (but not necessarily the same atom-atom mapping between conformers)

Json1file contains a list of conformers with the following info:
{
	'molname': 'string', 
	'charges': 'string', 
	'conformers': [{
		'ene': 'float', # xTB energy
		'geo': 'list of floats', # geometry
		'wbo': 'list of floats', # xTB bond orders
		'dih': 'list of ints', # dihedrals of the workpackage it was first found in
		'res': 'int', # resolution of the workpackage it was first found in
	}]
}

Therefore, the minimum file contents are along the lines of
{"molname": "something", "charges": "OHH", "conformers": []}

Usage:
   merge.py json1file json2file
"""
import json
import sys
import os
import pickle
import argparse

import numpy as np
import tqdm

import qml
from qml.kernels import get_global_kernel
from qml.representations import generate_fchl_acsf

# project-specific paramters. Do not change without changing executor2 worker.
QML_FCHL_SIGMA = 2
QML_FCHL_THRESHOLD = 0.98
ENERGY_THRESHOLD = 1e-4

# (nbody, res), defined by costorder.py
costorder = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (3, 0), (2, 3), (3, 1), (4, 0), (2, 4),
             (3, 2), (5, 0), (4, 1), (6, 0), (3, 3), (5, 1), (4, 2), (3, 4), (6, 1), (4, 3), (5, 2), (4, 4), (6, 2),
             (5, 3), (5, 4), (6, 3), (6, 4)]


class Merger(object):
    def __init__(self, merge_into_file, workpackage_file, keep_dihedrals=None, similarity_check=None):
        self._check_workpackage_status(workpackage_file)
        self._read_merge_into_file(merge_into_file, workpackage_file)
        self._init_caches(keep_dihedrals, similarity_check)
        self._merge_workpackages(workpackage_file)

    def _check_workpackage_status(self, workpackage_file):
        with open(workpackage_file) as fh:
            workpackages = fh.readlines()
        for line in workpackages:
            if line.startswith('JOB:'):
                raise ValueError('Workpackage is not finished yet.')

    def _merge_workpackages(self, workpackage_file):
        """ Read file with multiple workpackages."""
        with open(workpackage_file) as fh:
            workpackages = fh.readlines()
        for i, wp in tqdm.tqdm(enumerate(workpackages), desc="Merging workpackages", total=len(workpackages)):
            wp = json.loads(wp)
            if self.keep_dihedrals is None or set(wp['dih']).issubset(self.keep_dihedrals):
                self._consume_workpackage(wp, i)

    def _consume_workpackage(self, workpackage, wp_id):
        """ Insert non-duplicates of the workpackage into the database."""
        if workpackage['mol'] != self._dataset['molname']:
            raise ValueError("Different molecules in the two files.")

        contrib = False  # only switches to True if a new conformer is found
        sim_id = None
        for geometry, energy in zip(workpackage['geo'], workpackage['ene']):
            # convert string geometry
            atomlines = geometry.split("\n")
            coordinates = np.array([[float(__) for __ in _.split()[1:]] for _ in atomlines])
            dih = workpackage['dih']
            res = workpackage['res']

            # check for duplicates
            # prescreened = self._find_compatible(energy) # energy pre-filtering not done anymore
            prescreened = np.arange(len(self._dataset['conformers']))
            if self.similarity_check is not None and self.similarity_check[wp_id][0] is False:
                self._dataset['conformers'][self.similarity_check[wp_id][1]]['same_conf'].append([len(dih), res, dih])
                continue
            else:
                similarity = self._is_duplicate(prescreened, coordinates)
            if not np.max(similarity) > QML_FCHL_THRESHOLD or similarity is False:
                conformer = {'ene': energy,
                             'geo': list(coordinates.flatten()),
                             'dih': dih, 'res': res,
                             'same_conf': []}
                self._dataset['conf_contrib'][str((len(dih), res))].append(dih)
                self._dataset['conformers'].append(conformer)
                contrib = True
            else:
                if similarity is not False:
                    sim_id = np.argmax(similarity)
                    self._dataset['conformers'][sim_id]['same_conf'].append([len(dih), res, dih])
        self._conformer_contribution.append([contrib, sim_id])
        self._calc_statistics(workpackage['dih'], contrib=contrib)

    def _calc_statistics(self, dihedrals, contrib=False):
        for dihedral in dihedrals:
            dihedral = str(dihedral)
            if dihedral not in self._dataset['dihed_calcs'].keys() or dihedral not in self._dataset['last_dihed_contrib'].keys():
                self._dataset['dihed_calcs'][str(dihedral)] = 1
                self._dataset['last_dihed_contrib'][dihedral] = 1
            else:
                self._dataset['dihed_calcs'][str(dihedral)] += 1
                self._dataset['last_dihed_contrib'][dihedral] += 1 if not contrib else - self._dataset['last_dihed_contrib'][dihedral]
        self._dataset['total_calcs'] += 1

    def _find_compatible(self, energy):
        """ Returns a list of potentially compatible conformers."""
        nconfs = len(self._dataset['conformers'])

        compatible = []
        for confid in range(nconfs):
            if abs(self._dataset['conformers'][confid]['ene'] - energy) < ENERGY_THRESHOLD:
                compatible.append(confid)
        return compatible

    def _get_rep(self, confid):
        """ Lazily build representations for result conformers."""
        if confid not in self._rep_cache:
            coords = np.array(self._dataset['conformers'][confid]['geo']).reshape(-1, 3)
            self._rep_cache[confid] = generate_fchl_acsf(self._charges, coords, pad=len(self._charges))

        return self._rep_cache[confid]

    def _is_duplicate(self, haystack, needle):
        """ Accurate, yet expensive comparison operation. Checks for equivalents of the geometry needle in the list of conformers haystack."""
        rep = generate_fchl_acsf(self._charges, needle, pad=len(self._charges))
        reps = [self._get_rep(confid) for confid in haystack]
        if len(reps) == 0:
            return False
        sim = get_global_kernel(np.array([rep]), np.array(reps), np.array([self._charges]),
                                np.array([list(self._charges)] * len(reps)), QML_FCHL_SIGMA)
        return sim

    def _init_caches(self, keep_dihedrals, similarity_check):
        """ Lazily builds caches of result conformers."""
        self._rep_cache = {}
        self._charges = np.array([{'H': 1, 'C': 6, 'N': 7, 'O': 8}[_] for _ in self._dataset['charges']])
        self._conformer_contribution = []
        if keep_dihedrals:
            _keep_dihedrals = pickle.load(open(keep_dihedrals, 'rb'))
            self.keep_dihedrals = set([int(k) for k in _keep_dihedrals[self._dataset['molname']]])
        else:
            self.keep_dihedrals = None
        self.similarity_check = similarity_check if not None else None

    def _init_database(self, workpackage_file):
        with open(workpackage_file) as fh:
            workpackages = fh.readlines()
        for workpackage in workpackages:
            # Sometime workpackages are empty/have failed and we need to iterate until we get a successfull one
            wp = json.loads(workpackage)
            if wp['geo']:
                break
            else:
                continue
        charges = [i for i in wp['geo'][0].split() if i.isalpha()]
        self._dataset = {'molname': wp['mol'],
                         'charges': ''.join(charges),
                         'conformers': [],
                         'conf_contrib': {str(k): [] for k in costorder},
                         'total_calcs': 0,
                         'last_dihed_contrib': {},
                         'dihed_calcs': {}}

    def _read_merge_into_file(self, merge_into_file, workpackage_file):
        """ Parses JSON result file."""
        if os.path.isfile(merge_into_file):
            with open(merge_into_file) as fh:
                self._dataset = json.load(fh)
        else:
            self._init_database(workpackage_file)

        print(f"Read a database with {len(self._dataset['conformers'])} conformers")

    def save(self, outputfile):
        """ Dumps JSON result file."""
        with open(outputfile, 'w') as fh:
            json.dump(self._dataset, fh)
        print(f"Saved a database with {len(self._dataset['conformers'])} conformers")

    def get_contribution(self):
        return self._conformer_contribution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--merge_into_filename', type=str, help='Merge file from a previous batch')
    parser.add_argument('--workpackage', type=str, help='Clockwork workpackage file.')
    parser.add_argument('--keep_dihedrals', type=str, help='Output Directory', default=None)
    parser.add_argument('--outmerge', type=str, help='Output merge file')
    args = parser.parse_args()
    m = Merger(args.merge_into_filename, args.workpackage, args.keep_dihedrals)
    m.save(args.outmerge)
