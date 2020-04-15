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

import numpy as np
import tqdm

import qml
from qml.kernels import get_global_kernel
from qml.representations import generate_fchl_acsf

# project-specific paramters. Do not change without changing executor2 worker.
QML_FCHL_SIGMA = 2
QML_FCHL_THRESHOLD = 0.98
ENERGY_THRESHOLD = 1e-4
WBO_THRESHOLD = 5e-2  # TODO: this needs adjustments, sth between 1e-1 - 1e-2


class Merger(object):
    def __init__(self, merge_into_file, workpackage_file):
        self._read_merge_into_file(merge_into_file, workpackage_file)
        self._init_caches()
        self._merge_workpackages(workpackage_file)

    def _merge_workpackages(self, workpackage_file):
        """ Read file with multiple workpackages."""
        with open(workpackage_file) as fh:
            workpackages = fh.readlines()
        for wp in tqdm.tqdm(workpackages, desc="Merging workpackages"):
            wp = json.loads(wp)
            self._consume_workpackage(wp)

    def _consume_workpackage(self, workpackage):
        """ Insert non-duplicates of the workpackage into the database."""
        if workpackage['mol'] != self._dataset['molname']:
            raise ValueError("Different molecules in the two files.")

        for geometry, energy, bond_orders in zip(workpackage['geo'], workpackage['ene'], workpackage['wbo']):
            # convert string geometry
            atomlines = geometry.split("\n")
            coordinates = np.array([[float(__) for __ in _.split()[1:]] for _ in atomlines])

            # check for duplicates
            prescreened = self._find_compatible(geometry, energy, bond_orders)
            if prescreened is True or not self._is_duplicate(prescreened, coordinates):
                conformer = {'ene': energy, 'geo': list(coordinates.flatten()), 'wbo': bond_orders,
                             'dih': workpackage['dih'], 'res': workpackage['res']}
                self._dataset['conformers'].append(conformer)

    def _find_compatible(self, geometry, energy, bond_orders):
        """ Returns a list of potentially compatible conformers."""
        nconfs = len(self._dataset['conformers'])

        compatible = []
        for confid in range(nconfs):
            if abs(self._dataset['conformers'][confid]['ene'] - energy) < ENERGY_THRESHOLD:
                if (np.array(bond_orders) - np.array(self._dataset['conformers'][confid]['wbo'])).sum() > WBO_THRESHOLD:
                    return True
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
        return np.max(sim) > QML_FCHL_THRESHOLD

    def _init_caches(self):
        """ Lazily builds caches of result conformers."""
        self._rep_cache = {}
        self._charges = np.array([{'H': 1, 'C': 6, 'N': 7, 'O': 8}[_] for _ in self._dataset['charges']])

    def _init_database(self, workpackage_file):
        with open(workpackage_file) as fh:
            workpackages = fh.readlines()
        wp = json.loads(workpackages[0])
        charges = [i for i in wp['geo'][0].split() if i.isalpha()]
        self._dataset = {'molname': wp['mol'],
                         'charges': ''.join(charges),
                         'conformers': []}

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


if __name__ == '__main__':
    merge_into_filename, workpackage_filename = sys.argv[1:]
    m = Merger(merge_into_filename, workpackage_filename)
    basename = os.path.basename(merge_into_filename).split('.')[0]
    m.save(f'{basename}.merged')
