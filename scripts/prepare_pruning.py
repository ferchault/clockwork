import sys
import os
from glob import glob
import pickle

"""
Creates a master pruning dictionary that contains molecule name and dihedral number of all dihedrals that have been
pruned so far.
Also creates empty .dih files as a preparation for the pruning so that the molecule list stays complete after pruning.
(Empty file means converged molecule)
"""


batch_num = sys.argv[1]
data_path = '/mnt/FOXTROT/CONFORMER/merge_files/merge_info/'
molecules = glob(f'{data_path}/batch2/keep_dihedrals/*.dih')
prune_files = sorted(glob(f'{data_path}/*/pruned_dihedrals_cutoff-0.99.pkl'))

outpath = f'{data_path}/batch{batch_num}/keep_dihedrals/'


full_prune = pickle.load(open(prune_files[0], 'rb'))

for pfile in prune_files[1:]:
    batch_prune = pickle.load(open(pfile, 'rb'))

    for mol in batch_prune.keys():
        if mol in full_prune.keys():
            full_prune[mol].extend(batch_prune[mol])
        else:
            full_prune[mol] = batch_prune[mol]

c = 0
for mol in full_prune.keys():
    c += len(full_prune[mol])
print('Total number of dihedrals pruned', c)

pickle.dump(full_prune, open(f'{data_path}/master_prune.pkl', 'wb'))


os.makedirs(outpath, exist_ok=True)
for mol in molecules:
    open(f'{outpath}/{os.path.basename(mol)}', 'a').close()

