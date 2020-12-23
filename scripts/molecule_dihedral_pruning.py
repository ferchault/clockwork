import sys
import os
from glob import glob
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dihedral_pruning import load_merge_data, calculate_pruning_cost, plotting


def pruning_list(prune_data, cutoff, batch_num, ntotal=None):
    if batch_num <= 2:
        idx = np.where(np.array(prune_data['nconfs']) / max(prune_data['nconfs']) >= cutoff)[0][-1]
    else:
        conf_cut = round(ntotal * (1. - cutoff))
        idx = np.where(np.array(prune_data['nconfs']) >= max(prune_data['nconfs']) - conf_cut)[0][-1]

    total_diheds = max(prune_data["ndiheds"])
    diheds_pruned = total_diheds - prune_data["ndiheds"][idx]

    print(f'Conformer cutoff {cutoff} found at {prune_data["ldc"][idx]} computations. IDX {idx}')
    print(f'This prunes {diheds_pruned} dihedrals ({(diheds_pruned / total_diheds) * 100:.02f}%).')

    keep_list = {}
    for dih, mname in prune_data['dihedrals'][idx:]:
        if mname not in keep_list.keys():
            keep_list[mname] = []
        keep_list[mname].append(dih)

    prune_list = {}
    for dih, mname in prune_data['dihedrals'][:idx]:
        if mname not in prune_list.keys():
            prune_list[mname] = []
        prune_list[mname].append(dih)

    return keep_list, prune_list, prune_data["ldc"][idx]


def prune_all(merge_files, batch, outpath, past_pruning=None, cutoff=0.99):
    pm_prune_dict = {}
    keep_dih_dict = {}
    cutoffs = []
    mol_data = {}
    for m in merge_files:
        m_info = load_merge_data([m], batch_num=batch)

        p_info = calculate_pruning_cost(m_info, batch, past_pruning)
        mol_data[m] = p_info

        print(f'Newly found conformers: {max(p_info["nconfs"])}')
        print(f'Total amount of conformers: {m_info["ntotal"]}')
        keep_dih_list, prune_dih_list, comp_cutoff = pruning_list(p_info, cutoff,
                                                                  batch_num=batch,
                                                                  ntotal=m_info['ntotal'])

        for mol in prune_dih_list.keys():
            pm_prune_dict[mol] = prune_dih_list[mol]

        for mol in keep_dih_list.keys():
            keep_dih_dict[mol] = keep_dih_list[mol]

        cutoffs.append([comp_cutoff, m_info["ntotal"], max(p_info["nconfs"])])

    os.makedirs(f'{outpath}/batch{batch}', exist_ok=True)
    os.makedirs(f'{outpath}/batch{batch}/keep_dihedrals_pm', exist_ok=True)

    np.save(f'{outpath}/batch{batch}/batch_{batch}_molecule_cutoffs.npy', cutoffs)
    pickle.dump(pm_prune_dict, open(f'{outpath}/batch{batch}/pruned_dihedrals_cutoff-{cutoff}_pm.pkl', 'wb'))
    pickle.dump(keep_dih_dict, open(f'{outpath}/batch{batch}/keep_dihedrals_cutoff-{cutoff}pm.pkl', 'wb'))

    for mol in keep_dih_dict:
        keep_dih_dict[mol].sort(key=int)
        with open(f'{outpath}/batch{batch}/keep_dihedrals_pm/{mol}.dih', 'w') as outfile:
            outfile.write('-'.join(keep_dih_dict[mol]))


def mol_pruning(merge_file, batch, outpath, n_diheds=None, cutoff=0.99, save=True):
    m_info = load_merge_data([merge_file], batch_num=batch)
    mname = os.path.basename(outpath)
    p_info = calculate_pruning_cost(m_info, batch, n_diheds)
    print(f'Newly found conformers: {max(p_info["nconfs"])}')
    print(f'Total amount of conformers: {m_info["ntotal"]}')
    keep_dih_list, prune_dih_list, comp_cutoff = pruning_list(p_info, cutoff,
                                                              batch_num=batch,
                                                              ntotal=m_info['ntotal'])

    mol_info = {'keep_dihed': keep_dih_list,
                'prune_dihed': prune_dih_list,
                'cutoff': comp_cutoff,
                'new_confs': max(p_info["nconfs"]),
                'prune_info': p_info
                }

    if save:
        pickle.dump(mol_info, open(f'{outpath}/batch{batch}_prune_info.pkl', 'wb'))
        if max(p_info["nconfs"]) >= 1:
            sorted_diheds = np.sort(np.array(keep_dih_list[mname]).astype(int))
            with open(f'{outpath}/batch{batch}.dihedrals', 'w') as outfile:
                outfile.write('-'.join(sorted_diheds.astype(str)))
        else:
            Path(f'{outpath}/batch{batch}.dihedrals').touch(mode=0o770)
    return mol_info


if __name__ == '__main__':
    data_path, batchnr = sys.argv[1:]

    datapath = f'{data_path}/molecules/'
    mergefiles = sorted(glob(f'{data_path}/molecule_merge/batch{batchnr}/*.merged'))
    prune_infos = []
    for m in tqdm(mergefiles):
        molname = os.path.basename(m).split('.')[0]
        outp = f'{datapath}/{molname}'
        if os.path.isfile(f'{outp}/batch{int(batchnr)-1}.dihedrals'):
            with open(f'{outp}/batch{int(batchnr)-1}.dihedrals', 'r') as infile:
                _n_diheds = len(infile.readlines()[0].split('-'))
        else:
            _n_diheds = None
        prune_infos.append(mol_pruning(m, batchnr, outp, n_diheds=_n_diheds, cutoff=0.99, save=True))
    #pickle.dump(prune_infos, open(f'batch{batchnr}_prune_info.pkl', 'wb'))
