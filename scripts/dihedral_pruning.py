import os
import pickle
import glob
import json
import itertools as it

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.special as scs

wp_list = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (3, 0), (2, 3), (3, 1), (4, 0), (2, 4),
           (3, 2), (5, 0), (4, 1)]


def configuration_count(resolution):
    if resolution == 0:
        return 1
    return 2 ** (resolution - 1)


def combinations_in_group(T, R):
    combinations = []
    for case in it.product(range(R + 1), repeat=T):
        if R in case:
            combinations.append(case)
    return combinations


def group_cost(combinations):
    cost = 0
    for case in combinations:
        cost += np.prod(list(map(configuration_count, case)))
    return cost


def cost_saved(n_diheds_, start_wp_idx):
    cost_saved = 0
    for j in range(start_wp_idx, len(wp_list)):
        nbody, res = wp_list[j]
        cost_saved += scs.binom(n_diheds_, nbody) * group_cost(combinations_in_group(nbody, res))
    return cost_saved * 0.2 / 60 / 60  # 0.2s per calculations scaled to hours


def get_last_wp(mol, dihed):
    dih_contribs = []
    for conf in mol['conformers']:
        n, r = conf['dih'], conf['res']
        if dihed in n:
            dih_contribs.append((len(n), r))
    idx = sorted([wp_list.index(i) for i in dih_contribs])
    return idx[-1]  # returns the highest number, so the last workpackage it was relevant


def load_merge_data(merge_files):
    merge_info = {'ldc': [],
                  'nconfs': 0,
                  'mols': {}}

    for mfile in tqdm(merge_files):
        with open(mfile) as infile:
            m = json.load(infile)

        merge_info['nconfs'] += len(m['conformers'])

        for d in m['last_dihed_contrib'].keys():
            merge_info['ldc'].append([m['last_dihed_contrib'][d], d, m['molname']])

        merge_info['mols'][m['molname']] = {'n_confs': len(m['conformers']),
                                            'n_diheds': len(m['last_dihed_contrib'].keys()),
                                            'duplicates': []}

        for conf in m['conformers']:
            # Create a list of all duplicate configurations
            duplicates = [(dupl[0], dupl[1], tuple(dupl[2])) for dupl in conf['same_conf']]
            # Don't forget to append the first/original one
            duplicates.append((1, conf['res'], tuple(conf['dih'])))
            # Make it a set for faster comparison
            merge_info['mols'][m['molname']]['duplicates'].append(set(duplicates))

    return merge_info


def calculate_pruning_cost(merge_info):
    # Sort last dihedral contribution (ldc) in descending order
    last_contribs_idx = np.argsort(-np.array(merge_info['ldc'])[:, 0].astype(float))
    ldc = np.array(merge_info['ldc'])[last_contribs_idx]

    total_cost = 0
    for mname in merge_info['mols']:
        # Calculate total CPU cost so far
        total_cost += cost_saved(merge_info['mols'][mname]['n_diheds'], 0)

    prune_data = {'cost': [],
                  'ndiheds': [],
                  'nconfs': [],
                  'ldc': [],
                  'dihedrals': []}

    total_ndiheds = len(ldc)
    conf_counter = merge_info['nconfs']
    cost_count = total_cost
    for i in tqdm(range(len(ldc))):
        # last contribution, dihedral ID, molecule name
        lc, dih, mname = ldc[i]
        last_wp = 0

        # Loop over all conformers of a molecule
        for conf in merge_info['mols'][mname]['duplicates']:
            discard_list = []
            # Loop over every duplicate configurations of a conformer
            for dupl in conf:
                # if the dihedral is in one one the configurations, it gets discarded
                if int(dih) in dupl[2]:
                    discard_list.append(dupl)

            if discard_list:
                # Determine the ID of the last workpackage
                idx = sorted([wp_list.index(nbres[:2]) for nbres in discard_list])[-1]
                last_wp = idx if idx > last_wp else last_wp

                for d in discard_list:
                    # Discard every configurations that contained this dihedral
                    conf.discard(d)

            if not conf:
                # if the duplicate list is empty, the conformer gets removed
                merge_info['mols'][mname]['duplicates'].remove(conf)
                # Subtract this conformer fromn the total amount of conformesr
                conf_counter -= 1

        # Calculate the saved cost when this dihedral wouldn't have had existed
        cost_count -= (cost_saved(merge_info['mols'][mname]['n_diheds'], last_wp) - cost_saved(
            merge_info['mols'][mname]['n_diheds'] - 1, last_wp))
        merge_info['mols'][mname]['n_diheds'] -= 1
        total_ndiheds -= 1

        prune_data['cost'].append(cost_count)
        prune_data['ndiheds'].append(total_ndiheds)
        prune_data['nconfs'].append(conf_counter)
        prune_data['ldc'].append(int(lc))
        prune_data['dihedrals'].append((dih, mname))

    return prune_data


def pruning_list(prune_data, cutoff):
    molecule_list = {}
    for i in range(len(prune_data['ldc'])):
        if prune_data['ldc'][i] <= cutoff:
            break
        dih, mname = prune_data['dihedrals'][i]
        if mname not in molecule_list.keys():
            molecule_list[mname] = []
        molecule_list[mname].append(dih)
    return molecule_list


def plotting(data, outpath, norm=False):
    nconfs = data['nconfs']
    ndiheds = data['ndiheds']
    ncost = data['cost']
    if norm:
        nconfs = np.array(nconfs) / max(nconfs)
        ndiheds = np.array(ndiheds) / max(ndiheds)
        ncost = np.array(ncost) / max(ncost)

    fig, axs = plt.subplots(2, sharex=True)
    color = 'tab:red'
    axs[1].set_xlabel('Pruning Cutoff, computations since last contribution')
    axs[0].set_ylabel('Total Number of Conformers', color=color)
    axs[1].set_ylabel('Total Number of Conformers', color=color)

    axs[0].plot(data['ldc'], nconfs, color=color)
    axs[1].plot(data['ldc'], nconfs, color=color)

    axs[0].tick_params(axis='y', labelcolor=color)
    axs[1].tick_params(axis='y', labelcolor=color)
    ax02 = axs[0].twinx()  # instantiate a second axes that shares the same x-axis
    ax12 = axs[1].twinx()

    color = 'tab:blue'
    ax02.plot(data['ldc'], ndiheds, color=color)
    ax12.plot(data['ldc'], ncost, color=color)

    ax02.set_ylabel('Number of Dihedrals', color=color)
    ax12.set_ylabel('CPU Hours', color=color)
    ax02.tick_params(axis='y', labelcolor=color)
    ax12.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f'{outpath}/cost_overview.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mergepath', type=str, help='Path to the merged conformer files.')
    parser.add_argument('--outpath', type=str, help='Output path to store merge and pruning data.')
    parser.add_argument('--cutoff', type=int, help='Dihedral pruning cutoff. '
                                                   'Number of computations performed since last new conformer.')
    args = parser.parse_args()

    os.makedirs(args.outpath, exist_ok=True)

    if os.path.isfile(f'{args.outpath}/merge_data.pkl'):
        print('Merge info already exists. Load data instead of recomputing.')
        m_info = pickle.load(open(f'{args.outpath}/merge_data.pkl', 'rb'))
    else:
        m_files = sorted(glob.glob(f'{args.mergepath}/*.merged'))
        m_info = load_merge_data(m_files)
        pickle.dump(m_info, open(f'{args.outpath}/merge_data.pkl', 'wb'))

    if os.path.isfile(f'{args.outpath}/prune_data.pkl'):
        print('Pruning info already exists. Load data instead of recomputing.')
        p_info = pickle.load(open(f'{args.outpath}/prune_data.pkl', 'rb'))
    else:
        p_info = calculate_pruning_cost(m_info)
        pickle.dump(p_info, open(f'{args.outpath}/prune_data.pkl', 'wb'))

    p_list = pruning_list(p_info, args.cutoff)
    pickle.dump(m_info, open(f'{args.outpath}/pruned_dihedrals_cutoff-{int(args.cutoff)}.pkl', 'wb'))
    plotting(p_info, args.outpath, norm=True)
