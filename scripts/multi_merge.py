import os
from glob import glob
import multiprocessing as mp

from tqdm import tqdm

from merge import Merger


def merge_batch(inp_tuple):
    infile, outfile, prev_merge = inp_tuple
    try:
        m = Merger(prev_merge, infile)
        m.save(outfile)
    except ValueError as e:
        print(infile, e)


def batch_inp(mol_path, batch_num, wp_path):
    batches = []
    for mol in tqdm(mol_path):
        mol_name = os.path.dirname(mol).split('/')[-1]
        p_merge = f'{wp_path}/batch{batch_num-1}/{mol_name}_batch{batch_num-1}.merged'
        if not os.path.isfile(p_merge) and batch_num > 1:
            print(f'Batch {batch_num-1} of molecule {mol_name} has not been merged yet. Skipped.')
            continue
        outfile = f'{outpath}/{mol_name}_batch{batch_num}.merged'
        if os.path.isfile(outfile):
            print(f'Batch {batch_num} has already been merged. Skipped.')
            continue
        batches.append((mol, outfile, p_merge))
    return batches


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--molpath', action='store', help='', metavar="dir")
    parser.add_argument('--outpath', action='store_true', help='')
    parser.add_argument('--ncpus', action='store', help='', metavar="o")
    parser.add_argument('--nbatch', action='store', metavar="int", type=int)

    args = parser.parse_args()
    # python multi_merge.py --molpath /mnt/FOXTROT/CONFORMER/merge_files2 --outpath /mnt/FOXTROT/CONFORMER/merge_files2/ --ncpus 20 --nbatch 1

    outpath = f'{args.outpath}/batch{args.nbatch}'
    os.makedirs(outpath, exist_ok=True)

    b = batch_inp(mol_path=sorted(glob(f'{args.molpath}/molecules/*/batch{args.nbatch}.results')),
                  batch_num=args.nbatch,
                  wp_path=args.molpath)
    with mp.Pool(processes=args.ncpus) as pool:
        with tqdm(total=len(b), desc='Molecules') as pbar:
            for _ in pool.imap_unordered(merge_batch, b):
                pbar.update()

