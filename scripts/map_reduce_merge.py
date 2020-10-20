import os
n_threads = 1
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)

import tempfile
import multiprocessing as mp

import numpy as np

from merge import Merger


def make_batches(results_file, nbatches, outdir=None, seed=42):
    with open(results_file) as infile:
        lines = infile.readlines()

    n_lines = np.arange(len(lines))
    np.random.seed(seed)
    np.random.shuffle(n_lines)
    sort_idx = np.argsort(n_lines)
    batches = np.array_split(n_lines, nbatches)

    temppaths = []
    for i in range(nbatches):
        fd, path = tempfile.mkstemp(dir=outdir)
        with open(path, 'w') as tmp:
            for line_id in batches[i]:
                tmp.write(lines[line_id])
        temppaths.append(path)
    return temppaths, sort_idx


def worker(inp):
    m = Merger(*inp)
    return m.get_contribution()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--merge_into_filename', type=str)
    parser.add_argument('--workpackage', type=str)
    parser.add_argument('--outmerge', type=str)
    parser.add_argument('--n_procs', type=int)
    parser.add_argument('--tmp_dir', type=str,  default=None)
    parser.add_argument('--diherdralfilter', type=str,  default=None)

    args = parser.parse_args()

    tpaths, s_idx = make_batches(args.workpackage, args.n_procs, outdir=args.tmp_dir)

    pool = mp.Pool(processes=args.n_procs)
    pre_filter = pool.map(worker, zip([args.merge_into_filename]*args.n_procs, tpaths, [args.diherdralfilter]*args.n_procs))
    pre_filter = np.array([item for sublist in pre_filter for item in sublist])[s_idx]
    pre_filter = np.array([item for sublist in pre_filter for item in sublist], dtype='object')
    # np.save('similarity_results.npy', pre_filter)
    m = Merger(args.merge_into_filename, args.workpackage, None, similarity_check=pre_filter)
    m.save(args.outmerge)

    for tpath in tpaths:
        os.remove(tpath)
