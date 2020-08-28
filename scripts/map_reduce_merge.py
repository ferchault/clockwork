import os
n_threads = 1
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)

import tempfile
import multiprocessing as mp

import numpy as np

from merge import Merger


def make_batches(results_file, nbatches):
    with open(results_file) as infile:
        lines = infile.readlines()

    n_lines = np.arange(len(lines))
    np.random.shuffle(n_lines)
    sort_idx = np.argsort(n_lines)
    batches = np.array_split(n_lines, nbatches)

    temppaths = []
    for i in range(nbatches):
        fd, path = tempfile.mkstemp()
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
    parser.add_argument('--mergefile', type=str,)
    parser.add_argument('--workpackagefile', type=str,)
    parser.add_argument('--outfile', type=str,)
    parser.add_argument('--n_procs', type=int,)
    parser.add_argument('--diherdralfilter', type=str,  default=None)

    args = parser.parse_args()

    tpaths, s_idx = make_batches(args.workpackagefile, args.n_procs)

    pool = mp.Pool(processes=args.n_procs)
    pre_filter = pool.map(worker, zip(args.mergefile, tpaths))

    pre_filter = np.vstack(pre_filter)[s_idx]
    np.save('similarity_results.py', pre_filter)
    m = Merger(args.mergefile, args.workpackagefile, None, similarity_check=pre_filter)
    m.save(args.outfile)

    for tpath in tpaths:
        os.remove(tpath)
