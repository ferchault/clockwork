import sys
import os
from glob import glob
import subprocess
import multiprocessing as mp

from tqdm import tqdm
import cclib

nc2el = {6: 'C', 1: 'H', 8: 'O'}


def parse_data(filepath):
    parser = cclib.io.XYZReader(filepath)
    data = parser.parse()
    return data.atomcoords, data.atomnos


def run_xtb(input):
    outdir = os.path.dirname(input)
    os.chdir(outdir)
    cmd = f'xtb {input} --opt extreme --parallel 1'
    with open('xtb.log', 'w') as logfile:
        subprocess.run(cmd.split(), stdout=logfile, stderr=logfile)


def write_xyz(coords, nuclear_charges, optpath):
    elements = [nc2el[nc] for nc in nuclear_charges]
    with open(f'{optpath}/clockwork.xyz', 'w') as outfile:
        atoms = len(nuclear_charges)
        outfile.write(f'{atoms} \n\n')
        for j in range(atoms):
            outfile.write(f'{elements[j]} \t{coords[j][0]:.9f}\t{coords[j][1]:.9f}\t{coords[j][2]:.9f}\n')


if __name__ == '__main__':
    const_xyz_path, outpath = sys.argv[1:]
    constitutional_isomers = sorted(glob(f'{const_xyz_path}/*.xyz'))

    for ci in tqdm(constitutional_isomers):
        ci_name = os.path.basename(ci)
        os.makedirs(f'{outpath}/{ci_name}', exist_ok=True)
        crds, ncs = parse_data(ci)
        for i in range(len(crds)):
            ci_conf_out = f'{outpath}/{ci_name}/conf_{i:04d}/'
            os.makedirs(ci_conf_out, exist_ok=True)
            write_xyz(crds[i], ncs, ci_conf_out)

    xtb_jobs = sorted(glob(f'{outpath}/*/*/clockwork.xyz'))
    n_procs = 48
    with mp.Pool(processes=n_procs) as pool:
        with tqdm(total=len(xtb_jobs), desc='Molecules') as pbar:
            for _ in pool.imap_unordered(run_xtb, xtb_jobs):
                pbar.update()
