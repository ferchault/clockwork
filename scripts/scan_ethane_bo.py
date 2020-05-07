import sys
import os
from glob import glob
import subprocess

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def write_ethane(path):
    ethane_xyz = ['8\n',
                  ' energy: -7.336370187290 gnorm: 0.000776382740 xtb: 6.2.3 (conda-forge)\n',
                  'C         0.76023192793568    0.00000286028951    0.00016136802424\n',
                  'C        -0.76023192793579   -0.00000286036262   -0.00016136802914\n',
                  'H         1.14319252115693   -0.00981694147447    1.01830800144609\n',
                  'H         1.14348504765852   -0.87677821987606   -0.51733344476225\n',
                  'H         1.14348200014852    0.88659339685752   -0.50033549617078\n',
                  'H        -1.14348504764013    0.87677821990184    0.51733344476366\n',
                  'H        -1.14319252115645    0.00981694149764   -1.01830800144371\n',
                  'H        -1.14348200016727   -0.88659339683337    0.50033549617189\n']
    with open(f'{path}/ethane.xyz', 'w') as inputfile:
        inputfile.writelines(ethane_xyz)


def write_xtb_input(path, distance=4.5, steps=60):
    scan_lines = ['$constrain\n',
                  ' distance: 1, 2, 1.5\n',
                  '$scan\n',
                  f' 1:   1.5,  {distance}, {steps}\n',
                  '$end\n',
                  '\n']
    with open(f'{path}/scan.input', 'w') as inputfile:
        inputfile.writelines(scan_lines)


def run_xtb(cmd):
    with open('xtb.log', 'w') as logfile:
        subprocess.run(cmd.split(), stdout=logfile, stderr=logfile)


def split_scan(filename, conf_dir, distance, stepsize):
    with open(filename, 'r') as infile:
        lines = infile.readlines()
    n_atoms = int(lines[0])
    conformers = []
    for i in range(0, len(lines), n_atoms + 2):
        conformers.append(lines[i:i + n_atoms + 2])

    os.makedirs(conf_dir, exist_ok=True)
    for i, conf in zip(np.linspace(1.5, distance, stepsize), conformers):
        os.makedirs(f'{conf_dir}/ccdist_{i:.2f}', exist_ok=True)
        write_xyz(filename=f'{conf_dir}/ccdist_{i:.2f}/ccdist_{i:.2f}.xyz', data=conf)


def write_xyz(filename, data):
    with open(filename, 'w') as outfile:
        for line in data:
            outfile.write(line)


def read_wbo(filepath):
    with open(filepath) as infile:
        lines = infile.readlines()
    cc_bond = lines[0].split()
    bond = cc_bond[:2]
    if bond == ['1', '2']:
        return cc_bond[-1]
    else:
        return None


def read_energy():
    with open('xtb.log') as logfile:
        for line in logfile:
            if "  | TOTAL ENERGY  " in line:
                energy = float(line.strip().split()[-3])
                break
    return energy


def scan_ethane_wbo(distance, stepsize, path):
    os.makedirs(path, exist_ok=True)
    write_ethane(path)
    write_xtb_input(path, distance=distance, steps=stepsize)
    os.chdir(path)
    xtb_scan_cmd = f'xtb ethane.xyz --opt --input scan.input'
    run_xtb(xtb_scan_cmd)
    conf_dir = f'{path}/scan_path'
    split_scan('xtbscan.log', conf_dir, distance, stepsize)

    wbos = []
    distances = []
    energies = []
    path_mols = sorted(glob(f'{conf_dir}/*/*.xyz'))
    for mol in tqdm(path_mols):
        os.chdir(os.path.dirname(mol))
        xtb_sp_cmd = f'xtb {mol} --sp'
        run_xtb(xtb_sp_cmd)
        wbo = read_wbo('wbo')
        if wbo is not None:
            wbos.append(float(wbo))
            distances.append(float(''.join(filter(str.isdigit, os.path.basename(mol)))) * 10 ** -2)
            energies.append(read_energy())
    return wbos, distances, energies


if __name__ == '__main__':
    save_path = sys.argv[1:]
    w, d, e = scan_ethane_wbo(distance=4.5,
                              stepsize=60,
                              path=save_path)

    e = np.array(e)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Ethane C-C Scan')
    ax1.plot((e - min(e)) * 627.15, w)
    ax1.axhline(y=0.1, color='k', linestyle='--', label='Chemical Accuracy')
    ax1.set(xlabel='Energy [kcal/mol]', ylabel='Wiberg Bond Order')
    ax2.plot(d, w)
    ax2.axhline(y=0.1, color='k', linestyle='--', label='Chemical Accuracy')
    ax2.set(xlabel='Distance [$\AA$]')
    plt.show()
