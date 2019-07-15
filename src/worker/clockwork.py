
import functools
import joblib

import numpy as np
import itertools
import time

import scipy
from scipy import special

import matplotlib.pyplot as plt

cachedir = '.pycache'
memory = joblib.Memory(cachedir, verbose=0)

def clockwork(res, debug=False):
    """

    get start, step size and no. of steps from clockwork resolution n

    @param
        res int resolution
        debug boolean

    @return
        start
        step_size
        n_steps

    """

    if res == 0:
        start = 0
        step = 360
        n_steps = 1

    else:

        start = 360.0 / 2.0 ** (res)
        step = 360.0 / 2.0 ** (res-1)
        n_steps = 2 ** (res - 1)

    if debug:
        print(res, step, n_steps, start)

    return start, step, n_steps


def generate_torsion_combinations(torsions, n_body):
    torsions_idx = list(range(torsions))
    # TODO this is unq right?
    combinations = itertools.combinations(torsions_idx, n_body)
    return combinations

def generate_clockwork_combinations(resolution, n_body):

    # TODO rewrite to a generator

    rest = range(0, resolution)
    rest = list(rest) + [resolution]

    combinations = itertools.product(rest, repeat=n_body)
    combinations = list(combinations)

    # This will reduce the actual cost of combinations
    # TODO uncomment this
    combinations = [list(x) for x in combinations if resolution in x]

    return combinations


def generate_angles(resolution, n_torsions):
    """

    Setup angle iterator based on number of torsions

    """

    if type(resolution) == int:
        resolution = [resolution]*n_torsions

    angles = []

    for r in resolution:
        start, step, n_steps = clockwork(r)
        scan = np.arange(start, start+step*n_steps, step)
        angles.append(scan)

    iterator = itertools.product(*angles)

    return iterator


def clockwork_cost(clockres_list, n):
    """

    param:
        resolution list
        total number of torsions

    return
        cost

    """
    t = len(clockres_list)
    maxres = max(clockres_list)
    value = costfunc(t, maxres, n)
    return value


@functools.lru_cache()
def cost_resolution(resolution, n_body):
    """

    """

    count = 0
    resolution_combinations = generate_clockwork_combinations(resolution, n_body)
    for res in resolution_combinations:
        angle_iterator = generate_angles(res, n_body)
        for conf in angle_iterator:
            count += 1

    return count


def costfunc(n_body, resolution, total_torsions=30, numerical_count=True):
    """

    n_body int - how many torsions
    resolution int - max clockwork resolution
    total_torsions int - total number of torsions in system
    numerical_angle_count bool - extra slow

    """

    torsion_combinations = special.binom(total_torsions, n_body)

    if numerical_count:

        inner_loop = cost_resolution(resolution, n_body)

    else:
        resolution_combinations = (resolution+1)**n_body - (resolution)**n_body
        angle_combinations = (2**(resolution-1))**n_body # Estimate
        inner_loop = resolution_combinations*angle_combinations

    value = torsion_combinations * inner_loop

    return value


def count_costfunc(n_body, resolution, total_torsions=30):

    count = 0

    torsion_combinations = generate_torsion_combinations(total_torsions, n_body)

    # resolution_combinations = generate_clockwork_combinations(resolution, n_body)
    # resolution_combinations = list(resolution_combinations)
    # angle_iterator = generate_angles(res, n_body)
    # angle_iterator = list(angle_iterator)

    for tor in torsion_combinations:

        resolution_combinations = generate_clockwork_combinations(resolution, n_body)
        for res in resolution_combinations:

            angle_iterator = generate_angles(res, n_body)
            for conf in angle_iterator:
                count += 1

    return count


def next_cost(torres, clockres, costlist=None):
    """
    return next choice for the cost function
    """

    if costlist is None:
        costlist, costmatrix = generate_costlist()

    idx = costlist.index([clockres, torres])

    next_res = costlist[idx+1]

    return next_res


@memory.cache
def generate_costlist(max_torsions=5, max_clockwork=7, total_torsions=20):
# def generate_costlist(max_torsions=3, max_clockwork=6):

    torsions = np.asarray(list(range(1, max_torsions)))
    clockworks = np.asarray(list(range(1, max_clockwork)))

    N_tor = torsions.shape[0]
    N_clo = clockworks.shape[0]

    costmatrix = np.zeros((torsions.shape[0], clockworks.shape[0]))

    for i, n_tor in enumerate(torsions):
        for j, clockres in enumerate(clockworks):
            ouch = costfunc(n_tor, clockres, total_torsions=total_torsions)
            costmatrix[i, j] = ouch
            print(n_tor, clockres, ouch)

    # costmatrix = costmatrix.flatten()

    # Flat index array
    idxcost = np.argsort(costmatrix, axis=None)

    # convert flat index to coordinates
    idxcost = np.unravel_index(idxcost, costmatrix.shape)

    # Stack index pairwise
    # idxcost = np.vstack(idxcost).T

    cost_x = []
    cost_y = []

    for i,j in zip(idxcost[0], idxcost[1]):

        i_tor = torsions[i]
        j_clo = clockworks[j]
        cost_x.append([i_tor, j_clo])
        cost_y.append(costmatrix[i,j])

    return cost_x, np.asarray(cost_y)




def test():

    # 16 torsions

    total_torsions = 16
    resolution = 4
    n_body = 4

    torsion_combinations = generate_torsion_combinations(total_torsions, n_body)
    n_tor = len(list(torsion_combinations))
    print("combi tor", n_tor, special.binom(total_torsions, n_body))

    resolution_combinations = generate_clockwork_combinations(resolution, n_body)
    nx = len(resolution_combinations)
    print("combi res", nx, (resolution+1)**n_body - (resolution)**n_body)

    angles = generate_angles(resolution, n_body)
    na = len(list(angles))
    print("combi ang", na, (2**(resolution-1))**n_body)

    # cr = count_costfunc(n_body, resolution)
    ct = costfunc(n_body, resolution)

    print()
    # print("actual", cr)
    print("costfc", int(ct))

    return


def main():
    # plot cost
    ticks_x, costmatrix = generate_costlist(total_torsions=20)

    print("t r")
    for x, y in zip(ticks_x, costmatrix):
        print(*x, y)

    ticks_x = ["{:},{:}".format(*x) for x in ticks_x]
    ticks_x = np.asarray(ticks_x)
    xticks = list(range(len(ticks_x)))
    xticks = np.asarray(xticks)

    max_cost = 10**7

    idx = np.where(costmatrix < max_cost)

    ticks_x = ticks_x[idx]
    xticks = xticks[idx]
    costmatrix = costmatrix[idx]

    plt.figure(figsize=(15, 5))
    plt.plot(costmatrix, 'k.-',
            markersize=10,
            markeredgewidth=1.5, markeredgecolor='w')
    plt.xticks(xticks, ticks_x, rotation=-45)
    plt.xlabel("(Torsions, Clockwork)")
    plt.yscale("log")
    plt.grid(True, axis="y", color="k")

    ax = plt.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)


    # Add extra lines
    for total_torsions in [10, 30, 40, 50, 60, 70]:
        print("add extra")
        ticks_x, costmatrix = generate_costlist(total_torsions=total_torsions)
        idx = np.where(costmatrix < max_cost)
        costmatrix = costmatrix[idx]
        plt.plot(costmatrix, '.-',
                markersize=10,
                markeredgewidth=1.5, markeredgecolor='w')


    plt.minorticks_off()
    plt.savefig("linearcost", bbox_inches="tight")

    plt.clf()

    # Test
    # t, c = next_cost(2,2)
    # print(t,c)

if __name__ == '__main__':
    main()
