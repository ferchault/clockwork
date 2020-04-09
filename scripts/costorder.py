#!/usr/bin/env python
import scipy.special as scs
import itertools as it
import numpy as np

N = 40 # Number of torsions

def configuration_count(resolution):
	if resolution == 0:
		return 1
	return 2**(resolution - 1)

def combinations_in_group(T, R):
	combinations = []
	for case in it.product(range(R+1), repeat=T):
		if R in case:
			combinations.append(case)
	return combinations

def group_cost(combinations):
	cost = 0
	for case in combinations:
		cost += np.prod(list(map(configuration_count, case)))
	return cost

RS = 5
print ('     ' + ' '.join(['  R:%16d' % _ for _ in range(RS)]))

specifications = []
costs = []
for T in range(1, 7):
	res = []
	for R in range(RS):
		cost = scs.binom(N, T)*group_cost(combinations_in_group(T, R))
		res.append(cost)
		costs.append(cost)
		specifications.append((T, R))
	print ('T:%3d' % T + ' '.join(['%20d' % _ for _ in res]))

ranks = sorted(costs)
print ('     ' + ' '.join(['  R:%16d' % _ for _ in range(RS)]))
for T in range(1, 7):
	res = []
	for R in range(RS):
		cost = scs.binom(N, T)*group_cost(combinations_in_group(T, R))
		res.append(ranks.index(cost))
	print ('T:%3d' % T + ' '.join(['%20d' % _ for _ in res]))


print ("Order: nbody, resolution")
labels = []
for pos in np.argsort(costs):
	nbody, resolution = specifications[pos]
	labels.append(f"N{nbody}R{resolution}")
print (' '.join(labels))
