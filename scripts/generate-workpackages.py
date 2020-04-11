#!/usr/bin/env python
import sys
import itertools as it

# arguments
molname, bucket, dihs = sys.argv[1:]
dihs = dihs.split("-")
nbody, resolution = bucket.split("R")
nbody = int(nbody[1:])

# output
for wp in it.combinations(dihs, nbody):
	print ("molname:" + '-'.join(wp) + ":" + resolution) 
