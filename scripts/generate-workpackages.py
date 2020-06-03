#!/usr/bin/env python
import sys
import itertools as it
import argparse

# arguments
parser = argparse.ArgumentParser(description="builds a clockwork batch input file")
parser.add_argument("molname", type=str)
parser.add_argument("buckets", type=str)
parser.add_argument("dihedrals", type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    dihs = args.dihedrals.split("-")
    for bucket in args.buckets.split():
        nbody, resolution = bucket.split("R")
        nbody = int(nbody[1:])
        for wp in it.combinations(dihs, nbody):
            print(f"{args.molname}:" + "-".join(wp) + ":" + resolution)
