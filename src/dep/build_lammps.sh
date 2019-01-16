#!/bin/bash

git clone --depth 1 https://github.com/lammps/lammps

cd lammps
cd src

cd STUBS
make
cd ..

make g++_serial mode=shlib

cp liblammps.so ../../dist/lib


