#!/bin/bash

wget http://www.netlib.org/lapack/lapack-3.8.0.tar.gz
tar xzf lapack-3.8.0.tar.gz
cd lapack-3.8.0/
cp INSTALL/make.inc.gfortran make.inc
make -j4 lapacklib lapackelib
cp liblapack*.a ../dist/lib/
cp LAPACKE/include/*.h ../dist/include/
cd  ..

