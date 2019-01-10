#!/bin/bash

wget http://www.netlib.org/blas/blast-forum/cblas.tgz
tar xzf cblas.tgz
cd CBLAS
cp Makefile.LINUX Makefile.in
cp ../BLAS-3.8.0/blas_LINUX.a libblas.a
make -j4
cp lib/cblas_LINUX.a ../dist/lib/libcblas.a
cp include/* ../dist/include

