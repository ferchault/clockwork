#!/bin/bash

wget http://www.netlib.org/blas/blas-3.8.0.tgz
tar xzf blas-3.8.0.tgz
cd BLAS-3.8.0/
make
cp blas_LINUX.a ../dist/lib/libblas.a

