#!/bin/bash
mkdir dist

tar xzf openbabel-2.4.1.tar.gz
cd openbabel-2.4.1/
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=../../dist -DWITH_INCHI=OFF
make -j4
make install
cd ../..

wget https://github.com/redis/hiredis/archive/master.zip
unzip master.zip
cd hiredis-master
LIBRARY_PATH=lib PREFIX=../dist make
LIBRARY_PATH=lib PREFIX=../dist make install
cd ..


wget http://www.netlib.org/lapack/lapack-3.8.0.tar.gz
tar xzf lapack-3.8.0.tar.gz
cd lapack-3.8.0/
cp INSTALL/make.inc.gfortran make.inc
make -j4 lapacklib lapackelib
cp liblapack*.a ../dist/lib/
cp LAPACKE/include/*.h ../dist/include/
cd  ..

wget http://www.netlib.org/blas/blas-3.8.0.tgz
tar xzf blas-3.8.0.tgz
cd BLAS-3.8.0/
make
cp blas_LINUX.a ../dist/lib/libblas.a
cd ..


wget http://www.netlib.org/blas/blast-forum/cblas.tgz
tar xzf cblas.tgz
cd CBLAS
cp Makefile.LINUX Makefile.in
cp ../BLAS-3.8.0/blas_LINUX.a libblas.a
make -j4
cp lib/cblas_LINUX.a ../dist/lib/libcblas.a
cp include/* ../dist/include

