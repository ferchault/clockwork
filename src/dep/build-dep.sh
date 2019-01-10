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
PREFIX=../dist make
PREFIX=../dist make install


wget http://www.netlib.org/lapack/lapack-3.8.0.tar.gz
tar xzf lapack-3.8.0.tar.gz
cd lapack-3.8.0/
cp INSTALL/make.inc.gfortran make.inc
make lapacklib lapackelib

module load OpenBLAS/0.3.3-GCC-8.2.0-2.31.1 ScaLAPACK/2.0.2-gompi-2018.08-OpenBLAS-0.3.3
