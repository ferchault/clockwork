#!/bin/bash

tar xzf openbabel-2.4.1.tar.gz
cd openbabel-2.4.1/
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=../../dist -DWITH_INCHI=OFF
make -j4
make install
cd ../..

