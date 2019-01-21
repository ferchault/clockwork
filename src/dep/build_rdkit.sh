#!/bin/bash

git clone --depth 1 https://github.com/rdkit/rdkit

cd rdkit
mkdir build
cd build
cmake ..


