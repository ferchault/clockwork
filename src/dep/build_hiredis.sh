#!/bin/bash

wget https://github.com/redis/hiredis/archive/master.zip
unzip master.zip
cd hiredis-master
LIBRARY_PATH=lib PREFIX=../dist make
LIBRARY_PATH=lib PREFIX=../dist make install
cd ..

