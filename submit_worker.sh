#!/bin/bash

connection=$1

anaconda worker.py \
        -f \
        -torsion-list \
        --connect-redis "$1"

