#!/bin/bash

mkdir dist

bash build_openbabel.sh

bash build_hiredis.sh

bash build_lapack.sh

bash build_blas.sh

bash build_cblas.sh

