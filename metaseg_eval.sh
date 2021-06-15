#!/bin/bash
#
#	usage: ./meta_eval.sh
#

clear 

python metrics_setup.py build_ext --inplace

export OPENBLAS_NUM_THREADS=1

python metaseg_eval.py --NUM_CORES=10 --NUM_LASSO_LAMBDAS=50

