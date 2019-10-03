#!/usr/bin/env bash

# First, do single-machine test
nosetest *.py

# Second, test when there is no mpi
PYTHONPATH_SAVE=$PYTHONPATH
PYTHONPATH=$PWD/nompi:$PYTHONPATH
nosetests *.py
PYTHONPATH=PYTHONPATH_SAVE

# Third, test when mpi exists.
for i in {1..5}; do
  mpirun -n $i nosetests *.py
done
