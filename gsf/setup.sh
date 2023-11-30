#!/bin/bash

## Compile the fortran module with omp to get the shared lib twobody.so and the interface twobody.pyf
## If the default on your system is python3 and f2py is linked accordingly, you have to make the following two 
## changes: f2py3 --> f2py, and python3 --> python

f2py3 -m twobody --fcompiler=gfortran --f90flags='-fopenmp' -lgomp -c twobody.f95

f2py3 -h twobody.pyf -m twobody --overwrite-signature twobody.f95

## A few lines in twobody.pyf need a bit of editing, which is done by edit_pyf.py

python3 edit_pyf.py
