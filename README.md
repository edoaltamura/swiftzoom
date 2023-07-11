`SWIFTzoom`
==========

[![Documentation Status](https://readthedocs.org/projects/swiftzoom/badge/?version=latest)](https://swiftzoom.readthedocs.io/en/latest/?badge=latest)

A `Python 3` library to analyse zoom-in hydrodynamic simulations of groups and clusters of galaxies run with `SWIFT`.


Detailed documentation is available at the [ReadTheDocs](http://swiftzoom.readthedocs.org) repository.

Requirements
------------
This package requires `python` `v3.9.0` or higher. Lower versions are not tested and may present compatibility 
issues.

### Python packages


+ `numpy`, required for the core numerical routines.
+ `swiftsimio`, required to read data from the SWIFT HDF5 output files efficiently.
+ `unyt`, required for symbolic unit calculations (depends on `sympy`).

Installing
----------

YOu can install `swiftzoom` using the Python packaging manager `pip` or any other packaging manager that you prefer:

`pip install swiftzoom`
