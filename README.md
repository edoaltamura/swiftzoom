SWIFTzoom
==========

[![Documentation Status](https://readthedocs.org/projects/swiftzoom/badge/?version=latest)](https://swiftzoom.readthedocs.io/en/latest/?badge=latest)

A Python 3 library to analyse zoom-in hydrodynamic simulations of groups and clusters of galaxies run with [SWIFT](http://swift.dur.ac.uk), a parallel, 
multi-purpose numerical simulation software for hydrodynamic simulations in astrophysics and cosmology.


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

You can install `swiftzoom` using the Python packaging manager `pip` or any other packaging manager that you prefer:

`pip install swiftzoom`

Citing
----------
While under development, you can acknowledge this repository by citing [Altamura et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.3164A/abstract):
```
@ARTICLE{2023MNRAS.520.3164A,
       author = {{Altamura}, Edoardo and {Kay}, Scott T. and {Bower}, Richard G. and {Schaller}, Matthieu and {Bah{\'e}}, Yannick M. and {Schaye}, Joop and {Borrow}, Josh and {Towler}, Imogen},
        title = "{EAGLE-like simulation models do not solve the entropy core problem in groups and clusters of galaxies}",
      journal = {\mnras},
     keywords = {hydrodynamics, methods: numerical, software: simulations, galaxies: clusters, galaxies: fundamental parameters, galaxies: groups - tions, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2023,
        month = apr,
       volume = {520},
       number = {2},
        pages = {3164-3186},
          doi = {10.1093/mnras/stad342},
archivePrefix = {arXiv},
       eprint = {2210.09978},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.3164A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
