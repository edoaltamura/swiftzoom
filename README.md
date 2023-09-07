

![SWIFTzoom Logo Banner - Dark](https://github.com/edoaltamura/swiftzoom/blob/main/.github/workflows/swzoom-banner-dark.SVG#gh-dark-mode-only)
![SWIFTzoom Logo Banner - Light](https://github.com/edoaltamura/swiftzoom/blob/main/.github/workflows/swzoom-banner-light.SVG#gh-light-mode-only)

[![Python version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://pypi.org/project/swiftzoom/)
[![PyPI version](https://badge.fury.io/py/swiftzoom.svg)](https://pypi.org/project/swiftzoom/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/swiftzoom.svg)](https://anaconda.org/conda-forge/swiftzoom)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/edoaltamura/swiftzoom/blob/main/LICENSE.md)
[![Slack Organisation](https://img.shields.io/badge/slack-chat-blueviolet.svg?label=SWIFT%20Slack&logo=slack)](https://swiftsim.slack.com)
![CircleCI - Main Branch](https://img.shields.io/circleci/build/github/edoaltamura/swiftzoom/main?label=main)
![Develop Branch Build](https://img.shields.io/circleci/build/github/edoaltamura/swiftzoom/develop?label=develop)
[![Documentation](https://readthedocs.org/projects/swiftzoom/badge/?version=latest)](https://swiftzoom.readthedocs.io/en/latest/?badge=latest)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/7769/badge)](https://www.bestpractices.dev/projects/7769)
![GitHub repo size](https://img.shields.io/github/repo-size/edoaltamura/swiftzoom)


A Python library to analyse zoom-in hydrodynamic simulations of groups and clusters of galaxies run with [SWIFT](http://swift.dur.ac.uk), a parallel, 
multi-purpose numerical simulation software for hydrodynamic simulations in astrophysics and cosmology.


Complete documentation is available at the [ReadTheDocs](http://swiftzoom.readthedocs.org) repository.

Installing
----------

You can install SWIFTzoom using the Python packaging manager `pip` by typing:

`pip install swiftzoom`

Main features
------------
- **Project template**. A standard, intuitive and repeatable structure for data science pipelines for simulation-based projects. This feature is inspired by [Cookiecutter Data Science](https://github.com/drivendata/cookiecutter-data-science/) and [Kedro](https://github.com/kedro-org/kedro), the data science tool developed by [QuantumBlack, AI by McKinsey](https://www.mckinsey.com/capabilities/quantumblack/how-we-help-clients).
- **Snapshot-catalogue binding**. Combines the efficiency of [swiftsimio](https://github.com/SWIFTSIM/swiftsimio) and [swiftgalaxy](https://github.com/SWIFTSIM/swiftgalaxy) with halo-catalogue information from [velociraptor](https://github.com/SWIFTSIM/velociraptor-python) to accelerate the analysis of single objects.
- **Radial profiles**. A sub-module to compute radial distribution profiles of widely-used quantities (e.g. density, temperature, pressure, entropy) and a template for defining custom profiles.
- **Lagrangian tracking**. A tool for tracking an ensemble of particles backwards and forwards in time from a given snapshot by matching unique particle IDs.
- **Map visualisation**. A high-level easy-to-use abstraction of the [swiftsimio](https://github.com/SWIFTSIM/swiftsimio) visualisation submodule to produce maps of the particles in the simulation. 

Requirements
------------
This package requires `python` `v3.7.0` or higher. Lower versions are not tested and may present compatibility issues.

### Python packages
+ `numpy`, required for the core numerical routines.
+ `swiftsimio`, required to read data from the SWIFT HDF5 output files efficiently.
+ `unyt`, required for symbolic unit calculations (depends on `sympy`).

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

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
