Getting Started
===============

The SWIFT astrophysical simulation code (http://swift.dur.ac.uk) is used
widely. There exists many ways of reading the data from SWIFT, which outputs
HDF5 files. These range from reading directly using :mod:`h5py` to using a
complex system such as :mod:`yt`; however these either are unsatisfactory
(e.g. a lack of unit information in reading HDF5), or too complex for most
use-cases. This (thin) wrapper provides an object-oriented API to read
(dynamically) data from SWIFT.

Getting set up with :mod:`swiftsimio` is easy; it (by design) has very few
requirements. There are a number of optional packages that you can install
to make the experience better and these are recommended. All requirements
are detailed below.


Requirements
------------

This requires ``python`` ``v3.8.0`` or higher. Unfortunately it is not
possible to support :mod:`swiftsimio` on versions of python lower than this.
It is important that you upgrade if you are still a ``python2`` user.

Python packages
^^^^^^^^^^^^^^^

+ ``numpy``, required for the core numerical routines.
+ ``h5py``, required to read data from the SWIFT HDF5 output files.
+ ``unyt``, required for symbolic unit calculations (depends on ``sympy``).

Optional packages
^^^^^^^^^^^^^^^^^

+ ``numba``, highly recommended should you wish to use the in-built visualisation
  tools.
+ ``scipy``, required if you wish to generate smoothing lengths for particle types
  that do not store this variable in the snapshots (e.g. dark matter)
+ ``tqdm``, required for progress bars for some long-running tasks. If not installed
  no progress bar will be shown.


Installing
----------

:mod:`swiftzoom` can be installed using the python packaging manager, ``pip``,
or any other packaging manager that you wish to use:

``pip install swiftzoom``

Note that this will install any required packages for you.

To set up the code for development, first clone the latest master from GitHub:

``git clone https://github.com/SWIFTSIM/swiftzoom.git``

and install with ``pip`` using the ``-e`` flag,

``cd swiftzoom``

``pip install -e .``

.. TODO: Add contribution guide.
