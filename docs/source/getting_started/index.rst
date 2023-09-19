Getting Started
===============

The [SWIFT](http://swift.dur.ac.uk) hydrodynamics simulation code is used by hundreds of research studies and is instrumental in a wide range of use cases.
Galaxy formation simulations in astrophysics and computational cosmology is the main topic that SWIFT users focus on.
This library provides an agile framework for handling and analysing SWIFT data from zoom-in simulations. With its object-oriented design,
:mod:`swiftzoom` extends the functionalities of the following SWIFT toolkits:

+ :mod:`swiftsimio`
+ :mod:`swiftgalaxy`

Getting set up with :mod:`swiftzoom` is simple. its core runs with few
requirements, although optional packages are recommended to use this library to its full potential. The requirements
are detailed below.


Requirements
------------

This requires ``python`` ``v3.8.0`` or higher. Unfortunately it is not
possible to support :mod:`swiftzoom` on versions of python lower than this.
This code does not support ``python`` ``v2``.

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

``git clone https://github.com/edoaltamura/swiftzoom.git``

and install with ``pip`` using the ``-e`` flag,

``cd swiftzoom``

``pip install -e .``

.. TODO: Add contribution guide.

Usage
-----

Given your simulation outputs, you can import the data using the ``register`` submodule:

.. code-block:: python

    from swiftzoom import GroupZoom

    # Initialize a GroupZoom instance with a specific redshift.
    group_zoom = GroupZoom('/path/to/simulation/group', redshift=0.5)

    # Initialize a GroupZoom instance with a specific snapshot number.
    group_zoom = GroupZoom('/path/to/simulation/group', snapshot_number=10)

    # Set gas temperatures from internal energies.
    group_zoom.set_temperatures_from_internal_energies()

    # Get a mask for gas particles within a temperature range.
    gas_temperature_mask = group_zoom.get_mask_gas_temperature(tmin=1.e5 * K, tmax=1.e7 * K)


This code snippet will return a data handle ready to output, for instance, 3D radial profiles.