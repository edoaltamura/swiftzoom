"""
This file contains the `GroupZoom` object which:

+ represents a zoom-in region of a simulated group or cluster, initialized based on redshift or snapshot number
+ inherits the `SWIFTGalaxy` class from the `swiftgalaxy` library
+ if gas temperature is not included in the SWIFT snapshot, you can generate it using the method `obj.set_temperatures_from_internal_energies()`
+ a simple spherical aperture selection can be operated using the method `obj.get_mask_3d_radius_r500()`
"""

import re
import warnings
from typing import Optional, Dict
from unyt import unyt_quantity, K, mp, kb
import numpy as np
from swiftgalaxy import SWIFTGalaxy, Velociraptor
from swiftsimio import mask as sw_mask

from .output_list import OutputList
from .constants import mean_molecular_weight

from h5py import get_config

get_config().default_file_mode = 'r'


class GroupZoom(SWIFTGalaxy):
    """
    A class representing a zoom-in region of a simulated group or cluster, initialized based on redshift or snapshot
    number.
    """

    def __init__(
            self,
            run_directory: str,
            redshift: Optional[float] = None,
            snapshot_number: Optional[int] = None,
            halo_index: Optional[int] = 0,
            auto_recentre: Optional[bool] = True,
            import_all_particles: Optional[bool] = False,
    ) -> None:
        """
        Initialize a GroupZoom instance.

        Parameters:
            run_directory (str): The path to the simulation group's directory.
            redshift (Optional[float], optional): The desired redshift. Defaults to None.
            snapshot_number (Optional[int], optional): The snapshot number. Defaults to None.
            halo_index (Optional[int], optional): The index of the halo to focus on. Defaults to 0.
            auto_recentre (Optional[bool], optional): Whether to automatically recenter the region. Defaults to True.
            import_all_particles (Optional[bool], optional): Whether to import all particles in the box.
                Defaults to False.

        Attributes:
            run_directory (str): The path to the simulation group's directory.
            out_list (OutputList): An instance of OutputList for managing simulation outputs.
            import_all_particles (bool): Flag indicating whether to import all particles in the box.

        Methods:
            set_temperatures_from_internal_energies(self): Set gas temperatures from internal energies if not included.
            get_mask_gas_temperature(self, tmin: unyt_quantity = 1.e5 * K, tmax: unyt_quantity = 1.e15 * K) -> np.ndarray:
                Get a mask for gas particles based on temperature range.
            get_mask_3d_radius_r500(self, rmin: float = 0., rmax: float = 5.) -> Dict[str, np.ndarray]:
                Get masks for different particle types based on their 3D radius in units of r500.

        Examples:
        ```
        # Initialize a GroupZoom instance with a specific redshift.
        group_zoom = GroupZoom('/path/to/simulation/group', redshift=0.5)

        # Initialize a GroupZoom instance with a specific snapshot number.
        group_zoom = GroupZoom('/path/to/simulation/group', snapshot_number=10)

        # Set gas temperatures from internal energies.
        group_zoom.set_temperatures_from_internal_energies()

        # Get a mask for gas particles within a temperature range.
        gas_temperature_mask = group_zoom.get_mask_gas_temperature(tmin=1.e5 * K, tmax=1.e7 * K)

        # Get masks for different particle types based on their 3D radius.
        radius_masks = group_zoom.get_mask_3d_radius_r500(rmin=0.2, rmax=1.0)
        ```
        """

        self.run_directory = run_directory
        self.out_list = OutputList(run_directory)
        self.import_all_particles = import_all_particles

        if snapshot_number is not None:
            snap, cat = self.out_list.files_from_snap_number(snapshot_number)

        elif redshift is not None:
            snap, cat = self.out_list.files_from_redshift(redshift)

        else:
            raise ValueError("Redshift or snapshot_number must be defined.")

        cat = re.sub('\.properties$', '', cat)
        vr_object = Velociraptor(cat, halo_index=halo_index, centre_type='minpot', extra_mask=None)

        if self.import_all_particles:
            # Overrides the spatial mask selection done by swiftsimio for the particles in object <halo_index>.
            # This will import all particles in the box and increases memory usage.
            mask = sw_mask(snap)

            # load_region is a 3x2 list [[left, right], [bottom, top], [front, back]]
            load_region = [[0. * b, b] for b in mask.metadata.boxsize]
            mask.constrain_spatial(load_region)  # Constrain the mask
            spatial_mask_kwargs = dict(_spatial_mask=mask)
        else:
            spatial_mask_kwargs = dict()

        super().__init__(snap, vr_object, auto_recentre=auto_recentre, _extra_mask=None, **spatial_mask_kwargs)
        super().wrap_box()

        # If temperature arrays are not included (e.g.) adiabatic mode
        self.is_nonradiative = self.metadata.n_stars == 0

        if not hasattr(self.gas, 'temperatures') and self.is_nonradiative:
            warnings.warn('Genering temperatures from internal energies.', RuntimeWarning)
            self.set_temperatures_from_internal_energies()

    def set_temperatures_from_internal_energies(self) -> None:
        """
        Calculate and set gas temperatures from internal energies if not already included.

        This method calculates the gas temperatures based on internal energies and sets the 'temperatures' attribute for gas particles.

        Returns:
            None

        Examples:
        ```
        # Initialize a GroupZoom instance.
        group_zoom = GroupZoom('/path/to/simulation/group')

        # Set gas temperatures from internal energies.
        group_zoom.set_temperatures_from_internal_energies()
        ```
        """
        self.gas.internal_energies.convert_to_physical()
        setattr(
            self.gas, 'temperatures', (
                    self.gas.internal_energies *
                    (self.metadata.gas_gamma - 1) *
                    mean_molecular_weight * mp / kb
            )
        )

    def get_mask_gas_temperature(self, tmin: unyt_quantity = 1.e5 * K, tmax: unyt_quantity = 1.e15 * K) -> np.array:
        """
        Get a mask for gas particles based on temperature range.

        Parameters:
            tmin (unyt_quantity, optional): The minimum temperature in Kelvin. Defaults to 1.e5 * K.
            tmax (unyt_quantity, optional): The maximum temperature in Kelvin. Defaults to 1.e15 * K.

        Returns:
            np.ndarray: An array of indices representing gas particles within the specified temperature range.

        Examples:
        ```
        # Initialize a GroupZoom instance.
        group_zoom = GroupZoom('/path/to/simulation/group')

        # Get a mask for gas particles within a temperature range.
        gas_temperature_mask = group_zoom.get_mask_gas_temperature(tmin=1.e5 * K, tmax=1.e7 * K)
        ```
        """
        return np.where((self.gas.temperatures > tmin) & (self.gas.temperatures < tmax))[0]

    def get_mask_3d_radius_r500(self, rmin: float = 0., rmax: float = 6.) -> Dict[str, np.array]:
        """
        Get masks for different particle types based on their 3D radius in units of r500.

        Parameters:
            rmin (float, optional): The minimum scaled radius (in units of r500). Defaults to 0.
            rmax (float, optional): The maximum scaled radius (in units of r500). Defaults to 6.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing masks for different particle types.

        Example:
        ```
        # Initialize a GroupZoom instance.
        group_zoom = GroupZoom('/path/to/simulation/group')

        # Get masks for different particle types based on their 3D radius.
        radius_masks = group_zoom.get_mask_3d_radius_r500(rmin=0.15, rmax=0.5)
        ```
        """
        radial_mask = {}

        for particle_type in ['gas', 'dark_matter', 'stars', 'black_holes']:
            getattr(self.group_zoom, particle_type).spherical_coordinates.radius.convert_to_physical()
            r500 = self.group_zoom.halo_finder.spherical_overdensities.r_500_rhocrit
            radius_scaled = getattr(self.group_zoom, particle_type).spherical_coordinates.radius / r500
            radial_mask[particle_type] = (radius_scaled > rmin) & (radius_scaled < rmax)

        return radial_mask
