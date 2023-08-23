import numpy as np
import unyt
from swiftsimio import cosmo_array
from dataclasses import dataclass

from loader import GroupZoom
from .helper_functions import numpy_to_cosmo_array

black_hole_fields = [
    'accreted_angular_momenta', 'accretion_boost_factors', 'accretion_limited_time_steps', 
    'accretion_rates', 'agntotal_injected_energies', 'birth_gas_densities', 'birth_metallicities', 
    'coordinates', 'cumulative_number_of_seeds', 'dynamical_masses', 'eddington_fractions', 
    'element_masses', 'energy_reservoir_thresholds', 'energy_reservoirs', 'feedback_delta_t', 
    'fofgroup_ids', 'formation_scale_factors', 'gas_circular_velocities', 'gas_densities', 
    'gas_relative_velocities', 'gas_sound_speeds', 'gas_temperatures', 'iron_masses_from_snia', 
    'last_agnfeedback_scale_factors', 'last_high_eddington_fraction_scale_factors', 
    'last_major_merger_scale_factors', 'last_minor_merger_scale_factors', 'last_reposition_velocities', 
    'masses_from_agb', 'masses_from_snia', 'masses_from_snii', 'metal_masses', 'metal_masses_from_agb', 
    'metal_masses_from_snia', 'metal_masses_from_snii', 'number_of_agnevents', 'number_of_direct_swallows', 
    'number_of_gas_neighbours', 'number_of_heating_events', 'number_of_mergers', 
    'number_of_reposition_attempts', 'number_of_repositions', 'number_of_swallows', 'number_of_time_steps', 
    'particle_ids', 'progenitor_particle_ids', 'smoothed_birth_metallicities', 'smoothing_lengths', 
    'split_counts', 'split_trees', 'subgrid_densities', 'subgrid_masses', 'subgrid_sound_speeds', 
    'swallowed_angular_momenta', 'time_bins', 'total_accreted_masses', 'velocities', 'viscosity_factors'
]

kinematic_categories = {
        'cartesian_coordinates': ['x', 'y', 'z', 'xyz'],
        'cartesian_velocities': ['x', 'y', 'z', 'xyz'],
        'spherical_coordinates': ['r', 'radius', 'lon', 'longitude', 'az', 'azimuth', 'phi', 'lat', 'latitude', 'pol', 'polar', 'theta'],
        'spherical_velocities': ['r', 'radius', 'lon', 'longitude', 'az', 'azimuth', 'phi', 'lat', 'latitude', 'pol', 'polar', 'theta'],
        'cylindrical_coordinates': ['R', 'rho', 'radius', 'lon', 'longitude', 'az', 'azimuth', 'phi', 'z', 'height'],
        'cylindrical_velocities': ['R', 'rho', 'radius', 'lon', 'longitude', 'az', 'azimuth', 'phi', 'z', 'height'],        
}    

@dataclass
class BlackHole:
    """A class representing a black hole with various physical properties."""

    def set_fields(self, group_zoom: GroupZoom, particle_index: int) -> None:
        """
        Sets the values of the fields and kinematic categories for the black hole based on the particle index and
        values in the GroupZoom instance.

        Args:
        - group_zoom (GroupZoom): an instance of the GroupZoom class containing information about black holes.
        - particle_index (int): the index of the particle to use for setting the values.

        Returns:
        - None.
        """
        
        for field_name in black_hole_fields:
            
            if hasattr(group_zoom.black_holes, field_name):
                
                array_value = getattr(group_zoom.black_holes, field_name)
                self.convert_to_physical_float(array_value)                
                setattr(self, field_name, array_value[particle_index])
                
        for category_name, coordinates_names in kinematic_categories.items():
            
            category_value = getattr(group_zoom.black_holes, category_name)
            
            for coordinates_name in coordinates_names:
                
                field_name = f"{category_name:s}_{coordinates_name:s}"                
                array_value = getattr(category_value, coordinates_name)
                self.convert_to_physical_float(array_value)
                setattr(self, field_name, array_value[particle_index])
                
    @staticmethod
    def convert_to_physical_float(array: cosmo_array) -> None:
        """
        Converts a cosmo_array object to physical units, if the array is of dtype kind 'f'.

        Args:
            array (cosmo_array): The cosmo_array object to convert to physical units.

        Returns:
            None: This method does not return anything. It modifies the input cosmo_array object in place.

        Raises:
            TypeError: If the input argument is not of type cosmo_array.
        """        
        if array.dtype.kind == 'f':
            array.convert_to_physical()


class BlackHoles:
    
    def __init__(self, group_zoom: GroupZoom):
        
        self.group_zoom = group_zoom
        
        # Generate useful rescaled coordinates
        self.r_500 = group_zoom.halo_finder.spherical_overdensities.r_500_rhocrit.to('Mpc')

        self.group_zoom.black_holes.spherical_coordinates.radius.convert_to_physical()
        self.black_holes_radius_scaled = self.group_zoom.black_holes.spherical_coordinates.radius / self.r_500
        
    def find_central(self, search_radius: unyt.unyt_quantity = 100 * unyt.kpc) -> int:
        """
        Finds the index of the most massive black hole within a given search radius.

        Args:
            search_radius (unyt.unyt_quantity, optional): The search radius within which to look for the most massive black hole.
                Default value is 100 kpc.

        Returns:
            int: The index of the most massive black hole within the search radius.

        Raises:
            None: This method does not raise any exceptions.

        Notes:
            - This method updates the `central_black_hole` attribute of the class with a `BlackHole` object.
            - This method prints a message to the console if the central black hole found is not the most massive one in the simulation.
        """
                                      
        search_radius = numpy_to_cosmo_array(search_radius, self.black_holes_radius_scaled)        
        self.group_zoom.black_holes.subgrid_masses.convert_to_physical()
                
        mask_in_radius = np.where(self.black_holes_radius_scaled < search_radius)[0]
        index_in_mask_most_massive = np.argmax(self.group_zoom.black_holes.subgrid_masses[mask_in_radius])
        index_in_sw_data_central = mask_in_radius[index_in_mask_most_massive]
        selected_bh_mass = self.group_zoom.black_holes.subgrid_masses[index_in_sw_data_central]
        selected_bh_id = self.group_zoom.black_holes.particle_ids[index_in_sw_data_central]
                
        # As debug check, see if that BH is the most massive in the simulation
        largest_bh_index = np.argmax(self.group_zoom.black_holes.subgrid_masses)
        largest_bh_mass = self.group_zoom.black_holes.subgrid_masses[largest_bh_index].to("Msun")
        largest_bh_id = self.group_zoom.black_holes.particle_ids[largest_bh_index]

        if selected_bh_mass < largest_bh_mass and selected_bh_id != largest_bh_id:
            print(
                (
                    f"The central BH found is not the largest in the box:\n"
                    
                    f"\tCentral BH: mass = {selected_bh_mass.to('Msun'):.5E} |"
                    f"ID = {selected_bh_id.value:d} |"
                    f"Distance from the CoP = {self.group_zoom.black_holes.spherical_coordinates.radius[index_in_sw_data_central].to('kpc'):.2f} |"
                    f"Coordinates = {unyt.unyt_array(self.group_zoom.black_holes.coordinates)[index_in_sw_data_central]}\n"
                    
                    f"\tLargest BH: mass = {largest_bh_mass:.5E} |"
                    f"ID = {largest_bh_id.value:d} |"
                    f"Distance from the CoP = {self.group_zoom.black_holes.spherical_coordinates.radius[largest_bh_index].to('kpc'):.2f} |"
                    f"Coordinates = {unyt.unyt_array(self.group_zoom.black_holes.coordinates)[largest_bh_index]}\n"
                )
            )
            
        self.central_black_hole = BlackHole()
        self.central_black_hole.set_fields(self.group_zoom, index_in_sw_data_central)
        
        return index_in_sw_data_central
