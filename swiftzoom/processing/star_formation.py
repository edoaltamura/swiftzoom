import numpy as np
import unyt
from astropy.units import Gyr as astropy_Gyr
from astropy.cosmology import z_at_value

from loader import GroupZoom
from .helper_functions import numpy_to_cosmo_array


class StarFormationRate:
    """
    Computes the star formation rate and related properties.

    Args:
        group_zoom (GroupZoom): The GroupZoom object containing the simulation data.
        time_window (unyt.unyt_quantity, optional): The time window to consider for star formation rate calculation.
            Default is 10 Myr.
        bcg_radius_r500 (float, optional): The fraction of R_500 to define the radius of the BCG. Default is 0.2.
        
    Attributes:
        group_zoom (GroupZoom): The GroupZoom object containing the simulation data.
        r_500 (unyt.unyt_quantity): The R_500 radius in physical units.
        bcg_radius (unyt.unyt_array): The radius of the BCG in rescaled coordinates.
        mass_sfr (unyt.unyt_quantity): The total mass of stars formed within the BCG region.
        mass_bcg (unyt.unyt_quantity): The total mass of stars within the BCG region.
        sfr (unyt.unyt_quantity): The star formation rate within the BCG region.
        specific_sfr (unyt.unyt_quantity): The specific star formation rate within the BCG region.
        mass_stars_r500 (unyt.unyt_quantity): The total mass of all stars within R_500.
        mass_fraction_bcg (unyt.unyt_quantity): The fraction of BCG mass to the total mass within R_500.

    Examples:
        # Create a GroupZoom object
        group_zoom = GroupZoom()

        # Instantiate the StarFormationRate object
        sfr = StarFormationRate(group_zoom)

        # Access the computed properties
        print(f"BCG Star Formation Rate: {sfr.sfr}")
        print(f"Specific Star Formation Rate: {sfr.specific_sfr}")
    """
    
    def __init__(self, group_zoom: GroupZoom,
                 time_window: unyt.unyt_quantity = 10 * unyt.Myr,
                 bcg_radius_metric: str = 'physical',
                 bcg_radius: float = 0.05):
        """
        Initialize the StarFormationRate object.

        Args:
            group_zoom (GroupZoom): The GroupZoom object containing the simulation data.
            time_window (unyt.unyt_quantity, optional): The time window to consider for star formation rate calculation.
                Default is 10 Myr.
            bcg_radius_r500 (float, optional): The fraction of R_500 to define the radius of the BCG. Default is 0.2.
            
        Returns:
            None
        """
        
        if not bcg_radius_metric in ['r500', 'physical']:
            raise ValueError(f"The `bcg_radius_metric` value must be in ['r500', 'physical'], got {bcg_radius_metric}.")
        
        self.group_zoom = group_zoom
        
        # Generate useful rescaled coordinates
        self.group_zoom.stars.spherical_coordinates.radius.convert_to_physical()        
        self.group_zoom.stars.masses.convert_to_physical()
        
        self.r_500 = group_zoom.halo_finder.spherical_overdensities.r_500_rhocrit.to('Mpc')
        
        if bcg_radius_metric == 'physical':            
            self.bcg_radius = numpy_to_cosmo_array(bcg_radius * unyt.Mpc, self.group_zoom.stars.spherical_coordinates.radius)
            
        elif bcg_radius_metric == 'r500':            
            self.bcg_radius = numpy_to_cosmo_array(self.r_500 * bcg_radius, self.group_zoom.stars.spherical_coordinates.radius)        

        # Compute scale factor for cutoff star birth
        time_end = self.group_zoom.metadata.cosmology.age(self.group_zoom.metadata.redshift)
        time_start = time_end.to("Gyr").value - time_window.to("Gyr").value
        time_start *= astropy_Gyr
        
        z_start = z_at_value(self.group_zoom.metadata.cosmology.age, time_start)
        a_start = 1 / (1 + z_start)
        a_start = numpy_to_cosmo_array(np.array(a_start), self.group_zoom.stars.birth_scale_factors)

        # Select stars in BCG born after scale factor selected above        
        mask_sfr = np.where((self.group_zoom.stars.spherical_coordinates.radius < self.bcg_radius) &
                            (self.group_zoom.stars.birth_scale_factors >= a_start))[0]
        
        if mask_sfr.size == 0:
            print(f'No stars formed between scale factors a = [{a_start.value:.4f}, {self.group_zoom.metadata.scale_factor:.4f}]')
        
        mask_bcg = np.where(self.group_zoom.stars.spherical_coordinates.radius < self.bcg_radius)[0]

        self.mass_sfr = np.sum(self.group_zoom.stars.masses[mask_sfr]).to("Solar_Mass")
        self.mass_bcg = np.sum(self.group_zoom.stars.masses[mask_bcg]).to("Solar_Mass")
        self.sfr = (self.mass_sfr / time_window).to("Solar_Mass/yr")
        self.specific_sfr = (self.sfr / self.mass_bcg).to("1/Gyr")
        
        mask_all_stars = np.where(self.group_zoom.stars.spherical_coordinates.radius < self.r_500)[0]
        self.mass_stars_r500 = np.sum(self.group_zoom.stars.masses[mask_all_stars]).to("Solar_Mass")
        self.mass_fraction_bcg = self.mass_bcg / self.mass_stars_r500
        
