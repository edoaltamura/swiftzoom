import numpy as np
import unyt
from swiftsimio import cosmo_array
from astropy.cosmology import z_at_value

from loader import GroupZoom


class StarFormationRate:
    
    def __init__(self, group_zoom: GroupZoom,
                 time_window: unyt.unyt_quantity = 10 * unyt.Myr,
                 bcg_radius_r500: float = 0.2):
        
        self.group_zoom = group_zoom
        
        # Generate useful rescaled coordinates
        self.r_500 = group_zoom.halo_finder.spherical_overdensities.r_500_rhocrit.to('Mpc')
        self.bcg_radius = self.r_500 * bcg_radius_r500
        
        self.group_zoom.stars.spherical_coordinates.radius.convert_to_physical()        
        self.group_zoom.stars.masses.convert_to_physical()
        
        # Compute scale factor for cutoff star birth
        time_end = self.group_zoom.metadata.cosmology.age(self.group_zoom.metadata.redshift)
        time_start = time_end.to("Gyr").value - time_window.to("Gyr").value
        time_start *= Gyr

        z_start = z_at_value(self.group_zoom.metadata.cosmology.age, time_start)
        a_start = 1 / (1 + z_start)

        # Select stars in BCG born after scale factor selected above
        mask_sfr = np.where(
            (self.group_zoom.stars.spherical_coordinates.radius <= self.bcg_radius) &
            (self.group_zoom.stars.birth_scale_factors > a_start)
        )[0]
        mask_bcg = np.where(
            self.group_zoom.stars.spherical_coordinates.radius <= self.bcg_radius
        )[0]

        self.mass_sfr = np.sum(self.group_zoom.stars.masses[mask_sfr]).to("Msun")
        self.mass_bcg = np.sum(self.group_zoom.stars.masses[mask_bcg]).to("Msun")
        self.sfr = (self.mass_sfr / time_window).to("Msun/yr")
        self.specific_sfr = (self.sfr / self.mass_bcg).to("1/Gyr")
        
        
