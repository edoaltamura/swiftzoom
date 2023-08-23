import numpy as np
import unyt
from astropy.units import Gyr as astropy_Gyr
from astropy.cosmology import z_at_value

from loader import GroupZoom
from .helper_functions import numpy_to_cosmo_array

# Constants
mean_molecular_weight = 0.5954                  # Mean atomic weight for an ionized gas with primordial composition (X = 0.76, Z = 0)
gamma = 5 / 3                                   # Gas adiabatic index (assuming monoatomic)

class HotGas:
    """
    Class representing hot gas properties.

    Parameters:
        group_zoom (GroupZoom): The GroupZoom object containing gas data.
        tmin (unyt.unyt_quantity): The minimum temperature for considering gas as hot.

    Attributes:
        group_zoom (GroupZoom): The GroupZoom object containing the simulation data.
        r_500 (unyt.unyt_quantity): The R_500 radius in physical units.
        tmin (unyt.unyt_quantity): The minimum temperature threshold for AGN heating.
        hot_gas_fraction (unyt.unyt_quantity): The fraction of hot gas.
        number_agn_heated_particles (int): The number of gas particles heated by AGN.
        mass_heated_by_agn (unyt.unyt_quantity): The total mass of gas heated by AGN.
        mass_heated_by_agn_fraction (unyt.unyt_quantity): The fraction of gas heated by AGN relative to the hot gas fraction.
        median_entropy_before_agn_heated (unyt.unyt_quantity): The median entropy of hot gas before the last AGN event.
        median_entropy_after_agn_heated (unyt.unyt_quantity): The median entropy of hot gas after the last AGN event.
        median_entropy_jump_heated (unyt.unyt_quantity): The median entropy jump of hot gas at the last AGN event.
        number_agn_heated_particles_interval (int): The number of gas particles heated by AGN within the specified time window.
        mass_heated_by_agn_interval (unyt.unyt_quantity): The total mass of gas heated by AGN within the specified time window.
        mass_heated_by_agn_fraction_interval (unyt.unyt_quantity): The fraction of gas heated by AGN within the specified time window relative to the hot gas fraction.
        median_entropy_before_agn_heated_interval (unyt.unyt_quantity): The median entropy of hot gas before the last AGN event within the specified time window.
        median_entropy_after_agn_heated_interval (unyt.unyt_quantity): The median entropy of hot gas after the last AGN event within the specified time window.
        median_entropy_jump_heated_interval (unyt.unyt_quantity): The median entropy jump of hot gas at the last AGN event within the specified time window.
        number_snii_heated_particles (int): The number of gas particles heated by SNII.
        mass_heated_by_snii (unyt.unyt_quantity): The total mass of gas heated by SNII.
        mass_heated_by_snii_fraction (unyt.unyt_quantity): The fraction of gas heated by SNII relative to the hot gas fraction.

    Examples:
        # Create a GroupZoom object
        group_zoom = GroupZoom(...)
        
        # Create an HotGas object
        agn_mass_heating = HotGas(group_zoom, time_window=20 * unyt.Myr)
        
        # Access the computed properties
        print(f"Number of gas particles heated by AGN: {agn_mass_heating.number_agn_heated_particles}")
        print(f"Total mass of gas heated by AGN: {agn_mass_heating.mass_heated_by_agn:.2e}")
        print(f"Fraction of gas heated by AGN relative to hot gas: {agn_mass_heating.mass_heated_by_agn_fraction:.4f}")
        print(f"Median entropy of hot gas before the last AGN event: {agn_mass_heating.median_entropy_before_agn_heated:.4f}")
        print(f"Median entropy of hot gas after the last AGN event: {agn_mass_heating.median_entropy_after_agn_heated:.4f}")
        print(f"Median entropy jump of hot gas at the last AGN event: {agn_mass_heating.median_entropy_jump_heated:.4f}")
        print(f"Number of gas particles heated by AGN within the specified time window: {agn_mass_heating.number_agn_heated_particles_interval}")
        print(f"Total mass of gas heated by AGN within the specified time window: {agn_mass_heating.mass_heated_by_agn_interval:.2e}")
        print(f"Fraction of gas heated by AGN within the specified time window relative to hot gas: {agn_mass_heating.mass_heated_by_agn_fraction_interval:.4f}")
        print(f"Median entropy of hot gas before the last AGN event within the specified time window: {agn_mass_heating.median_entropy_before_agn_heated_interval:.4f}")
        print(f"Median entropy of hot gas after the last AGN event within the specified time window: {agn_mass_heating.median_entropy_after_agn_heated_interval:.4f}")
        print(f"Median entropy jump of hot gas at the last AGN event within the specified time window: {agn_mass_heating.median_entropy_jump_heated_interval:.4f}")
        print(f"Number of gas particles heated by SNII: {agn_mass_heating.number_snii_heated_particles}")
        print(f"Total mass of gas heated by SNII: {agn_mass_heating.mass_heated_by_snii:.2e}")
        print(f"Fraction of gas heated by SNII relative to hot gas: {agn_mass_heating.mass_heated_by_snii_fraction:.4f}")

    """
    
    def __init__(self, group_zoom: GroupZoom,
                 tmin: unyt.unyt_quantity = 1.e5 * unyt.K):
        """
        Initialize HotGas object.

        Args:
            group_zoom (GroupZoom): The GroupZoom object containing gas data.
            tmin (unyt.unyt_quantity): The minimum temperature for considering gas as hot.
            
        Returns:
            None
            
        Raises:
            None
        """
        self.group_zoom = group_zoom
        self.tmin = numpy_to_cosmo_array(tmin, self.group_zoom.gas.temperatures)
        
        # Self-similar quantities
        self.r_500 = group_zoom.halo_finder.spherical_overdensities.r_500_rhocrit.to('Mpc')
        self.mass_500 = group_zoom.halo_finder.spherical_overdensities.mass_500_rhocrit.to('Solar_Mass')
        
        # Generate useful rescaled coordinates
        self.group_zoom.gas.spherical_coordinates.radius.convert_to_physical()        
        self.group_zoom.gas.masses.convert_to_physical()
        
        # Compute hot gas properties
        mask_hot_gas = np.where((self.group_zoom.gas.spherical_coordinates.radius < self.r_500) & 
                                (self.group_zoom.gas.temperatures > self.tmin))[0]
        
        self.mass_hot_gas = np.sum(self.group_zoom.gas.masses[mask_hot_gas]).to("Solar_Mass")
        self.hot_gas_fraction = self.mass_hot_gas / self.mass_500
        
        # Compute cold gas properties
        mask_cold_gas = np.where((self.group_zoom.gas.spherical_coordinates.radius < self.r_500) & 
                                (self.group_zoom.gas.temperatures < self.tmin))[0]
        
        self.mass_cold_gas = np.sum(self.group_zoom.gas.masses[mask_cold_gas]).to("Solar_Mass")
        self.cold_gas_fraction = self.mass_cold_gas / self.mass_500
                
    def _set_entropies_agn(self, mask = ..., suffix: str = "") -> None:
        
        if len(suffix) > 0:
            suffix = f"_{suffix:s}"
            
        if mask.size == 0:
            setattr(self, f"median_entropy_before_agn{suffix:s}", np.nan)
            setattr(self, f"median_entropy_at_agn{suffix:s}", np.nan)
            setattr(self, f"median_entropy_jump{suffix:s}", np.nan)
            return
        
        self.group_zoom.gas.entropies_before_last_agnevent.convert_to_physical()
        self.group_zoom.gas.densities_before_last_agnevent.convert_to_physical()
        self.group_zoom.gas.entropies_at_last_agnevent.convert_to_physical()
        self.group_zoom.gas.densities_at_last_agnevent.convert_to_physical()
        
        entropy = self.group_zoom.gas.entropies_before_last_agnevent[mask]
        density = self.group_zoom.gas.densities_before_last_agnevent[mask]
        electron_number_density = (density / unyt.mh / mean_molecular_weight).to('cm**-3')        
        temperature = mean_molecular_weight * (gamma - 1) * (entropy * density ** (gamma - 1)) / (gamma - 1) * unyt.mh / unyt.kb
        entropy_before = unyt.kb * temperature / electron_number_density ** (2 / 3)
        entropy_before.convert_to_units('keV*cm**2')
        median_entropy = unyt.unyt_array(np.percentile(entropy_before, 50), entropy_before.units)
        setattr(self, f"median_entropy_before_agn{suffix:s}", median_entropy)
        
        entropy = self.group_zoom.gas.entropies_at_last_agnevent[mask]
        density = self.group_zoom.gas.densities_at_last_agnevent[mask]
        electron_number_density = (density / unyt.mh / mean_molecular_weight).to('cm**-3')        
        temperature = mean_molecular_weight * (gamma - 1) * (entropy * density ** (gamma - 1)) / (gamma - 1) * unyt.mh / unyt.kb
        entropy_after = unyt.kb * temperature / electron_number_density ** (2 / 3)
        entropy_after.convert_to_units('keV*cm**2')
        median_entropy = unyt.unyt_array(np.percentile(entropy_after, 50), entropy_after.units)
        setattr(self, f"median_entropy_at_agn{suffix:s}", median_entropy)
        
        entropy_difference = entropy_after - entropy_before
        median_entropy = unyt.unyt_array(np.percentile(entropy_difference, 50), entropy_difference.units)
        setattr(self, f"median_entropy_jump{suffix:s}", median_entropy)
        
        
    def get_mass_heated_by_agn(self, time_window: unyt.unyt_quantity = 50 * unyt.Myr):
        """
        Compute the mass of gas heated by AGN.

        Args:
            time_window (unyt.unyt_quantity): The time window to consider for computing the mass of gas heated by AGN.
                                              Default is 10 Myr.

        Returns:
            None
            
        Raises:
            None
        """
        
        # Cumulative AGN events before this snapshot's redshift        
        mask_heated_gas = np.where((self.group_zoom.gas.spherical_coordinates.radius < self.r_500) & 
                                   (self.group_zoom.gas.temperatures > self.tmin) &
                                   (self.group_zoom.gas.heated_by_agnfeedback.value > 0) &
                                   (self.group_zoom.gas.densities_at_last_agnevent.value > 0))[0]
        
        self.number_agn_heated_particles = mask_heated_gas.size
        
        if self.number_agn_heated_particles == 0:
            print(f'No AGN events before redshift z = {self.group_zoom.metadata.redshift:.3f}')
        
        self.mass_heated_by_agn = np.sum(self.group_zoom.gas.masses[mask_heated_gas]).to("Solar_Mass")
        self.mass_heated_by_agn_fraction = self.mass_heated_by_agn / self.hot_gas_fraction
        
        # Entropies of hot gas heated by AGN
        self._set_entropies_agn(mask=mask_heated_gas, suffix="heated")
        
        # Compute scale factor for cutoff last AGN event
        time_end = self.group_zoom.metadata.cosmology.age(self.group_zoom.metadata.redshift)
        time_start = time_end.to("Gyr").value - time_window.to("Gyr").value
        time_start *= astropy_Gyr
        
        z_start = z_at_value(self.group_zoom.metadata.cosmology.age, time_start)
        a_start = 1 / (1 + z_start)
        a_start = numpy_to_cosmo_array(np.array(a_start), self.group_zoom.gas.last_agnfeedback_scale_factors)
        
        mask_heated_gas_interval = np.where(
            (self.group_zoom.gas.spherical_coordinates.radius < self.r_500) & 
            (self.group_zoom.gas.heated_by_agnfeedback.value > 0) &
            (self.group_zoom.gas.last_agnfeedback_scale_factors >= a_start) &
            (self.group_zoom.gas.densities_at_last_agnevent.value > 0)
        )[0]
        
        self.number_agn_heated_particles_interval = mask_heated_gas_interval.size
        
        if self.number_agn_heated_particles_interval == 0:
            print(f'No AGN events between scale-factors a = [{a_start.value:.4f}, {self.group_zoom.metadata.scale_factor:.4f}]')
        
        self.mass_heated_by_agn_interval = np.sum(self.group_zoom.gas.masses[mask_heated_gas_interval]).to("Solar_Mass")
        self.mass_heated_by_agn_fraction_interval = self.mass_heated_by_agn_interval / self.hot_gas_fraction
        
        # Entropies of hot gas heated by AGN
        self._set_entropies_agn(mask=mask_heated_gas, suffix="heated_interval")
        
    def get_mass_heated_by_snii(self):
        """
        Compute the mass of gas heated by SNII.
        
        Returns:
            None
            
        Raises:
            None
        """
        
        # Cumulative AGN events before this snapshot's redshift        
        mask_heated_gas = np.where((self.group_zoom.gas.spherical_coordinates.radius < self.r_500) & 
                                   (self.group_zoom.gas.heated_by_sniifeedback.value > 0))[0]
        
        self.number_snii_heated_particles = mask_heated_gas.size
        
        if self.number_snii_heated_particles == 0:
            print(f'No SNII events before scale-factor a = {self.group_zoom.metadata.scale_factor:.4f}')
        
        self.mass_heated_by_snii = np.sum(self.group_zoom.gas.masses[mask_heated_gas]).to("Solar_Mass")
        self.mass_heated_by_snii_fraction = self.mass_heated_by_snii / self.mass_hot_gas
