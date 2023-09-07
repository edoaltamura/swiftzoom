import numpy as np
import unyt
import h5py
from typing import Optional
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.spatial.transform import Rotation

from ..loading import GroupZoom

# from .xray_cloudy import interpolate_X_Ray
from .helper_functions import astropy_to_unyt, histogram_unyt, cumsum_unyt, numpy_to_cosmo_array
from .electron_number_density import get_electron_number_density, get_electron_weighted_gas_mass
from .spherical_overdensities import SphericalOverdensities


"""NOTES

masses = data.gas.masses.value
density = data.gas.densities
volume_proxy = masses/density
volume_proxy = volume_proxy.value

Z = data.gas.metal_mass_fractions.value / 0.0133714 # in solar
v2 = data.gas.velocity_dispersions
sound_velocity2 = 5./3. * kb * data.gas.temperatures / (1.3 * mh)
ratio_v_sound = (v2 / sound_velocity2)**.5

TODO: metallicity profile and individual metals
TODO: PRESSURE PROFILE: pressure = (5./3. -1) * internal_energy * density; pressure.convert_to_units(unyt.g/cm/unyt.s**2)
TODO: shock weighted profiles
"""

# Constants
mean_molecular_weight = 0.5954                  # Mean atomic weight for an ionized gas with primordial composition (X = 0.76, Z = 0)
mean_atomic_weight_per_free_electron = 1.14
primordial_hydrogen_mass_fraction = 0.76
solar_metallicity = 0.0133714


class Profile:
    """
    Base class to generate thermodynamic profiles for galaxy cluster simulations.

    Parameters
    ----------
    group_zoom : GroupZoom
        Group zoom object containing the simulation data.
    nbins : int, optional
        Number of radial bins for profile. Default is 30.
    rmin : float, optional
        Minimum radius for the profile in units of R_500. Default is 0.05.
    rmax : float, optional
        Maximum radius for the profile in units of R_500. Default is 2.0.
    use_alternative_scaling : bool, optional
        If True, an alternative scaling is used. Default is False.
    select_hot_gas : bool, optional
        If True, the hot gas is masked in the profile. Default is True.
    """
    
    def __init__(
            self,
            group_zoom: GroupZoom,
            nbins: int = 30,
            rmin: float = 0.05,
            rmax: float = 2.0,
            use_alternative_scaling: Optional[bool] = False,
            select_hot_gas: Optional[bool] = True
    ):
        
        self.group_zoom = group_zoom
        self.use_alternative_scaling = use_alternative_scaling
        self.select_hot_gas = select_hot_gas
        self.nbins = nbins
        _cosmo_array_template = self.group_zoom.gas.coordinates[0]
        _cosmo_array_template /= _cosmo_array_template
        self.rmin = numpy_to_cosmo_array(rmin, _cosmo_array_template)
        self.rmax = numpy_to_cosmo_array(rmax, _cosmo_array_template)
        
        # Compute log10(rmin) and log10(rmax) for convenience
        self.log_rmin = np.log10(self.rmin)
        self.log_rmax = np.log10(self.rmax)
        
        # Mask all gas in profiles
        self.mask_hot_gas = ... 
        
        if self.select_hot_gas:
            # Mask hot gas in profiles
            self.mask_hot_gas = group_zoom.get_mask_gas_temperature()
                
        # Generate log-spaced bins between the minimum and maximum radii
        lbins = np.logspace(self.log_rmin, self.log_rmax, num=self.nbins + 1, base=10, endpoint=True)
        
        # Convert bins and bin centres to unyt units
        self.bins = lbins * unyt.dimensionless
        self.bin_centres = 10 ** (0.5 * np.log10(lbins[1:] * lbins[:-1])) * unyt.dimensionless
        self.log_bin_centres = np.log10(lbins[1:] * lbins[:-1]) / 2 * unyt.dimensionless
        
        if hasattr(group_zoom.halo_finder.spherical_overdensities, 'r_500_rhocrit'):
            # Get R_500 and M_500 from simulation data
            self.r_500 = group_zoom.halo_finder.spherical_overdensities.r_500_rhocrit.to('Mpc')
            self.mass_500 = group_zoom.halo_finder.spherical_overdensities.mass_500_rhocrit.to('Solar_Mass')
            
        else:
            # Compute R_500 and M_500 from particle data
            print('Spherical overdensities not found in catalogue. Computing from particle data.')
            so = SphericalOverdensities.delta_500(group_zoom)
            self.r_500 = so.r_delta
            self.mass_500 = so.m_delta
            
        # Get R_200 and M_200 from simulation data
        self.r_200 = group_zoom.halo_finder.radii.r_200crit.to('Mpc')
        self.mass_200 = group_zoom.halo_finder.masses.mass_200crit.to('Solar_Mass')
        
        # Generate useful quantities from cosmology and metadata attributes
        self.redshift = self.group_zoom.metadata.redshift
        self.cosmology = self.group_zoom.metadata.cosmology
        
        self.critical_density_z = astropy_to_unyt(self.cosmology.critical_density(self.redshift)).to('Msun/Mpc**3')
        self.critical_density_0 = astropy_to_unyt(self.cosmology.critical_density0).to('Msun/Mpc**3')
        self.e_function_z = self.cosmology.efunc(self.redshift)
        self.baryon_fraction = self.cosmology.Ob0 / self.cosmology.Om0

        # Generate useful rescaled coordinates
        self.group_zoom.gas.spherical_coordinates.radius.convert_to_physical()
        self.gas_radius_scaled = self.group_zoom.gas.spherical_coordinates.radius / self.r_500
        self.shell_volumes = (4 / 3 * np.pi) * self.r_500 ** 3 * (lbins[1:] ** 3 - lbins[:-1] ** 3)
        
    
    def quick_plot_terminal(self, y_values_name: str, y_label: str = 'log10(y)') -> None:
        """
        Plot the given profile quantity using the `plotille` library and print the resulting plot to the terminal.

        Parameters
        ----------
        y_values_name : str
            The name of the attribute of the profile object that contains the y-axis values to plot.
        y_label : str, optional
            The label to use for the y-axis of the plot (default is 'log10(y)').

        Returns
        -------
        None

        Raises
        ------
        ModuleNotFoundError
            If the `plotille` library is not installed.
        ValueError
            If any NaN value is found in the y_values_name attribute, as this cannot be plotted.
        
        Notes
        -----
        This method requires the `plotille` library to be installed.
        """
        try:
            import plotille
        except ImportError:
            raise ModuleNotFoundError(f"The module 'plotille' is not found. You can install it with `pip install plotille`.")
        
        log_y_values = np.log10(getattr(self, y_values_name))
        
        if any(np.isnan(log_y_values)):
            print(getattr(self, y_values_name))
            raise ValueError('Nan encountered in the y_values and cannot plot to the terminal.')
        
        fig = plotille.Figure()
        fig.width = 100
        fig.height = 30
        fig.color_mode = 'byte'
        fig.origin = True
        fig.set_x_limits(min_=np.log10(self.rmin), max_=np.log10(self.rmax))
        fig.set_y_limits(min_=log_y_values.min(), max_=log_y_values.max())
        
        fig.plot(
            np.log10(self.bin_centres), 
            log_y_values, 
            lc=200,
        )
        
        fig.x_label = 'log10(r/r_500)'
        fig.y_label = y_label
        
        print(fig.show())
        
        
    def quick_plot_matplotlib(self, y_values_name: str, y_label: str = 'y', log_y: bool = True, axes = None) -> None:
        """
        Plot a thermodynamic profile using Matplotlib.

        Parameters:
        -----------
        y_values_name: str
            The name of the attribute in `self` that contains the y-axis data to plot.
        y_label: str, optional
            The label for the y-axis of the plot. Default is 'y'.
        log_y: bool, optional
            If True, plot the y-axis on a log scale. Default is True.
        axes: matplotlib.axes.Axes, optional
            An existing axes object to plot onto. If None, a new figure and axes object will be created. Default is None.

        Returns:
        --------
        None
        """
            
        from matplotlib import pyplot as plt
            
        try:
            plt.style.use("mnras.mplstyle")
        except OSError:
            print("The matplotlib stylesheet 'mnras.mplstyle' is not found. Reverting to default.")
            
        # Get the units of the y-axis data and create a LaTeX string for the units label if not dimensionless
        units = getattr(self, y_values_name).units
        units_label = ''
        if units != unyt.dimensionless:
            units_label = f' $\\left[{units.latex_repr:s}\\right]$'
            
        # Create a new figure and axes object if none are provided
        if axes is None:
            fig, axes = plt.subplots()
        else:
            fig = plt.gcf()
            
        # Set the x-axis to log scale
        axes.set_xscale('log')
        # Set the y-axis to log scale if log_y is True
        if log_y:
            axes.set_yscale('log')
        
        # Plot the y-axis data against the bin centres on the x-axis
        axes.plot(self.bin_centres, getattr(self, y_values_name))
        
        # Set the x- and y-axis labels
        axes.set_xlabel(r"$r/r_{500}$")
        axes.set_ylabel(y_label + units_label)
        
        # Add a grid to the plot
        axes.grid(
            color="grey",
            which="both",
            alpha=0.3,
            linewidth=0.3,
            linestyle='--',
            zorder=0,
        )
                
        plt.show()


class ParticleNumberProfile(Profile):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        number_profile = np.histogram(
            self.gas_radius_scaled[self.mask_hot_gas].value,
            bins=self.bins
        )[0]
        
        self.number_profile = number_profile
        
    def get_radius_at_particle_number(self, n_particles: int = 50) -> unyt.unyt_quantity:
        
        nan_mask = np.where(self.number_profile == 0)[0]
        
        if nan_mask.size > 0:                    
            # Clip out bins without particles, starting from the largest radius eligible
            nan_mask_maxid = np.max(nan_mask)
            _bin_centres = self.bin_centres[nan_mask_maxid + 1:]
            _number_profile_masked = self.number_profile[nan_mask_maxid + 1:]
        else:
            _bin_centres = self.bin_centres
            _number_profile_masked = self.number_profile
                
        number_interpolate = interp1d(            
            np.log10(_number_profile_masked),
            np.log10(_bin_centres),
            kind='linear',
            fill_value='extrapolate'
        )
        log_radius_threshold = number_interpolate(np.log10(n_particles))
        
        return np.power(10, log_radius_threshold) * unyt.dimensionless
    
    def get_mask_at_particle_count(self, n_particles: int = 50, boolean: bool = False) -> np.ndarray:
        
        radius_threshold = self.get_radius_at_particle_number(n_particles=n_particles)
        
        if boolean:
            return self.bin_centres >= radius_threshold
        
        return np.where(self.bin_centres >= radius_threshold)[0]


class GasMassProfile(Profile):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                
        self.group_zoom.gas.masses.convert_to_physical()
                
        mass_profile = histogram_unyt(
            self.gas_radius_scaled[self.mask_hot_gas],
            bins=self.bins,
            weights=self.group_zoom.gas.masses[self.mask_hot_gas],
        )
        self.gas_mass_profile = mass_profile.to('Solar_Mass')
        
        
class DarkMatterMassProfile(Profile):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                
        self.group_zoom.dark_matter.masses.convert_to_physical()
        self.group_zoom.dark_matter.spherical_coordinates.radius.convert_to_physical()
                
        mass_profile = histogram_unyt(
            self.group_zoom.dark_matter.spherical_coordinates.radius / self.r_500,
            bins=self.bins,
            weights=self.group_zoom.dark_matter.masses,
        )
        self.dark_matter_mass_profile = mass_profile.to('Solar_Mass')
        

class StarMassProfile(Profile):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                
        self.group_zoom.stars.masses.convert_to_physical()
        self.group_zoom.stars.spherical_coordinates.radius.convert_to_physical()
                
        mass_profile = histogram_unyt(
            self.group_zoom.stars.spherical_coordinates.radius / self.r_500,
            bins=self.bins,
            weights=self.group_zoom.stars.masses,
        )
        self.stars_mass_profile = mass_profile.to('Solar_Mass')


class CumulativeMassProfile(GasMassProfile, DarkMatterMassProfile, StarMassProfile):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                
        self.total_mass_profile = self.gas_mass_profile + self.dark_matter_mass_profile + self.stars_mass_profile
        self.cumulative_mass_profile = cumsum_unyt(self.total_mass_profile)
                    

class DensityProfile(GasMassProfile):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        density_profile = self.gas_mass_profile / self.shell_volumes
        density_profile.convert_to_units(self.critical_density_z.units)
        
        self.density_profile = density_profile
        self.density_profile_scaled = density_profile / self.critical_density_z
        

class TemperatureProfile(GasMassProfile):
    
    def __init__(self, *args, **kwargs):        
        super().__init__(*args, **kwargs)
        
        mass_weighted_temperature_profile = histogram_unyt(
            self.gas_radius_scaled[self.mask_hot_gas],
            bins=self.bins,
            weights=self.group_zoom.gas.temperatures[self.mask_hot_gas],
            normalizer=self.group_zoom.gas.masses[self.mask_hot_gas]
        )
        self.mass_weighted_temperature_profile_kelvin = mass_weighted_temperature_profile
        mass_weighted_temperature_profile = (mass_weighted_temperature_profile * unyt.kb).to('keV')

        kBT500 = unyt.G * mean_molecular_weight * self.mass_500 * unyt.mp / self.r_500 / 2

        if self.use_alternative_scaling:
            kBT500 = (
                unyt.unyt_quantity(1.9, 'keV') * 
                self.e_function_z ** (2 / 3) * 
                (self.mass_500.to(unyt.Solar_Mass).value / 1E14) ** (2 / 3) * 
                (self.group_zoom.metadata.cosmology.h / 0.7) ** (2 / 3)
            )

        self.kBT500 = kBT500.to('keV')
        
        self.mass_weighted_temperature_profile_kev = mass_weighted_temperature_profile
        self.mass_weighted_temperature_profile_scaled = mass_weighted_temperature_profile / self.kBT500
    
        
class ElectronNumberDensityProfile(Profile):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # electron_corrected_mass_profile = self.gas_mass_profile / mean_atomic_weight_per_free_electron
        electron_corrected_mass_profile = histogram_unyt(
                    self.gas_radius_scaled[self.mask_hot_gas],
                    bins=self.bins,
                    weights=get_electron_weighted_gas_mass(self.group_zoom)[self.mask_hot_gas],
                )
        
        electron_corrected_density_profile = electron_corrected_mass_profile / self.shell_volumes
        electron_corrected_density_profile.convert_to_units(self.critical_density_z.units)
        number_density_profile = electron_corrected_density_profile.to('g*cm**-3') / unyt.mp
        
        ne500 = 500 * self.baryon_fraction * self.critical_density_z / (mean_atomic_weight_per_free_electron * unyt.mp)
        self.ne500 = ne500.to('1/cm**3')
        
        self.number_density_profile = number_density_profile
        self.number_density_profile_scaled = number_density_profile / self.ne500            


class EntropyProfile(TemperatureProfile, ElectronNumberDensityProfile):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
         
        entropy_profile = self.mass_weighted_temperature_profile_kev / (self.number_density_profile ** (2 / 3))
        entropy_profile.convert_to_units('keV*cm**2')
        
        K500 = self.kBT500 / (self.ne500 ** (2 / 3))
        
        if self.use_alternative_scaling:
            K500 = (
                unyt.unyt_quantity(370, 'keV*cm**2') * 
                self.e_function_z ** (-2 / 3) * 
                (self.mass_500.to(unyt.Solar_Mass).value / 1E14) ** (2 / 3) * 
                (self.baryon_fraction / 0.15) ** (-2 / 3) * 
                (self.group_zoom.metadata.cosmology.h / 0.7) ** (-2 / 3)
            )
            
        self.K500 = K500.to('keV*cm**2')
        
        self.entropy_profile = entropy_profile
        self.entropy_profile_scaled = entropy_profile / self.K500
        
        
class EntropyGradientProfile(EntropyProfile):
    
    def __init__(self, *args, **kwargs):
        
        nbins = kwargs.get("nbins")
        rmin = kwargs.get("rmin")
        rmax = kwargs.get("rmax")
        
        new_nbins = nbins + 1
        new_rmin = 10 ** (np.log10(rmin) - (np.log10(rmax) - np.log10(rmin)) / (new_nbins + 1) / 2)
        new_rmax = 10 ** (np.log10(rmax) + (np.log10(rmax) - np.log10(rmin)) / (new_nbins + 1) / 2)
        
        kwargs.update(dict(nbins=new_nbins, rmin=new_rmin, rmax=new_rmax))
        
        super().__init__(*args, **kwargs)
        
        delta_log_entropy = np.log10(self.entropy_profile_scaled.value)[1:] - np.log10(self.entropy_profile_scaled.value)[:-1]
        delta_log_radius = self.log_bin_centres[1:] - self.log_bin_centres[:-1]
        
        self._noncontracted_log_bin_centres = self.log_bin_centres
        self._noncontracted_bin_centres = self.bin_centres
        
        self.log_bin_centres = (self.log_bin_centres[1:] + self.log_bin_centres[:-1]) / 2 * unyt.dimensionless
        self.bin_centres = 10 ** ((self.log_bin_centres[1:] + self.log_bin_centres[:-1]) / 2) * unyt.dimensionless
        entropy_gradient_profile = delta_log_entropy / delta_log_radius
        
        self.entropy_gradient_profile = entropy_gradient_profile * unyt.dimensionless
        
        
class DynamicalTimeProfile(CumulativeMassProfile):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        radius_physical = self.bin_centres * self.r_500
        self.gravitational_acceleration_profile = unyt.G * self.cumulative_mass_profile / radius_physical ** 2
        
        self.log_dynamical_time_profile = 0.5 * np.log10((2 * radius_physical / self.gravitational_acceleration_profile).to('Myr**2')) * unyt.dimensionless
        self.dynamical_time_profile = np.sqrt(2 * radius_physical / self.gravitational_acceleration_profile).to('Myr')
        

class FreeFallTimeProfile(GasMassProfile):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.group_zoom.gas.densities.convert_to_physical()
        free_fall_times = np.sqrt(3 * np.pi / (32 * unyt.G * self.group_zoom.gas.densities)).to('Myr')

        free_fall_time_profile = histogram_unyt(
            self.gas_radius_scaled[self.mask_hot_gas],
            bins=self.bins,
            weights=free_fall_times[self.mask_hot_gas],
            normalizer=self.group_zoom.gas.masses[self.mask_hot_gas]
        )
        
        self.free_fall_time_profile = free_fall_time_profile
        
        
class CoolingTimeProfile(GasMassProfile):
    
    cooling_table_path = '/cosma/home/dp004/dc-alta2/data7/xl-zooms/hydro/UV_dust1_CR1_G1_shield1.hdf5'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.load_rates()
        self.compute_cooling_rates()
        self.set_cooling_times()    
        
        cooling_time_profile = histogram_unyt(
            self.gas_radius_scaled[self.mask_hot_gas],
            bins=self.bins,
            weights=self.log_cooling_times_myr[self.mask_hot_gas],
            normalizer=self.group_zoom.gas.masses[self.mask_hot_gas]
        )
        
        self.log_cooling_time_profile = cooling_time_profile * unyt.dimensionless
        self.cooling_time_profile = 10 ** cooling_time_profile * unyt.Myr
        
        
    def load_rates(self):
        
        with h5py.File(self.cooling_table_path, "r") as cooling_table:
            
            self.cooling_rates = np.log10(
                np.power(10., cooling_table["/Tdep/Cooling"][0, :, :, :, -2]) + 
                np.power(10., cooling_table["/Tdep/Cooling"][0, :, :, :, -1])
            )
            self.heating_rates = np.log10(
                np.power(10., cooling_table["/Tdep/Heating"][0, :, :, :, -2]) + 
                np.power(10., cooling_table["/Tdep/Heating"][0, :, :, :, -1])
            )
            
            # Get the axes grids
            self.density_bins = cooling_table["/TableBins/DensityBins"][:]
            self.U_bins = cooling_table["/TableBins/InternalEnergyBins"][:]
            self.Z_bins = cooling_table["/TableBins/MetallicityBins"][:]
            self.z_bins = cooling_table["/TableBins/RedshiftBins"][:]
            self.T_bins = cooling_table["/TableBins/TemperatureBins"][:]
        
        self.net_rates = np.log10(np.abs(np.power(10., self.heating_rates) - np.power(10., self.cooling_rates)))
        
    
    def compute_cooling_rates(self):
        
        self.group_zoom.gas.densities.convert_to_physical()
        
        interpolate_net_rates = RegularGridInterpolator(
            (self.T_bins, self.Z_bins, self.density_bins),
            self.net_rates,
            method="linear",
            bounds_error=False,
            fill_value=-30
        )
        
        gas_nH = (
            self.group_zoom.gas.densities / unyt.mp * 
            self.group_zoom.gas.element_mass_fractions.hydrogen
        ).to(unyt.cm ** -3)
        
        log_gas_nH = np.log10(gas_nH)
        log_gas_T = np.log10(self.group_zoom.gas.temperatures)
        
        with np.errstate(divide='ignore'):
            log_gas_Z = np.log10(self.group_zoom.gas.metal_mass_fractions.value / solar_metallicity)
            
        # construct the matrix that we input in the interpolator
        values_to_int = np.zeros((len(log_gas_T), 3))
        values_to_int[:, 0] = log_gas_T
        values_to_int[:, 1] = log_gas_Z
        values_to_int[:, 2] = log_gas_nH

        net_rates_found = interpolate_net_rates(values_to_int)

        cooling_times = np.log10(3 / 2 * 1.38e-16) + log_gas_T - log_gas_nH - net_rates_found - np.log10(3.154e13)
        
        self.log_cooling_times_myr = cooling_times * unyt.dimensionless
        self.cooling_times = 10 ** cooling_times * unyt.Myr
        
    def set_cooling_times(self):
        
        setattr(self.group_zoom.gas, 'cooling_times', self.cooling_times)
        
            
class DynamicalOverCoolingTimeProfile(DynamicalTimeProfile, CoolingTimeProfile):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dynamical_over_cooling_time_profile = self.dynamical_time_profile / self.cooling_time_profile
        self.cooling_over_dynamical_time_profile = self.cooling_time_profile / self.dynamical_time_profile
        self.dynamical_over_cooling_time_profile_squared = self.dynamical_over_cooling_time_profile ** 2
        
        
class XrayEmissivityProfile(GasMassProfile):
    
    cooling_table_path = '/cosma/home/dp004/dc-alta2/data7/xl-zooms/hydro/UV_dust1_CR1_G1_shield1.hdf5'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.compute_emissivities()
        self.set_emissivities()  
        
        xray_luminosity_profile = histogram_unyt(
            self.gas_radius_scaled[self.mask_hot_gas],
            bins=self.bins,
            weights=self.xray_luminosities[self.mask_hot_gas],
            normalizer=self.group_zoom.gas.masses[self.mask_hot_gas]
            
        )
        
        self.xray_luminosity_profile = xray_luminosity_profile
        
    def compute_emissivities(self):

        raise NotImplementedError
        
        self.group_zoom.gas.densities.convert_to_physical()
        
        data_nH = np.log10(self.group_zoom.gas.element_mass_fractions.hydrogen * self.group_zoom.gas.densities.to('g*cm**-3') / unyt.mp)
        data_T = np.log10(self.group_zoom.gas.temperatures.value)

        # Interpolate the Cloudy table to get emissivities
        log_emissivities = interpolate_X_Ray(
                data_nH,
                data_T,
                self.group_zoom.gas.element_mass_fractions,
                fill_value=-50.
            )
        self.emissivities = unyt.unyt_array(10 ** log_emissivities, 'erg/s/cm**3')
        self.xray_luminosities = emissivities * self.group_zoom.gas.masses / self.group_zoom.gas.densities
        
        
    def set_emissivities(self):
        
        setattr(self.group_zoom.gas, 'emissivities', self.emissivities)
        setattr(self.group_zoom.gas, 'xray_luminosities', self.xray_luminosities)
        

class SphericalSectorSampler:
    
    def __init__(self):
        pass
    
    @staticmethod
    def spherical_sector_mask(points, opening_angle_degrees: float) -> np.ndarray:
        """
        Generate a spherical sector from a set of points.

        This function rotates the input points by a random rotation matrix, and then
        selects the points within a specified angle of the zenith. It also calculates
        the volume of the resulting spherical sector.

        Parameters:
            points (np.ndarray): An Nx3 array of points in 3D space.
            opening_angle_degrees (float): The angle of the spherical sector, in degrees.

        Returns:
            sector_points (np.ndarray): The points within the spherical sector.
            sector_volume (float): The volume of the spherical sector.
        """
        
        # Generate a random rotation matrix
        rot = Rotation.random()

        # Rotate the points
        rotated_points = rot.apply(points)

        # Calculate the zenith angles of the rotated points
        zenith_angles = np.arccos(rotated_points[:, 2] / np.linalg.norm(rotated_points, axis=1))
        
        return zenith_angles <= np.radians(opening_angle_degrees / 2)
    
    
    @staticmethod
    def calculate_sector_volumes(opening_angle_degrees: float, radial_bins: np.ndarray) -> np.ndarray:
        """
        Calculates the volumes of the spherical sectors for each radial bin, given an opening angle in degrees.

        Parameters:
            opening_angle_degrees (float): The opening angle of the sectors in degrees.
            radial_bins (numpy.ndarray): An array of radial bins.

        Returns:
            numpy.ndarray: An array of sector volumes, one for each radial bin.

        Raises:
            ValueError: If the opening angle is less than 0 or greater than 180 degrees, or if the radial bins are not monotonically increasing.

        Example:
            >>> calculate_sector_volumes(45.0, np.array([1.0, 2.0, 3.0]))
            array([0.0314, 0.2513, 0.8466])
        """
        if opening_angle_degrees < 0 or opening_angle_degrees > 180:
            raise ValueError("Opening angle must be between 0 and 180 degrees.")
        
        if not np.all(np.diff(radial_bins) > 0):
            raise ValueError("Radial bins must be monotonically increasing.")
        
        half_opening_angle_radians = np.deg2rad(opening_angle_degrees / 2)
        sector_volumes = 2 / 3 * np.pi * radial_bins ** 3 * (1 - np.cos(half_opening_angle_radians))
        
        return sector_volumes[:-1] - sector_volumes[1:]


if __name__ == '__main__':
    run_directory = '/cosma/home/dp004/dc-alta2/shocks_paper_data/VR18_-8res_Ref/'
    redshift = 0.
    
    zoom = GroupZoom(run_directory, redshift=redshift)
    entropy_profile_object = EntropyProfile(zoom, nbins=30, rmin=0.01, rmax=2.5)   
    entropy_profile_object.quick_plot_terminal('entropy_profile_scaled', y_label='log10(K/K_500)')
