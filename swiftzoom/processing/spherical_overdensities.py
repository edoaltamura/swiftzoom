import numpy as np
from scipy.interpolate import interp1d
import unyt

from ..loading import GroupZoom
from .helper_functions import astropy_to_unyt, histogram_unyt, cumsum_unyt, numpy_to_cosmo_array

class SphericalOverdensities:
    """
    This class computes the spherical overdensity r_{{delta}} and associated mass m_{{delta}}, for a given
    group_zoom and density_contrast. 
    """
    
    def __init__(self, group_zoom: GroupZoom, density_contrast: float):
        """
        Initializes an instance of the SphericalOverdensities class by computing the
        spherical overdensity r_{{delta}} and associated mass m_{{delta}} for the given 
        group_zoom and density contrast.
        
        Parameters
        ----------
        group_zoom : GroupZoom
            An instance of the GroupZoom class that contains metadata and properties of a
            simulated galaxy cluster.
        density_contrast : float
            The density contrast against which to compute the spherical overdensity.
        """

        self.group_zoom = group_zoom
        self.density_contrast = density_contrast

        # Find the critical density
        rho_crit = astropy_to_unyt(
            self.group_zoom.metadata.cosmology.critical_density(self.group_zoom.metadata.redshift)
        ).to('Msun/Mpc**3')

        # Convert radius and masses to physical units
        self.group_zoom.gas.spherical_coordinates.radius.convert_to_physical()
        self.group_zoom.dark_matter.spherical_coordinates.radius.convert_to_physical()
        radial_distances_collect = [
            self.group_zoom.gas.spherical_coordinates.radius,
            self.group_zoom.dark_matter.spherical_coordinates.radius,
        ]
        
        self.group_zoom.gas.masses.convert_to_physical()
        self.group_zoom.dark_matter.masses.convert_to_physical()
        masses_collect = [
            self.group_zoom.gas.masses,
            self.group_zoom.dark_matter.masses,
        ]
        
        if self.group_zoom.metadata.n_stars > 0:
            
            self.group_zoom.stars.spherical_coordinates.radius.convert_to_physical()
            radial_distances_collect.append(self.group_zoom.stars.spherical_coordinates.radius)
            
            self.group_zoom.stars.masses.convert_to_physical()
            masses_collect.append(self.group_zoom.stars.masses)
            
        if self.group_zoom.metadata.n_black_holes > 0:
            
            self.group_zoom.black_holes.spherical_coordinates.radius.convert_to_physical()
            radial_distances_collect.append(self.group_zoom.black_holes.spherical_coordinates.radius)
            
            self.group_zoom.black_holes.subgrid_masses.convert_to_physical()
            masses_collect.append(self.group_zoom.black_holes.subgrid_masses)

        # Combine radial distances and masses
        radial_distances = np.concatenate(radial_distances_collect)
        masses = np.concatenate(masses_collect)
        
        del radial_distances_collect, masses_collect
        
        # Convert radial distances and masses to unyt arrays
        radial_distances = unyt.unyt_array(radial_distances, unyt.Mpc)
        masses = unyt.unyt_array(masses, self.group_zoom.units.mass)

        # Define radial bins and shell volumes
        # Choose to scale the limits with the scale factor, to cover also high-z data
        min_radius = 0.01 * self.group_zoom.metadata.a # Mpc
        max_radius = 5 * self.group_zoom.metadata.a # Mpc
        num_bins = int(1000 * self.group_zoom.metadata.a)
        
        lbins = np.logspace(np.log10(min_radius), np.log10(max_radius), num_bins) * unyt.Mpc
        radial_bin_centres = 10.0 ** (0.5 * np.log10(lbins[1:] * lbins[:-1])) * radial_distances.units
        volume_sphere = (4. * np.pi / 3.) * lbins[1:] ** 3

        # Compute mass weights, replace zeros with Nans
        mass_weights = histogram_unyt(radial_distances, bins=lbins, weights=masses)
        mass_weights[mass_weights == 0] = np.nan
        
        # Compute central mass and cumulative mass profile
        central_mass = np.sum(masses[radial_distances < min_radius])
        cumulative_mass_profile = cumsum_unyt(mass_weights) + central_mass
        
        # Compute density profile and interpolations
        density_profile = cumulative_mass_profile / volume_sphere / rho_crit
        
        density_interpolate = interp1d(
            np.log10(density_profile.value),
            np.log10(radial_bin_centres.value),
            kind='cubic'
        )
        mass_interpolate = interp1d(
            np.log10(radial_bin_centres.value),
            np.log10(cumulative_mass_profile.value),
            kind='cubic'
        )

        log_r_delta = density_interpolate(np.log10(self.density_contrast))
        self.r_delta = numpy_to_cosmo_array(10 ** log_r_delta * unyt.Mpc, self.group_zoom.gas.spherical_coordinates.radius)
        self.m_delta = numpy_to_cosmo_array(10 ** mass_interpolate(log_r_delta) * mass_weights.units, self.group_zoom.gas.masses)

    @classmethod
    def delta_2500(cls, group_zoom: GroupZoom) -> 'SphericalOverdensities':
        """Return an instance of SphericalOverdensities with a density contrast of 2500.

        Parameters
        ----------
        group_zoom : GroupZoom
            An instance of the GroupZoom class representing the zoomed region of a cosmological simulation.

        Returns
        -------
        SphericalOverdensities
            An instance of the SphericalOverdensities class with the specified density contrast.

        """
        return cls(group_zoom=group_zoom, density_contrast=2500.)
    
    @classmethod
    def delta_500(cls, group_zoom: GroupZoom) -> 'SphericalOverdensities':
        """Return an instance of SphericalOverdensities with a density contrast of 500.

        Parameters
        ----------
        group_zoom : GroupZoom
            An instance of the GroupZoom class representing the zoomed region of a cosmological simulation.

        Returns
        -------
        SphericalOverdensities
            An instance of the SphericalOverdensities class with the specified density contrast.

        """
        return cls(group_zoom=group_zoom, density_contrast=500.)
    
    @classmethod
    def delta_200(cls, group_zoom: GroupZoom) -> 'SphericalOverdensities':
        """Return an instance of SphericalOverdensities with a density contrast of 200.

        Parameters
        ----------
        group_zoom : GroupZoom
            An instance of the GroupZoom class representing the zoomed region of a cosmological simulation.

        Returns
        -------
        SphericalOverdensities
            An instance of the SphericalOverdensities class with the specified density contrast.

        """
        return cls(group_zoom=group_zoom, density_contrast=200.)
