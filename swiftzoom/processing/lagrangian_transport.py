from typing import Optional, Union, Dict, Tuple
import warnings
import unyt
import numpy as np
from swiftsimio import cosmo_array, objects

from ..loading import GroupZoom
from ..loading.constants import mean_molecular_weight, mean_atomic_weight_per_free_electron

from .helper_functions import astropy_to_unyt, numpy_to_cosmo_array, cosmo_to_unyt_array
from .spherical_overdensities import SphericalOverdensities
from .electron_number_density import get_electron_number_density


def check_mask_empty_nan(func):
        
    def wrapper(*args, **kwargs):
        self = args[0]  # assuming self is the first argument
        mask = kwargs.get('mask')
        
        if not mask is Ellipsis and mask.size == 0:
            warnings.warn("Empty particle mask detected. Returning zero value.", FutureWarning)
            return 0. * unyt.dimensionless

        return func(*args, **kwargs)
    
    return wrapper


class ParticleTracker:
    
    def __init__(self, reference_group_zoom: GroupZoom):
        
        
        self.reference_group_zoom = reference_group_zoom
        assert self.reference_group_zoom.import_all_particles, \
            "Lagrangian tracking requires importing all particles to avoid losing any. Set GroupZoom(..., import_all_particles=True)."

        if hasattr(reference_group_zoom.halo_finder.spherical_overdensities, 'r_500_rhocrit'):
            # Get R_500 and M_500 from simulation data
            self.r_500 = reference_group_zoom.halo_finder.spherical_overdensities.r_500_rhocrit.to('Mpc')
            self.mass_500 = reference_group_zoom.halo_finder.spherical_overdensities.mass_500_rhocrit.to('Solar_Mass')
            
        else:
            # Compute R_500 and M_500 from particle data
            warnings.warn("Spherical overdensities not found in catalogue. Computing from particle data.", RuntimeWarning)
            so = SphericalOverdensities.delta_500(reference_group_zoom)
            self.r_500 = so.r_delta
            self.mass_500 = so.m_delta
        
        # Generate useful quantities from cosmology and metadata attributes
        self.redshift = self.reference_group_zoom.metadata.redshift
        self.cosmology = self.reference_group_zoom.metadata.cosmology
        
        self.critical_density_z = astropy_to_unyt(self.cosmology.critical_density(self.redshift)).to('Msun/Mpc**3')
        self.critical_density_z0 = astropy_to_unyt(self.cosmology.critical_density0).to('Msun/Mpc**3')
        self.mean_background_density_z = self.critical_density_z * self.cosmology.Om0
        self.mean_background_density_z0 = self.critical_density_z0 * self.cosmology.Om0
        self.e_function_z = self.cosmology.efunc(self.redshift)
        self.baryon_fraction = self.cosmology.Ob0 / self.cosmology.Om0

        # Generate useful rescaled coordinates
        self.reference_group_zoom.gas.spherical_coordinates.radius.convert_to_physical()
        self.gas_radius_scaled = self.reference_group_zoom.gas.spherical_coordinates.radius / self.r_500
        
        # Self-similar scaling
        kBT500 = unyt.G * mean_molecular_weight * self.mass_500 * unyt.mp / self.r_500 / 2
        self.kBT500_reference = kBT500.to('keV')
        
        ne500 = 500 * self.baryon_fraction * self.critical_density_z / (mean_atomic_weight_per_free_electron * unyt.mp)
        self.ne500_reference = ne500.to('1/cm**3')
        
        K500 = self.kBT500_reference / (self.ne500_reference ** (2 / 3))
        self.K500_reference = K500.to('keV*cm**2')
        
        # Redshift-zero reference
        z0_group_zoom = self.jump_snapshot(new_redshift=0.)

        if hasattr(z0_group_zoom.halo_finder.spherical_overdensities, 'r_500_rhocrit'):
            self.r_500_z0 = z0_group_zoom.halo_finder.spherical_overdensities.r_500_rhocrit.to('Mpc')
            self.mass_500_z0 = z0_group_zoom.halo_finder.spherical_overdensities.mass_500_rhocrit.to('Solar_Mass')
            
        else:
            warnings.warn("Spherical overdensities not found in catalogue. Computing from particle data.", RuntimeWarning)
            so = SphericalOverdensities.delta_500(z0_group_zoom)
            self.r_500_z0 = so.r_delta
            self.mass_500_z0 = so.m_delta
            
        kBT500 = unyt.G * mean_molecular_weight * self.mass_500_z0 * unyt.mp / self.r_500_z0 / 2
        self.kBT500_z0 = kBT500.to('keV')
        
        ne500 = 500 * self.baryon_fraction * self.critical_density_z0 / (mean_atomic_weight_per_free_electron * unyt.mp)
        self.ne500_z0 = ne500.to('1/cm**3')
        
        K500 = self.kBT500_z0 / (self.ne500_z0 ** (2 / 3))
        self.K500_z0 = K500.to('keV*cm**2')
        
            
    def jump_snapshot(self, new_redshift: float, **kwargs) -> GroupZoom:

        if new_redshift == self.redshift:
            warnings.warn("Same redshift as reference. Use the `<class>.reference_group_zoom` attribute.", RuntimeWarning)
            return self.reference_group_zoom

        return GroupZoom(run_directory=self.reference_group_zoom.run_directory, redshift=new_redshift, import_all_particles=True, **kwargs)
    
    def update_attributes(self, new_attributes: Dict[str, Union[float, unyt.unyt_quantity]]) -> None:
        
        for key, new_value in new_attributes.items():
            if hasattr(self, key):
                setattr(self, key, new_value)
                
    @check_mask_empty_nan
    def get_gas_entropies(self, group_zoom: GroupZoom, mask = ..., scaled: bool = False) -> unyt.unyt_array:
        
        density_selected = get_electron_number_density(group_zoom)[mask]
        density_selected.convert_to_physical()
        
        temperature_selected = group_zoom.gas.temperatures[mask] * unyt.kb
        temperature_selected.convert_to_physical()
        temperature_selected.convert_to_units('keV')
        
        entropies_selected = temperature_selected / (density_selected ** (2 / 3))
        entropies_selected.convert_to_physical()
        entropies_selected.convert_to_units('keV*cm**2')
        
        if scaled:
            return entropies_selected / self.K500_z0
        
        return entropies_selected
    
    @check_mask_empty_nan
    def get_gas_temperatures(self, group_zoom: GroupZoom, mask = ..., scaled: bool = False) -> unyt.unyt_quantity:

        temperature_selected = group_zoom.gas.temperatures[mask] * unyt.kb
        temperature_selected.convert_to_physical()
        temperature_selected.convert_to_units('keV')
        
        if scaled:
            return temperature_selected / self.kBT500_z0
        
        return temperature_selected
    
    @check_mask_empty_nan
    def get_gas_densities(self, group_zoom: GroupZoom, mask = ..., scaled: bool = False, dynamic: bool = False) -> unyt.unyt_quantity:

        density_selected = group_zoom.gas.densities[mask]
        density_selected.convert_to_physical()
        
        if scaled and dynamic:
            critical_density_z = astropy_to_unyt(group_zoom.metadata.cosmology.critical_density(group_zoom.metadata.redshift)).to('Msun/Mpc**3')
            return density_selected / critical_density_z
            
        elif scaled and not dynamic:
            return density_selected / self.critical_density_z0
        
        return density_selected    
    
    @check_mask_empty_nan
    def get_gas_radii(self, group_zoom: GroupZoom, mask = ..., scaled: bool = False, dynamic: bool = False) -> unyt.unyt_array:

        radii_selected = group_zoom.gas.spherical_coordinates.radius[mask]
        radii_selected.convert_to_physical()
        radii_selected.convert_to_units('Mpc')
        
        if scaled and dynamic:
            return radii_selected / group_zoom.halo_finder.spherical_overdensities.r_500_rhocrit.to('Mpc')
        
        elif scaled and not dynamic:
            return radii_selected / self.r_500_z0
        
        return radii_selected
    
    @check_mask_empty_nan
    def get_gas_coordinates(self, group_zoom: GroupZoom, mask = ..., scaled: bool = False) -> unyt.unyt_array:

        coordinates_selected = group_zoom.gas.coordinates[mask]
        coordinates_selected.convert_to_physical()
        coordinates_selected.convert_to_units('Mpc')
        
        if scaled:
            return coordinates_selected / self.r_500_z0
                
        return coordinates_selected
    
    @check_mask_empty_nan
    def get_gas_radial_velocities(self, group_zoom: GroupZoom, mask = ...) -> unyt.unyt_array:

        velocities_selected = group_zoom.gas.spherical_velocities.radius[mask]
        velocities_selected.convert_to_physical()
        velocities_selected.convert_to_units('km/s')
        
        return velocities_selected
    
    def compute_median_entropy(self, group_zoom: GroupZoom, mask = ..., **kwargs) -> unyt.unyt_quantity:
        
        entropies_selected = self.get_gas_entropies(group_zoom, mask=mask, **kwargs)
        median_entropy = unyt.unyt_quantity(np.percentile(entropies_selected, 50), entropies_selected.units)        
        return median_entropy
    
    def compute_median_temperature(self, group_zoom: GroupZoom, mask = ..., **kwargs) -> unyt.unyt_quantity:

        temperature_selected = self.get_gas_temperatures(group_zoom, mask=mask, **kwargs)
        median_temperature = unyt.unyt_quantity(np.percentile(temperature_selected, 50), temperature_selected.units)        
        return median_temperature
    
    def compute_median_density(self, group_zoom: GroupZoom, mask = ..., **kwargs) -> unyt.unyt_quantity:

        density_selected = self.get_gas_densities(group_zoom, mask=mask, **kwargs)
        median_density = unyt.unyt_quantity(np.percentile(density_selected, 50), density_selected.units)
        return median_density
    
    def compute_median_radius(self, group_zoom: GroupZoom, mask = ..., **kwargs) -> unyt.unyt_quantity:

        radius_selected = self.get_gas_radii(group_zoom, mask=mask, **kwargs)
        median_radius = unyt.unyt_quantity(np.percentile(radius_selected, 50), radius_selected.units)
        return median_radius
    
    def compute_median_radial_velocity(self, group_zoom: GroupZoom, mask = ...) -> unyt.unyt_quantity:

        velocity_selected = self.get_gas_radial_velocities(group_zoom, mask=mask)
        median_velocity = unyt.unyt_quantity(np.percentile(velocity_selected, 50), velocity_selected.units)
        return median_velocity
    
    @check_mask_empty_nan
    def plot_shell(self, group_zoom: GroupZoom, mask = ..., draw_sphere: bool = True) -> None:
        
        from matplotlib import pyplot as plt

        coordinates = group_zoom.gas.coordinates[mask]
        radial_distances = group_zoom.gas.spherical_coordinates.radius[mask]
        
        # Reduce the number of particles
        take_every_n = int(np.log10(len(coordinates))) + 1
        coordinates = coordinates[::take_every_n]

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        fig.set_facecolor("white")

        ax.scatter3D(*coordinates.T, s=5, ec="none", fc="k")

        if draw_sphere:

            # Draw centre of the sphere
            ax.scatter3D(0, 0, 0, s=100, color="r", marker="x")
            radius_sphere = np.max(radial_distances).value
            u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 100j]
            x = radius_sphere * np.cos(u) * np.sin(v)
            y = radius_sphere * np.sin(u) * np.sin(v)
            z = radius_sphere * np.cos(v)
            ax.plot_wireframe(x, y, z, color="lime", alpha=0.1, zorder=0)

        plt.show()

        fig = plt.figure()
        ax = fig.subplots()
        plt.hist(radial_distances.value, bins=10, histtype="step")
        plt.show()

class LagrangianTransport(ParticleTracker):
    
    def __init__(
            self,
            reference_group_zoom: GroupZoom,
            rmin: Optional[float] = 0.,
            rmax: float = 1.,
            tmin: unyt.unyt_quantity = 1.e5 * unyt.K, 
            tmax: unyt.unyt_quantity = 1.e15 * unyt.K
    ):
        
        super().__init__(reference_group_zoom=reference_group_zoom)
        self.rmin = numpy_to_cosmo_array(rmin, self.gas_radius_scaled)
        self.rmax = numpy_to_cosmo_array(rmax, self.gas_radius_scaled)
        self.tmin = numpy_to_cosmo_array(tmin, self.reference_group_zoom.gas.temperatures)
        self.tmax = numpy_to_cosmo_array(tmax, self.reference_group_zoom.gas.temperatures)
        
        # Mask all gas in shell            
        self.reference_particle_mask = np.where((self.gas_radius_scaled > rmin) & 
                                                (self.gas_radius_scaled < rmax) &
                                                (self.reference_group_zoom.gas.temperatures > tmin) & 
                                                (self.reference_group_zoom.gas.temperatures < tmax) &
                                                (self.reference_group_zoom.gas.split_counts.value == 0))[0]
        
        self.reference_particle_ids = self.reference_group_zoom.gas.particle_ids[self.reference_particle_mask]
        self.reference_n_particles = self.reference_particle_mask.size
        self.turned_into_stars = 0
        
    def compute_median_entropy_reference(self, **kwargs) -> unyt.unyt_quantity:
        return self.compute_median_entropy(self.reference_group_zoom, mask=self.reference_particle_mask, **kwargs)
    
    def compute_median_temperature_reference(self, **kwargs) -> unyt.unyt_quantity:
        return self.compute_median_temperature(self.reference_group_zoom, mask=self.reference_particle_mask, **kwargs)
    
    def compute_median_density_reference(self, **kwargs) -> unyt.unyt_quantity:
        return self.compute_median_density(self.reference_group_zoom, mask=self.reference_particle_mask, **kwargs)
    
    def get_gas_entropies_reference(self, **kwargs) -> unyt.unyt_array:
        return self.get_gas_entropies(self.reference_group_zoom, mask=self.reference_particle_mask, **kwargs)
    
    def get_gas_radii_reference(self, **kwargs) -> unyt.unyt_array:
        return self.get_gas_radii(self.reference_group_zoom, mask=self.reference_particle_mask, **kwargs)
    
    
    def compute_median_entropy_snapshot(self, group_zoom: GroupZoom, **kwargs) -> unyt.unyt_quantity:
        
        snapshot_a = self.reference_particle_ids
        snapshot_b = group_zoom.gas.particle_ids        
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        
        return self.compute_median_entropy(group_zoom, mask=particle_mask, **kwargs)    
    
    def compute_median_temperature_snapshot(self, group_zoom: GroupZoom, **kwargs) -> unyt.unyt_quantity:
        
        snapshot_a = self.reference_particle_ids
        snapshot_b = group_zoom.gas.particle_ids        
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        
        return self.compute_median_temperature(group_zoom, mask=particle_mask, **kwargs)
        
    def compute_median_density_snapshot(self, group_zoom: GroupZoom, **kwargs) -> unyt.unyt_quantity:
        
        snapshot_a = self.reference_particle_ids
        snapshot_b = group_zoom.gas.particle_ids        
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        
        return self.compute_median_density(group_zoom, mask=particle_mask, **kwargs)
    
    def compute_median_radius_snapshot(self, group_zoom: GroupZoom, **kwargs) -> unyt.unyt_quantity:
        
        snapshot_a = self.reference_particle_ids
        snapshot_b = group_zoom.gas.particle_ids        
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        
        # Formed stars
        if not group_zoom.is_nonradiative:
            print("Gas turned into stars:", np.where(np.isin(group_zoom.stars.particle_ids, snapshot_a, assume_unique=True))[0].size)      
            self.turned_into_stars = np.where(np.isin(group_zoom.stars.particle_ids, snapshot_a, assume_unique=True))[0].size  
        
        return self.compute_median_radius(group_zoom, mask=particle_mask, **kwargs)
    
    def compute_median_radial_velocity_snapshot(self, group_zoom: GroupZoom, **kwargs) -> unyt.unyt_quantity:
        
        snapshot_a = self.reference_particle_ids
        snapshot_b = group_zoom.gas.particle_ids        
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        
        return self.compute_median_radial_velocity(group_zoom, mask=particle_mask, **kwargs)
    
    def get_gas_entropies_snapshot(self, group_zoom: GroupZoom, **kwargs) -> unyt.unyt_array:
        
        snapshot_a = self.reference_particle_ids
        snapshot_b = group_zoom.gas.particle_ids        
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        
        return self.get_gas_entropies(group_zoom, mask=particle_mask, **kwargs)
    
    def get_gas_temperatures_snapshot(self, group_zoom: GroupZoom, **kwargs) -> unyt.unyt_array:
        
        snapshot_a = self.reference_particle_ids
        snapshot_b = group_zoom.gas.particle_ids        
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        
        return self.get_gas_temperatures(group_zoom, mask=particle_mask, **kwargs)
    
    def get_gas_radii_snapshot(self, group_zoom: GroupZoom, **kwargs) -> unyt.unyt_array:
        
        snapshot_a = self.reference_particle_ids
        snapshot_b = group_zoom.gas.particle_ids        
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        
        return self.get_gas_radii(group_zoom, mask=particle_mask, **kwargs)
    
    def get_gas_coordinates_snapshot(self, group_zoom: GroupZoom, **kwargs) -> unyt.unyt_array:
        
        snapshot_a = self.reference_particle_ids
        snapshot_b = group_zoom.gas.particle_ids        
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        
        return self.get_gas_coordinates(group_zoom, mask=particle_mask, **kwargs)
    
    
class EjectedGas(ParticleTracker):
    
    def __init__(
            self,
            group_zoom_ref_model: GroupZoom,
            group_zoom_nr_model: GroupZoom,
            r_boundary: float = 1.,
            rmin: Optional[float] = 1.,
            rmax: Optional[float] = 10.,
            tmin: unyt.unyt_quantity = 1.e5 * unyt.K, 
            tmax: unyt.unyt_quantity = 1.e15 * unyt.K
    ):
        
        super().__init__(reference_group_zoom=group_zoom_ref_model)
        self.r_boundary = numpy_to_cosmo_array(r_boundary, self.gas_radius_scaled)
        self.rmin = numpy_to_cosmo_array(rmin, self.gas_radius_scaled)
        self.rmax = numpy_to_cosmo_array(rmax, self.gas_radius_scaled)
        self.tmin = numpy_to_cosmo_array(tmin, self.reference_group_zoom.gas.temperatures)
        self.tmax = numpy_to_cosmo_array(tmax, self.reference_group_zoom.gas.temperatures)
        
        # Create start and target objects for later reference
        self.group_zoom_ref_model = self.reference_group_zoom
        self.group_zoom_nr_model_history = ParticleTracker(group_zoom_nr_model)
        self.group_zoom_nr_model = self.group_zoom_nr_model_history.reference_group_zoom
        
        # Mask all gas hot and ejected in the Ref model
        self.particle_mask_ref_model = np.where((self.gas_radius_scaled.value >= self.rmin.value) &
                                                (self.gas_radius_scaled.value <= self.rmax.value) &
                                                (self.group_zoom_ref_model.gas.temperatures.value > self.tmin.value) & 
                                                (self.group_zoom_ref_model.gas.temperatures.value < self.tmax.value))[0]
        
        # Mask all gas at any temperature and enclosed in the NR model
        self.particle_mask_nr_model = np.where(self.group_zoom_nr_model_history.gas_radius_scaled.value <= self.r_boundary.value)[0]
        
        # Combine the IDs of the particles that satisfy both conditions (i.e. SQL UNION key_tables)
        _particle_ids_ref_model = self.group_zoom_ref_model.gas.particle_ids[self.particle_mask_ref_model]
        _particle_ids_nr_model = self.group_zoom_nr_model.gas.particle_ids[self.particle_mask_nr_model]
        _particle_mask_ejected = np.isin(_particle_ids_ref_model, _particle_ids_nr_model, assume_unique=True)
        
        # Save ID of ejected particles
        self.ejected_n_particles = _particle_mask_ejected.size
        self.ejected_particle_ids = _particle_ids_ref_model[_particle_mask_ejected]
    
    def compute_median_entropy_snapshot(self, group_zoom: GroupZoom, **kwargs) -> unyt.unyt_quantity:
        
        snapshot_a = self.ejected_particle_ids
        snapshot_b = group_zoom.gas.particle_ids        
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        
        return self.compute_median_entropy(group_zoom, mask=particle_mask, **kwargs)    
    
    def compute_median_temperature_snapshot(self, group_zoom: GroupZoom, **kwargs) -> unyt.unyt_quantity:
        
        snapshot_a = self.ejected_particle_ids
        snapshot_b = group_zoom.gas.particle_ids        
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        
        return self.compute_median_temperature(group_zoom, mask=particle_mask, **kwargs)
    
    def compute_median_density_snapshot(self, group_zoom: GroupZoom, **kwargs) -> unyt.unyt_quantity:
        
        snapshot_a = self.ejected_particle_ids
        snapshot_b = group_zoom.gas.particle_ids        
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        
        return self.compute_median_density(group_zoom, mask=particle_mask, **kwargs)
    
    def compute_median_radius_snapshot(self, group_zoom: GroupZoom, **kwargs) -> unyt.unyt_quantity:
        
        snapshot_a = self.ejected_particle_ids
        snapshot_b = group_zoom.gas.particle_ids        
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        
        return self.compute_median_radius(group_zoom, mask=particle_mask, **kwargs)
    
    def compute_median_radial_velocity_snapshot(self, group_zoom: GroupZoom, **kwargs) -> unyt.unyt_quantity:
        
        snapshot_a = self.ejected_particle_ids
        snapshot_b = group_zoom.gas.particle_ids        
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        
        return self.compute_median_radial_velocity(group_zoom, mask=particle_mask, **kwargs)
    
    def get_gas_entropies_snapshot(self, group_zoom: GroupZoom, **kwargs) -> unyt.unyt_array:
        
        snapshot_a = self.ejected_particle_ids
        snapshot_b = group_zoom.gas.particle_ids
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        sort_key = np.argsort(snapshot_b[particle_mask])
        
        selected_entropies = self.get_gas_entropies(group_zoom, mask=particle_mask, **kwargs)
        sorted_entropies = selected_entropies[sort_key]
        
        return sorted_entropies
    
    def get_gas_radii_snapshot(self, group_zoom: GroupZoom, **kwargs) -> unyt.unyt_array:
        
        snapshot_a = self.ejected_particle_ids
        snapshot_b = group_zoom.gas.particle_ids        
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        sort_key = np.argsort(snapshot_b[particle_mask])
        
        selected_radii = self.get_gas_radii(group_zoom, mask=particle_mask, **kwargs)
        sorted_radii = selected_radii[sort_key]
        
        return sorted_radii
    
    def _get_gas_ejected_sorted_ids_snapshot(self, group_zoom: GroupZoom) -> np.ndarray:
        
        snapshot_a = self.ejected_particle_ids
        snapshot_b = group_zoom.gas.particle_ids        
        particle_mask = np.where(np.isin(snapshot_b, snapshot_a, assume_unique=True))[0]
        sort_key = np.argsort(snapshot_b[particle_mask])
            
        return snapshot_b[particle_mask][sort_key]
    
    def get_gas_entropies_models_match(self, group_zoom_1: GroupZoom, group_zoom_2: GroupZoom) -> unyt.unyt_array:
                
        entropies_1 = self.get_gas_entropies_snapshot(group_zoom_1, scaled=False)
        entropies_2 = self.get_gas_entropies_snapshot(group_zoom_2, scaled=False)
        
        try:            
            return entropies_1 / entropies_2
            
        except objects.InvalidScaleFactor:
            # Some snapshots do not have exactly matching redshifts. Ignore this error if redshifts are approximately equal.
            return unyt.unyt_array(entropies_1.value / entropies_2.value, unyt.dimensionless)
            
    def get_gas_radii_models_match(self, group_zoom_1: GroupZoom, group_zoom_2: GroupZoom) -> unyt.unyt_array:
        
        radii_1 = self.get_gas_radii_snapshot(group_zoom_1, scaled=False)
        radii_2 = self.get_gas_radii_snapshot(group_zoom_2, scaled=False)
        
        try:            
            return radii_1 / radii_2
            
        except objects.InvalidScaleFactor:
            # Some snapshots do not have exactly matching redshifts. Ignore this error if redshifts are approximately equal.
            return unyt.unyt_array(radii_1.value / radii_2.value, unyt.dimensionless)
            
    def _get_gas_ejected_sorted_ids_models_match(self, group_zoom_1: GroupZoom, group_zoom_2: GroupZoom) -> Tuple[np.ndarray]:
        
        ids_1 = self._get_gas_ejected_sorted_ids_snapshot(group_zoom_1)
        ids_2 = self._get_gas_ejected_sorted_ids_snapshot(group_zoom_2)
        
        return ids_1, ids_2
    
