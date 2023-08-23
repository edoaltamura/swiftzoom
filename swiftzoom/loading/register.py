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
    
    def __init__(
        self, 
        run_directory: str, 
        redshift: Optional[float] = None, 
        snapshot_number: Optional[int] = None,
        halo_index: Optional[int] = 0,
        auto_recentre: Optional[bool] = True,
        import_all_particles: Optional[bool] = False,
    ) -> None:
        
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
            load_region = [[0. * b, b] for b in mask.metadata.boxsize]      # load_region is a 3x2 list [[left, right], [bottom, top], [front, back]]
            mask.constrain_spatial(load_region)                             # Constrain the mask
            spatial_mask_kwargs = dict(_spatial_mask=mask)
        else:
            spatial_mask_kwargs = dict()

        super().__init__(snap, vr_object, auto_recentre=auto_recentre, _extra_mask=None, **spatial_mask_kwargs)
        super().wrap_box()
        
        self.is_nonradiative = self.metadata.n_stars == 0                   # If temperature arrays are not included (e.g.) adiabatic mode
        
        if not hasattr(self.gas, 'temperatures') and self.is_nonradiative:
            warnings.warn('Genering temperatures from internal energies.', RuntimeWarning)       
            self.set_temperatures_from_internal_energies()
        
    def set_temperatures_from_internal_energies(self):
              
        self.gas.internal_energies.convert_to_physical()
        setattr(
                self.gas, 'temperatures', (
                self.gas.internal_energies * 
                (self.metadata.gas_gamma - 1) * 
                mean_molecular_weight * mp / kb
            )
        )
        
    def get_mask_gas_temperature(self, tmin: unyt_quantity = 1.e5 * K, tmax: unyt_quantity = 1.e15 * K) -> np.ndarray:
        
        return np.where((self.gas.temperatures > tmin) & (self.gas.temperatures < tmax))[0]
    
        
    def get_mask_3d_radius_r500(self, rmin: float = 0., rmax: float = 5.) -> Dict[str, np.ndarray]:
        
        radial_mask = {}
        
        for particle_type in ['gas', 'dark_matter', 'stars', 'black_holes']:
            
            getattr(self.group_zoom, particle_type).spherical_coordinates.radius.convert_to_physical()
            
            radius_scaled = (
                getattr(self.group_zoom, particle_type).spherical_coordinates.radius / self.group_zoom.halo_finder.spherical_overdensities.r_500_rhocrit
            )
        
            radial_mask[particle_type] = (radius_scaled > rmin) & (radius_scaled < rmin)
            
        return radial_mask
        
        
        
if __name__ == '__main__':
    
    run_directory = '/cosma/home/dp004/dc-alta2/shocks_paper_data/VR2915_-8res_Ref/'
    
    zoom = GroupZoom(run_directory, redshift=0., halo_index=0)
    print(zoom.gas.__dir__())
    
    print((zoom.gas.temperatures.value > 1e5))
    # print(zoom.halo_finder.masses.mvir)
