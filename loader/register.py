import re
from typing import Optional, Dict
from unyt import unyt_quantity, K, mp, kb
import numpy as np
from swiftgalaxy import SWIFTGalaxy, Velociraptor, MaskCollection

from .output_list import OutputList

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
    ) -> None:
        
        out_list = OutputList(run_directory)
        
        if snapshot_number is not None:
            snap, cat = out_list.files_from_snap_number(snapshot_number)
            
        elif redshift is not None:
            snap, cat = out_list.files_from_redshift(redshift)
            
        else:
            raise ValueError("Redshift or snapshot_number must be defined.")
            
        cat = re.sub('\.properties$', '', cat)
        vr_object = Velociraptor(cat, halo_index=halo_index, centre_type='minpot', extra_mask=None)
                
        super().__init__(snap, vr_object, auto_recentre=auto_recentre, _spatial_mask=None)
        super().wrap_box()
        
        self.is_nonradiative = self.metadata.n_stars == 0
        # If temperature arrays are not included (e.g.) adiabatic mode
        if not hasattr(self.gas, 'temperatures') and self.is_nonradiative:            
            self.set_temperatures_from_internal_energies()
        
    def set_temperatures_from_internal_energies(self):
        
        mean_molecular_weight = 0.5954  # mean atomic weight for an ionized gas with primordial (X = 0.76, Z = 0) composition  
        
        print('Genering temperatures from internal energies.')
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