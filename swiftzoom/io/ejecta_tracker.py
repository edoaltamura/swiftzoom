import os
from collections import ChainMap
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import unyt

from astropy.cosmology import Planck18_arXiv_v2 as planck18

import boilerplate
from swiftzoom.loader import GroupZoom, Dict2HDF
from swiftzoom.properties.lagrangian_transport import EjectedGas

radial_masks = [dict(rmin=0., rmax=1.), 
                dict(rmin=1., rmax=2.), 
                dict(rmin=2., rmax=3.), 
                dict(rmin=3., rmax=4.), 
                dict(rmin=4., rmax=5.), 
                dict(rmin=5., rmax=10.),
                dict(rmin=10., rmax=20.)]

def make_redshift_list(z_start: float = 0., z_end: float = 14., num_redshifts: int = 10) -> np.array:

    a = np.logspace(np.log10(1 / (z_end + 1)), z_start, num=num_redshifts)
    redshifts = 1 / a[::-1] - 1
    return redshifts

def sweep_redshifts(object_a: str, redshift_max: float = 14., num_redshifts: int = 10, 
                    save_hdf: bool = True, parallelize: bool = True) -> dict:
    
    # Initialise arrays for radial intervals
    redshifts_tracked = []    
    entropy_tracked = {f"{rm['rmin']:.0f}_{rm['rmax']:.0f}": [] for rm in radial_masks}
    temperature_tracked = {f"{rm['rmin']:.0f}_{rm['rmax']:.0f}": [] for rm in radial_masks}
    density_tracked = {f"{rm['rmin']:.0f}_{rm['rmax']:.0f}": [] for rm in radial_masks}
    
    # Ref requires all particles imported, while NR only gas inside r500
    print('Loading Ref data')
    group_ref = GroupZoom(f'../data/01_raw/{object_a:s}_Ref/', redshift=0., import_all_particles=True)
    print('Loading Nonradiative data')
    group_nr = GroupZoom(f'../data/01_raw/{object_a:s}_Nonradiative/', redshift=0., import_all_particles=True)
    
    ejected_objs = {}
    for rm in tqdm(radial_masks, desc='Ejection list'):
        ejected_objs[f"{rm['rmin']:.0f}_{rm['rmax']:.0f}"] = EjectedGas(group_ref, group_nr, **rm)

    def process_redshift(new_redshift, ejection_list, pbar) -> tuple:
        
        _entropy = []
        _temperature = []
        _density = []

        new_group = GroupZoom(run_directory=f'../data/01_raw/{object_a:s}_Ref/', redshift=new_redshift, import_all_particles=True)        
        
        for rm_key, ejected_obj in ejection_list.items():
            
            _entropy.append(ejected_obj.compute_median_entropy_snapshot(group_zoom=new_group, scaled=True))
            _temperature.append(ejected_obj.compute_median_temperature_snapshot(group_zoom=new_group, scaled=True))
            _density.append(ejected_obj.compute_median_density_snapshot(group_zoom=new_group, scaled=True, dynamic=True))

        redshift = new_group.metadata.redshift

        # Update the progress bar
        pbar.update(1)

        return (*_entropy, *_temperature, *_density, redshift)

    # Create the progress bar
    with tqdm(total=num_redshifts, desc=f"{object_a:s}") as pbar:
        
        redshifts = make_redshift_list(z_end=redshift_max, num_redshifts=num_redshifts)
        
        # Parallelize the loop over redshifts
        if parallelize:
            # Parallelize the loop over redshifts     
            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(process_redshift)(new_redshift, ejected_objs, pbar) for new_redshift in redshifts
            )
        else:
            results = [process_redshift(new_redshift, ejected_objs, pbar) for new_redshift in redshifts]
            
    # Unpack the results
    print('Unpacking results')
    for result in results:
        
        redshifts_tracked.append(result[-1])
        
        for idx, rm in enumerate(radial_masks):
            
            rm_key = f"{rm['rmin']:.0f}_{rm['rmax']:.0f}"            
            entropy_tracked[rm_key].append(result[idx])
            temperature_tracked[rm_key].append(result[idx + len(radial_masks)])
            density_tracked[rm_key].append(result[idx + 2 * len(radial_masks)])

    # Collect results into one dictionary
    quantities = [entropy_tracked, temperature_tracked, density_tracked]
    quantities_names = ["entropy", "temperature", "density"]
    
    data_output = [{f"{n}_{k}": unyt.unyt_array(v) for k, v in q.items()} for n, q in zip(quantities_names, quantities)]
    data_output = dict(ChainMap(*data_output))
    
    # Append redshift last    
    data_output.update({"redshifts": np.array(redshifts_tracked)})
    
    if save_hdf:
        filename = f"ejected_history_{object_a:s}_{num_redshifts:d}.hdf5"
        hdf_obj = Dict2HDF(filename=filename)
        hdf_obj.save_dict_to_hdf5(data_output)
        file_size = os.path.getsize(hdf_obj.filename) / (1024 * 1024.0)
        print(f"Saved data to file: [{file_size:.2f} MB] {filename}")
    
    return data_output


if __name__ == '__main__':
    
    import warnings
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        sweep_redshifts('VR18_-8res', num_redshifts=30, parallelize=True)
