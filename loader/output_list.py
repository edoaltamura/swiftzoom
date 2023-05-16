import os
import numpy as np
import pandas as pd

class OutputList(object):
    
    def __init__(self, run_directory: str):
        
        self.run_directory = run_directory
        
        # Check that there are as many outputs as in the output_list
        try:
            output_list_file = os.path.join(self.run_directory, 'snap_redshifts.txt')
            assert os.path.isfile(output_list_file)
        except AssertionError:
            output_list_file = os.path.join(self.run_directory, 'output_list.txt')
            assert os.path.isfile(output_list_file), f"Cannot find file: {output_list_file:s}"
        
        output_list = pd.read_csv(output_list_file)                

        if " Select Output" in output_list.columns:
            self.number_snapshots_in_outputlist = np.logical_or.reduce(
                    [output_list[" Select Output"] == " Snapshot"]
                ).sum()
        else:
            self.number_snapshots_in_outputlist = len(output_list["# Redshift"].values)
            
            # If it doesn't contain the name list, assume they're all snapshots
            output_list[" Select Output"] = [" Snapshot"] * self.number_snapshots_in_outputlist
            
        self.output_name = np.empty(self.number_snapshots_in_outputlist, dtype=str)
        self.output_redshifts = np.empty(self.number_snapshots_in_outputlist, dtype=float)
        
        for i, (redshift, name) in enumerate(zip(
            output_list["# Redshift"].values, output_list[" Select Output"]
        )):
            self.output_name[i] = name.strip()
            self.output_redshifts[i] = redshift
            
    def match_redshift(self, redshift_query: float):
        
        array = self.output_redshifts
        idx = (np.abs(array - redshift_query)).argmin()
        return array[idx], idx
    
    def files_from_snap_number(self, snapshot_number: int, extra_returns: bool = False):
        snapshot_path = ''
        for file in os.listdir(os.path.join(self.run_directory, 'snapshots')):
            if file.endswith(f"_{snapshot_number:04d}.hdf5"):
                snapshot_path = os.path.join(self.run_directory, 'snapshots', file)
                break
        assert snapshot_path, f"Could not find snapshot file."
        
        catalogue_path = ''
        for subdir in os.listdir(os.path.join(self.run_directory, 'stf')):
            if subdir.endswith(f"_{snapshot_number:04d}"):
                for file in os.listdir(os.path.join(self.run_directory, 'stf', subdir)):
                    if file.endswith(f"_{snapshot_number:04d}.properties"):
                        catalogue_path = os.path.join(self.run_directory, 'stf', subdir, file)
                        break
        assert catalogue_path, f"Could not find catalogue file."
        
        if extra_returns:
            redshift = self.output_redshifts[snapshot_number]
            return snapshot_path, catalogue_path, snapshot_number, redshift
        else:
            return snapshot_path, catalogue_path

    
    def files_from_redshift(self, redshift_query: float, extra_returns: bool = False):
        
        nearest_redshift, snap_number = self.match_redshift(redshift_query)
        snapshot_path, catalogue_path = self.files_from_snap_number(snap_number)
        
        if extra_returns:
            return snapshot_path, catalogue_path, snap_number, nearest_redshift
        else:
            return snapshot_path, catalogue_path
        
        
if __name__ == '__main__':
    out_obj = OutputList('/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR2915_+1res_Bipolar_fixedAGNdT8.5_sep2021')
    
    print("\n--- Testing `files_from_snap_number`".upper())
    path_to_snap, path_to_catalogue, snap_number, nearest_redshift = out_obj.files_from_snap_number(150, extra_returns=True)
    print(f"path_to_snap: {path_to_snap}") 
    print(f"path_to_catalogue: {path_to_catalogue}") 
    print(f"snap_number: {snap_number}") 
    print(f"nearest_redshift: {nearest_redshift}")
    
    print("\n--- Testing `files_from_redshift`".upper())
    path_to_snap, path_to_catalogue, snap_number, nearest_redshift = out_obj.files_from_redshift(4, extra_returns=True)
    print(f"path_to_snap: {path_to_snap}") 
    print(f"path_to_catalogue: {path_to_catalogue}") 
    print(f"snap_number: {snap_number}") 
    print(f"nearest_redshift: {nearest_redshift}")
        