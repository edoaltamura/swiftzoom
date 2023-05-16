import os
os.environ['mpi_warn_on_fork'] = '0'

import asyncio
import joblib
import h5py
from itertools import product
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from warnings import warn
import unyt
from astropy import units
from scipy.interpolate import interp1d
import velociraptor
from velociraptor.catalogue.catalogue import VelociraptorCatalogue

import sys
sys.path.append("..")
sys.path.append("../..")

from output_list import OutputList


class CustomPickler:
    
    def __init__(self, filename: str, relative_path: bool = False):
        """
        Initialize the custom pickler object.
        
        Parameters:
            filename (str): the name of the file to save or load data from
            relative_path (bool): indicate whether the path of the file should be relative or absolute
        """
        if relative_path:
            self.filename = os.path.join(
                default_output_directory, "intermediate", filename
            )
        else:
            self.filename = filename


    def large_file_warning(self) -> None:
        """
        Show a warning message if the file size is greater than 500MB
        
        Returns:
            None
        """
        file_size_b = os.path.getsize(self.filename)
        if not xlargs.quiet and file_size_b > 524288000:
            warn(
                (
                    "[io] Detected file larger than 500 MB! "
                    "Trying to import all contents of pkl to memory at once. "
                    "If the file is large, you may run out of memory or degrade the "
                    "performance. You can use the `MultiObjPickler.get_pickle_generator` "
                    "to access a generator, which returns only one pickled object at a "
                    "time."
                ),
                category=ResourceWarning,
            )


class Dict2HDF(CustomPickler):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_dict_to_hdf5(self, dictionary: dict, mode: str = "w"):
        """
        Saves a nested python dictionary to an hdf5 file.
        
        Parameters:
            dictionary (dict): the nested python dictionary to be saved.
            mode (str): the mode in which to open the hdf5 file.
                
        Returns:
            None
        """
        with h5py.File(self.filename, mode) as h5file:
            self._recursively_save_dict_contents_to_group(h5file, "/", dictionary)

    def _recursively_save_dict_contents_to_group(self, h5file, path, dic):
        """
        Recursively saves the contents of a nested python dictionary to an hdf5 group.
        
        Parameters:
            h5file (h5py.File): the hdf5 file object.
            path (str): the path to the group in the hdf5 file.
            dic (dict): the nested python dictionary to be saved.
            
        Returns:
            None
        """
        for key, item in dic.items():
            if isinstance(item, (np.ndarray, int, float, str, bytes)):
                try:
                    h5file[path + key] = item
                except:
                    print('Could not save', path + key, type(item))
                
            elif isinstance(item, dict):
                self._recursively_save_dict_contents_to_group(
                    h5file, path + key + "/", item
                )
            else:
                raise ValueError("Cannot save %s type" % type(item))

    def load_dict_from_hdf5(self):
        """
        Loads the contents of an hdf5 file into a nested python dictionary.
        
        Returns:
            ans (dict): a nested python dictionary containing the contents of the hdf5 file.
        """
        with h5py.File(self.filename, "r") as h5file:
            return self._recursively_load_dict_contents_from_group(h5file, "/")

    def _recursively_load_dict_contents_from_group(self, h5file, path):
        """
        Recursively loads the contents of an hdf5 group into a nested python dictionary.
        
        Parameters:
            h5file (h5py.File): the hdf5 file object that contains the group.
            path (str): the path to the group in the hdf5 file.
            
        Returns:
            ans (dict): a nested python dictionary containing the contents of the hdf5 group.
        """
        ans = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item[...]
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = self._recursively_load_dict_contents_from_group(
                    h5file, path + key + "/"
                )
        return ans


@dataclass
class Quantity:
    name: str
    shape: tuple
    unit: unyt.unyt_quantity
    dtype: type
    

class TimeEvolutionCompute(object):
    host_dir = '/cosma/home/dp004/dc-alta2/shocks_paper_data'
    output_directory = f'{host_dir:s}/analysis'
    
    halos = [
        # 'VR2915_-8res', 
        # 'VR2915_+1res', 
        # 'VR2915_+8res', 
        # 'VR18_-8res',
        'VR18_+1res',
    ]
    models = ['Ref', 'Nonradiative']
    
    vr_quantities = [
            Quantity('Mass_200crit', (), 1E10 * unyt.Solar_Mass, float),
            Quantity('Mass_200mean', (), 1E10 * unyt.Solar_Mass, float),
            Quantity('Mvir', (), 1E10 * unyt.Solar_Mass, float),
            Quantity('Num_of_groups', (), unyt.dimensionless, int),
            Quantity('R_200crit', (), unyt.Mpc, float),
            Quantity('R_200mean', (), unyt.Mpc, float),
            Quantity('Rvir', (), unyt.Mpc, float),
            Quantity('SO_Mass_500_rhocrit', (), 1E10 * unyt.Solar_Mass, float),
            Quantity('SO_Mass_gas_500_rhocrit', (), 1E10 * unyt.Solar_Mass, float),
            Quantity('SO_Mass_gas_highT_0.100000_times_500.000000_rhocrit', (), 1E10 * unyt.Solar_Mass, float),
            Quantity('SO_Mass_gas_highT_1.000000_times_500.000000_rhocrit', (), 1E10 * unyt.Solar_Mass, float),
            Quantity('SO_T_gas_highT_0.100000_times_500.000000_rhocrit', (), unyt.K, float),
            Quantity('SO_T_gas_highT_1.000000_times_500.000000_rhocrit', (), unyt.K, float),
            Quantity('SO_Zmet_gas_highT_0.100000_times_500.000000_rhocrit', (), unyt.dimensionless, float),
            Quantity('SO_Zmet_gas_highT_1.000000_times_500.000000_rhocrit', (), unyt.dimensionless, float),
            Quantity('SO_R_500_rhocrit', (), unyt.Mpc, float),
            Quantity('Structuretype', (), unyt.dimensionless, int),
            Quantity('VXcminpot', (), unyt.km / unyt.s, float),
            Quantity('VYcminpot', (), unyt.km / unyt.s, float),
            Quantity('VZcminpot', (), unyt.km / unyt.s, float),
        ]
    
    derived_quantities = [        
        Quantity('critical_density', (), 1E10 * unyt.Solar_Mass / unyt.Mpc ** 3, float),
        Quantity('T500', (), unyt.K, float),
        Quantity('P500', (), unyt.keV * unyt.cm ** 3, float),
        Quantity('K500', (), unyt.keV * unyt.cm ** 2, float),
        Quantity('BCG_mass_0p1r500', (), 1E10 * unyt.Solar_Mass, float),
        Quantity('gas_mass_agn_heated_1Gyr', (), 1E10 * unyt.Solar_Mass, float),
        Quantity('gas_mass_sn_heated_1Gyr', (), 1E10 * unyt.Solar_Mass, float),
        Quantity('gas_mass_shocked', (), 1E10 * unyt.Solar_Mass, float),
        Quantity('sSFR_0p1r500_10Myr', (), 1 / unyt.Gyr, float),
        Quantity('sSFR_1p0r500_10Myr', (), 1 / unyt.Gyr, float),
        Quantity('entropy_0p1r500', (), unyt.keV * unyt.cm ** 2, float),
        Quantity('entropy_1p0r500', (), unyt.keV * unyt.cm ** 2, float),
        
        Quantity('density_profile', (30,), unyt.dimensionless, float),
        Quantity('pressure_profile', (30,), unyt.dimensionless, float),
        Quantity('temperature_profile', (30,), unyt.dimensionless, float),
        Quantity('entropy_profile', (30,), unyt.dimensionless, float),
        Quantity('hot_gas_fraction_profile', (30,), unyt.dimensionless, float),
        Quantity('cold_gas_fraction_profile', (30,), unyt.dimensionless, float),
    ]
    

    def __init__(self, model: str = 'Ref') -> None:

        self.model = model
        
        self.data = {}
        
        for halo in self.halos:
            self.setup_data_dictionary(halo)
            self.setup_vr_quantities(halo)
            self.allocate_vr_quantities(halo)
        
        # # Save analysis outputs and format files
        filename = f'redshift_evolution_{self.model}.hdf5'
        filepath = os.path.join(self.output_directory, filename)

        if os.path.isfile(filepath):
            warn(f"Saving data to hdf5 file. This will operation will overwrite {filepath:s}.", UserWarning)
            os.remove(filepath)

        # Save the data structure to HDF5 file
        Dict2HDF(filepath).save_dict_to_hdf5(self.data)

        self.print_dict_structure(self.data, verbose=False)
        
    async def _setup_data_dictionary_worker(self, index: int, halo: str) -> None:
        
        redshift = self.data[halo]['output_list'][index]
                
        try:
            output_list_obj = OutputList(self.data[halo]['run_directory'])
            snap, cat = output_list_obj.files_from_redshift(float(redshift))
            
        except Exception as e:
            print(halo, f"z={redshift}", e)
            return
        
        # If redshift is already processed, don't double count and leave fields as None
        redshift_int_exp10 = int(redshift * 1E10)
        if np.isin(redshift_int_exp10, self.data[halo]['redshifts_int_exp10']):
            return
        
        self.data[halo]['redshifts_int_exp10'][index] = redshift_int_exp10
        self.data[halo]['snapshot_files'][index] = snap
        self.data[halo]['catalogue_files'][index] = cat
        
        
    def setup_data_dictionary(self, halo: str) -> None:

        run_directory = f'{self.host_dir:s}/{halo:s}_{self.model:s}'
        
        try:
            redshifts = np.loadtxt(f'{run_directory:s}/output_list.txt')
        except OSError:
            redshifts = np.loadtxt(f'{run_directory:s}/snap_redshifts.txt')
        
        self.data[halo] = {}
        self.data[halo]['run_directory'] = run_directory
        self.data[halo]['output_list'] = redshifts
        self.data[halo]['output_list_size'] = len(redshifts)
        self.data[halo]['redshifts_int_exp10'] = np.empty_like(redshifts, dtype=int)
        self.data[halo]['snapshot_files'] = np.empty_like(redshifts, dtype='U512')
        self.data[halo]['catalogue_files'] = np.empty_like(redshifts, dtype='U512')
                                
        loop = asyncio.get_event_loop()       
        tasks = []
        for i in tqdm(range(len(redshifts)), desc=f"Collecting files for {halo:s}"):
            tasks.append(asyncio.ensure_future(self._setup_data_dictionary_worker(i, halo)))
            
        loop.run_until_complete(asyncio.wait(tasks))


    async def _setup_vr_quantities_worker(self, vr_quantity: Quantity, halo: str) -> None:
        """
        Asynchronous function that sets up a specific quantity for a halo from the Velociraptor
        catalogue.
        
        Parameters:
            vr_quantity (str): a string indicating which Velociraptor quantity to set up.
            halo (str): a string indicating which halo to set up the quantity for.

        Returns:
            None
        """
        
        self.data[halo][vr_quantity.name] = np.zeros(
            (self.data[halo]['output_list_size'], 2), 
            dtype=vr_quantity.dtype
        )
                
    def setup_vr_quantities(self, halo: str) -> None:
                    
        loop = asyncio.get_event_loop()  
        tasks = []            
        for vr_quantity in self.vr_quantities:
            tasks.append(
                asyncio.ensure_future(
                    self._setup_vr_quantities_worker(vr_quantity, halo)
                )
            )
        
        loop.run_until_complete(asyncio.wait(tasks))
                

    async def _allocate_vr_single_quantity_worker(self, index: int, vr_quantity: Quantity, redshift_sequence: np.ndarray, vr_data: VelociraptorCatalogue) -> None:
        """
        Asynchronous function that allocates the specific Velociraptor quantities for a halo from a hdf5 file.
        
        Parameters:
            index (int): index of the file to be read. The index indicates the redshift.
            halo (str): the halo to which the read data will be assigned to.
            
        Returns:
            None
        """
        redshift_sequence[index] = vr_data[vr_quantity.name][:2]        
        
    
    def _allocate_vr_quantities_worker(self, index: int, halo: str) -> None:
        """
        Main worker function that performs the main loop and calls the secondary loop worker.

        Parameters:
            index (int): the index of the data that needs to be processed

        """
        if self.data[halo]['catalogue_files'][index] is None:
            return
        
        with h5py.File(self.data[halo]['catalogue_files'][index], 'r') as vr_data:
            
            loop = asyncio.get_event_loop()  
            tasks = []
            redshift_sequence = np.zeros(len(self.vr_quantities))
            
            for vr_quantity in self.vr_quantities:
                tasks.append(
                    asyncio.ensure_future(
                        self._allocate_vr_single_quantity_worker(index, vr_quantity, redshift_sequence, vr_data)
                    )
                )

            loop.run_until_complete(asyncio.wait(tasks))
            
            self.data[halo][vr_quantity.name] = redshift_sequence            
            

    def allocate_vr_quantities(self, halo: str) -> None:
        """
        Run the parallelized loop using joblib.
        """
        redshift_iterator = range(self.data[halo]['output_list_size'])
        tqdm_desc = f"Alloc VR info for {halo:s}"
        
        with joblib.Parallel(n_jobs=joblib.cpu_count()) as parallel:
            parallel(
                joblib.delayed(self._allocate_vr_quantities_worker)(i, halo) for i in tqdm(redshift_iterator, desc=tqdm_desc)
            )

    def print_dict_structure(self, d, indent: int = 0, verbose: bool = False):
        """
        Print the structure of a nested dictionary, including the size of each array, if the value is an array.
        The function works by recursively calling itself when it finds a nested dictionary.
        
        Parameters:
            d (dict): The input nested dictionary to print the structure of.
            indent (int, optional): The indentation level for the output. Default value is 0.
        
        Returns:
            None
        """    
        for key, value in d.items():
            print(" "*indent + str(key))
            if isinstance(value, dict):
                self.print_dict_structure(value, indent+2, verbose=verbose)
            elif isinstance(value, (list, tuple, set, np.ndarray)):                    
                print(" "*(indent+2) + f"{'Size: '}{len(value)}")
                if verbose:
                    print(" "*(indent+2) + f"{'Value: '}{value}")
            else:
                print(" "*(indent+2) + str(value))    
                        
                        
class TimeEvolutionRead(object):
    
    filepath = os.path.join('/cosma/home/dp004/dc-alta2/data7/xl-zooms/analysis', 'redshift_evolution_Ref_contracted.hdf5')
    
    def __init__(self):        
        
        self.data = Dict2HDF(self.filepath).load_dict_from_hdf5()        
        self.times = np.asarray([cosmology.age(z).value for z in self.data['Redshifts']]) * unyt.Gyr
        self.times.convert_to_units('Myr')
        
        self.data['Halos'] = [halo.decode('ascii') for halo in self.data['Halos']]
        
    def get_redshift_range(self, z_min: float = 0., z_max: float = 10.) -> dict:
        
        redshift_filter = np.where((self.data['Redshifts'] < z_max) & (self.data['Redshifts'] > z_min))[0]
        selected_data = self.data.copy()
        
        # Filter redshift list
        selected_data['Redshifts'] = selected_data['Redshifts'][redshift_filter]
        selected_data['Times'] = self.times[redshift_filter]
        
        # Filter all other fields by redshift
        for halo_name in selected_data['Halos']:
            for dataset_name, dataset in selected_data[halo_name].items():     
                selected_data[halo_name][dataset_name] = dataset[redshift_filter]
                    
        return selected_data
    
    @staticmethod
    def bin_dataset(dataset: np.ndarray, window: int = 5) -> np.ndarray:
        
        ax = dict(axis=0) if dataset.ndim == 2 else dict()
        smooth_dataset = np.asarray([np.median(dataset[i:i + window], **ax) for i in range(0, len(dataset), window)])
        return smooth_dataset
            
    
    @staticmethod
    def make_title(axes, title, padding=3, **kwargs):
        axes.annotate(
            text=title,
            xy=(0.5, 0.99),
            xytext=(padding - 1, -(padding - 1)),
            textcoords='offset pixels',
            xycoords='axes fraction',
            bbox=dict(facecolor='w', edgecolor='none', alpha=0.65, zorder=0, pad=padding),
            color="k",
            ha="center",
            va="top",
            alpha=0.9,
            **kwargs
        )
                        
if __name__ == '__main__':
    # Generate the data with the RedshiftEvolution instance
    obj = TimeEvolutionCompute()
    
def self_scaling_temperature(mass, radius):
    """
    Computes the self-scaling temperature T500 of a galaxy cluster.

    Parameters:
        mass (float): mass of the cluster within the radius R500 where the mean density is 500 times the critical density of the universe.
        radius (float): the radius R500 

    Returns:
        T500 (float): self-scaling temperature of the cluster.
    """
    G = 6.67408e-11 #m^3 kg^-1 s^-2
    k_B = 1.38064852e-23 #m^2 kg s^-2 K^-1
    T500 = (mass*G/(2*radius))/k_B
    return T500