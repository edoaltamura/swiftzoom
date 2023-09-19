# -*- coding: utf-8 -*-
"""Tools for handling intermediate analysis products

This package contains classes that use the ``pickle`` library to dump
and read back into memory intermediate products of the analysis.

Generally, quantities which take a long time to compute and do not
need to be freshly calculated every time should be dumped to disk and
read back when requires.

.. warning::

    Class attributes are lost in pickling
    When you pickled the instance you haven't pickled the class attributes,
    just the instance attributes. So when you unpickle it you get just the
    instance attributes back.
"""
import os
from warnings import warn
from pandas import DataFrame, read_pickle
import h5py
import numpy as np

try:
    import _pickle as pickle

    # `HIGHEST_PROTOCOL` attribute not defined in `_pickle`
    pickle.HIGHEST_PROTOCOL = -1

except ModuleNotFoundError:
    import pickle

default_output_directory = '/cosma/home/dp004/dc-alta2/shocks_analysis/data/02_interim'


def sizeof_fmt(num: float, suffix: str = "B") -> str:
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:

        if abs(num) < 1024.:
            return "%3.1f%s%s" % (num, unit, suffix)

        num /= 1024.0

    return "%.1f%s%s" % (num, "Yi", suffix)


class CustomPickler(object):

    def __init__(self, filename: str, relative_path: bool = True) -> None:

        if relative_path:
            self.filename = os.path.join(default_output_directory, filename)

        else:
            self.filename = filename

    def large_file_warning(self) -> None:

        file_size_b = os.path.getsize(self.filename)

        if file_size_b > 524288000:
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


class SingleObjPickler(CustomPickler):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def dump_to_pickle(self, obj):
        with open(self.filename, "wb") as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        file_size = sizeof_fmt(os.path.getsize(self.filename))
        print(f"[io] Object saved to pkl [{file_size:s}]: {self.filename:s}")

    def load_from_pickle(self):
        self.large_file_warning()

        with open(self.filename, "rb") as pickle_file:
            content = pickle.load(pickle_file)

        return content


class MultiObjPickler(CustomPickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def dump_to_pickle(self, obj_collection):
        with open(self.filename, "wb") as output:  # Overwrites any existing file.
            for obj in obj_collection:
                pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        file_size = sizeof_fmt(os.path.getsize(self.filename))
        print(f"[io] Object saved to pkl [{file_size:s}]: {self.filename:s}")

    def get_pickle_generator(self):
        """Unpickle a file of pickled data."""
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"File {self.filename} not found.")

        file_size = sizeof_fmt(os.path.getsize(self.filename))
        print(f"[io] Loading from pkl [{file_size:s}]: {self.filename:s}...")

        with open(self.filename, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    def load_from_pickle(self):
        self.large_file_warning()
        collection_pkl = []
        for obj in self.get_pickle_generator():
            collection_pkl.append(obj)
        return collection_pkl


class DataframePickler(CustomPickler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def dump_to_pickle(self, obj: DataFrame):
        # Save data in pickle format (saves python object)
        obj.to_pickle(self.filename)

        # Save data in text format (useful for consulting)
        obj.to_csv(
            self.filename.replace("pkl", "txt"),
            header=True,
            index=False,
            sep=",",
            mode="w",
        )

        file_size = sizeof_fmt(os.path.getsize(self.filename))
        print(f"[io] Object saved to pkl [{file_size:s}]: {self.filename:s}")

    def load_from_pickle(self):
        self.large_file_warning()

        return read_pickle(self.filename)


class Dict2HDF(CustomPickler):

    def __init__(self, *args, **kwargs) -> None:
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

            if isinstance(item, (np.ndarray, bool, int, float, str, bytes)):

                try:
                    h5file[path + key] = item
                except:
                    print(f'Could not save <{path + key:s}> of type <{type(item):s}>')

            elif isinstance(item, dict):
                self._recursively_save_dict_contents_to_group(h5file, path + key + "/", item)

            else:
                raise TypeError(f"Cannot save {type(item):s} type")

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

                out = item[...]
                if isinstance(out, np.ndarray) and out.size == 1:
                    if out.dtype == float:
                        out = float(out)
                    elif out.dtype == int:
                        out = int(out)
                    elif out.dtype == bool:
                        out = bool(out)
                    elif out.dtype == object:
                        out = out.tolist().decode('utf-8')
                ans[key] = out

            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = self._recursively_load_dict_contents_from_group(h5file, path + key + "/")
        return ans
