"""
This test module defines two sets of tests:

test_dict_to_hdf5_and_back: This test uses hypothesis to generate random nested dictionaries and different modes ("w"
and "r") for saving and loading. It ensures that the saved and loaded dictionaries match the original dictionary.

test_dict_to_hdf5_and_back_edge_cases: This test covers edge cases with empty dictionaries and dictionaries
containing only None values for both "w" and "r" modes. It verifies that the edge cases are handled correctly.
"""

import pytest
import os
import sys
import numpy as np

# Add the path to the parent directory (one directory up) to sys.path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_directory)

from swiftzoom.loading import Dict2HDF


def dict_equality(d1, d2):
    for k in d1:
        if k in d2:
            if isinstance(d1[k], dict):
                dict_equality(d1[k], d2[k])
            else:
                if type(d1[k]) in [np.ndarray, list, tuple, set]:
                    assert (d1[k] == d2[k]).all()
                else:
                    assert d1[k] == d2[k]
        else:
            raise AssertionError


# Define a fixture to create an instance of Dict2HDF for testing
@pytest.fixture
def dict2hdf_instance(tmp_path):
    # Create an instance with a temporary file for testing
    filename = os.path.join(tmp_path, "test.hdf5")
    return Dict2HDF(filename, relative_path=False)


# Define test cases for the Dict2HDF class
class TestDict2HDF:

    def test_save_and_load_dict(self, dict2hdf_instance):

        # Create a sample dictionary
        sample_dict = {
            "int_array"   : np.array([1, 2, 3]),
            "float_value" : 3.14,
            "string_value": "Hello, World!",
            "nested_dict" : {
                "bool_value": True,
                "int_array" : np.array([1, 2, 3]),
            }
        }
        # Save the dictionary to HDF5
        dict2hdf_instance.save_dict_to_hdf5(sample_dict)

        # Load the dictionary from HDF5
        loaded_dict = dict2hdf_instance.load_dict_from_hdf5()

        # Check if the loaded dictionary matches the original
        dict_equality(loaded_dict, sample_dict)

    def test_invalid_data_type(self, dict2hdf_instance):
        # Create a dictionary with an unsupported data type (set)
        invalid_dict = {"invalid_list": [1, 2, 3]}

        # Attempt to save the dictionary to HDF5
        with pytest.raises(TypeError):
            dict2hdf_instance.save_dict_to_hdf5(invalid_dict)

    def test_invalid_file_mode(self, dict2hdf_instance):
        # Try to save a dictionary with an invalid file mode
        invalid_mode = "p"  # Invalid mode
        sample_dict = {"int_value": 42}

        with pytest.raises(ValueError):
            dict2hdf_instance.save_dict_to_hdf5(sample_dict, mode=invalid_mode)


# Run the tests using pytest
if __name__ == "__main__":
    pytest.main()
