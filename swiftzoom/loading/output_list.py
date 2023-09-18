import os
import numpy as np
import pandas as pd
from typing import Tuple, Union

SimpleOutputType = Tuple[str, str]
ExtendedOutputType = Tuple[str, str, int, float]


class OutputList(object):
    """
    A class for managing simulation output lists and finding files associated with specific snapshots or redshifts.

    Args:
        run_directory (str): The path to the directory containing simulation output files.

    Attributes:
        run_directory (str): The path to the simulation output directory.
        number_snapshots_in_outputlist (int): The number of snapshots in the output list.
        output_name (numpy.ndarray): An array containing the names of the simulation outputs (snapshots).
        output_redshifts (numpy.ndarray): An array containing the redshift values corresponding to the snapshots.

    Methods:
        match_redshift(redshift_query: float) -> Tuple[float, int]:
            Find the closest redshift in the output list to a given redshift_query.

        files_from_snap_number(snapshot_number: int, extra_returns: bool = False) -> Union[str, Tuple[str, str, int, float]]:
            Retrieve file paths associated with a specific snapshot number.

        files_from_redshift(redshift_query: float, extra_returns: bool = False) -> Union[str, Tuple[str, str, int, float]]:
            Retrieve file paths associated with a specific redshift.

    Example Usage:
    ```
    # Initialize an OutputList instance with the simulation output directory.
    output_list = OutputList('/path/to/simulation/output')

    # Find the nearest redshift to a given query redshift.
    nearest_redshift, snapshot_number = output_list.match_redshift(0.5)
    print(f"Nearest Redshift: {nearest_redshift}, Snapshot Number: {snapshot_number}")

    # Retrieve file paths for a specific snapshot number.
    snapshot_path, catalogue_path = output_list.files_from_snap_number(10)
    print(f"Snapshot Path: {snapshot_path}, Catalogue Path: {catalogue_path}")

    # Retrieve file paths for a specific redshift.
    snapshot_path, catalogue_path = output_list.files_from_redshift(0.5)
    print(f"Snapshot Path: {snapshot_path}, Catalogue Path: {catalogue_path}")
    ```
    """

    def __init__(self, run_directory: str) -> None:
        """
        Initialize the OutputList object by reading the simulation output list and extracting snapshot information.

        Args:
            run_directory (str): The path to the directory containing simulation output files.
        """

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

    def match_redshift(self, redshift_query: float) -> Tuple[np.array, int]:
        """
        Find the closest redshift in the output list to a given redshift_query.

        Args:
            redshift_query (float): The redshift value to match.

        Returns:
            Tuple[float, int]: A tuple containing the nearest redshift and the corresponding snapshot number.
        """

        array = self.output_redshifts
        idx = (np.abs(array - redshift_query)).argmin()
        return array[idx], idx

    def files_from_snap_number(self, snapshot_number: int,
                               extra_returns: bool = False) -> Union[SimpleOutputType, ExtendedOutputType]:
        """
        Retrieve file paths associated with a specific snapshot number.

        Args:
            snapshot_number (int): The snapshot number.
            extra_returns (bool, optional): If True, return additional information.
                Defaults to False.

        Returns:
            Union[str, Tuple[str, str, int, float]]: If extra_returns is False, returns a tuple containing
            the snapshot path and the catalogue path. If extra_returns is True, also includes snapshot number
            and nearest redshift.

        Raises:
            AssertionError: If the snapshot or catalogue file is not found.
        """

        snapshot_path = ''
        for file in os.listdir(os.path.join(self.run_directory, 'snapshots')):
            if file.endswith(f"_{snapshot_number:04d}.hdf5"):
                snapshot_path = os.path.join(self.run_directory, 'snapshots', file)
                break
        assert snapshot_path, f"Could not find snapshot file <{snapshot_number:d}>."

        catalogue_path = ''
        for subdir in os.listdir(os.path.join(self.run_directory, 'stf')):
            if subdir.endswith(f"_{snapshot_number:04d}"):
                for file in os.listdir(os.path.join(self.run_directory, 'stf', subdir)):
                    if file.endswith(f"_{snapshot_number:04d}.properties"):
                        catalogue_path = os.path.join(self.run_directory, 'stf', subdir, file)
                        break

        assert catalogue_path, f"Could not find catalogue file <{snapshot_number:d}>."

        if extra_returns:
            redshift = self.output_redshifts[snapshot_number]
            return snapshot_path, catalogue_path, snapshot_number, redshift
        else:
            return snapshot_path, catalogue_path

    def files_from_redshift(self, redshift_query: float,
                            extra_returns: bool = False) -> Union[SimpleOutputType, ExtendedOutputType]:
        """
        Retrieve file paths associated with a specific redshift.

        Args:
            redshift_query (float): The redshift value to match.
            extra_returns (bool, optional): If True, return additional information.
                Defaults to False.

        Returns: Union[Tuple[str, str, int, float], Tuple[str, str]]:: A tuple containing file paths. If
        `extra_returns` is True, the tuple includes snapshot number and nearest redshift.

        Raises:
            AssertionError: If the snapshot or catalogue file is not found.

        Example Usage:
        ```
        # Initialize an OutputList instance with the simulation output directory.
        output_list = OutputList('/path/to/simulation/output')

        # Retrieve file paths for a specific redshift (without extra information).
        snapshot_path, catalogue_path = output_list.files_from_redshift(0.5)
        print(f"Snapshot Path: {snapshot_path}, Catalogue Path: {catalogue_path}")

        # Retrieve file paths for a specific redshift (with extra information). snapshot_path, catalogue_path,
        snap_number, nearest_redshift = output_list.files_from_redshift(0.5, extra_returns=True) print(f"Snapshot
        Path: {snapshot_path}, Catalogue Path: {catalogue_path}") print(f"Snapshot Number: {snap_number},
        Nearest Redshift: {nearest_redshift}")
        ```
        """
        nearest_redshift, snap_number = self.match_redshift(redshift_query)
        snapshot_path, catalogue_path = self.files_from_snap_number(snap_number)

        if extra_returns:
            return snapshot_path, catalogue_path, snap_number, nearest_redshift
        else:
            return snapshot_path, catalogue_path
