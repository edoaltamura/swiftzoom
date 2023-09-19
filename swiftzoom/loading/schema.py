import pathlib
import yaml
import warnings
from typing import List


patterns = {
    "swift_parameters"   : ["*.yml"],
    "output_list"        : ["output_list.txt", "snap_redshifts.txt"],
    "star_formation_rate": ["SFR.txt"],
    "statistics"         : ["statistics.txt"],
    "timesteps"          : ["timesteps*.txt"],
    "out_files"          : ["*.err", ".*out"],
    "snapshots"          : ["*.hdf5"],
    "stf"                : [],
}


class DirectorySchema(object):

    def __init__(self, run_directory: str, ignore_warnings: bool = False):
        """Initialize a DirectorySchema instance.

        Args:
            run_directory (str): The path to the directory to be analyzed.
            ignore_warnings (bool, optional): If True, suppress warnings. Defaults to False.
        """
        self.root = pathlib.Path(run_directory)
        self.quiet = ignore_warnings

        self.validate_swift_outputs()
        self.validate_swift_outputs()

    def get_extension_matches(self, key: str, recursive: bool = False) -> List[pathlib.Path]:
        """Get files in the directory matching specified extensions.

        Args:
            key (str): The key to identify the pattern from `patterns`.
            recursive (bool, optional): If True, search files recursively. Defaults to False.

        Returns:
            List[pathlib.Path]: A list of pathlib.Path objects representing matching files.
        """
        if recursive:
            return [p for p in self.root.rglob('*') if p.suffix in patterns[key]]

        return [p for p in self.root.iterdir() if p.suffix in patterns[key]]

    def validate_swift_outputs(self) -> None:
        """Validate SWIFT outputs.

        Raises:
            yaml.YAMLError: If there is an issue parsing a YAML file.
            UserWarning: If SWIFT parameter files or other SWIFT-related files are not found.
        """

        swift_parameter_file_found = False
        for file in self.get_extension_matches('swift_parameters'):

            if 'used' not in file:

                with open(file.resolve(), 'r') as stream:
                    try:
                        return yaml.load(stream)
                    except yaml.YAMLError as exception:
                        raise exception

                swift_parameter_file_found = True

        if swift_parameter_file_found or self.quiet:
            pass
        else:
            warnings.warn(UserWarning,
                          f"SWIFT parameter file not found in {self.root}. Expecting pattern {patterns['swift_parameters']}.")

        for sw_file in ["output_list", "star_formation_rate", "statistics", "timesteps"]:

            _file_found = False
            for _ in self.get_extension_matches(sw_file):
                _file_found = True

            if _file_found or self.quiet:
                continue

            warnings.warn(UserWarning,
                          f"SWIFT sw_file not found in {self.root}. Expecting pattern {patterns[sw_file]}.")

    def validate_snapshots(self):
        """
        Validate snapshot files.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError

    def validate_catalogues(self):
        """
        Validate catalogue files.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError

    def validate_std_logs(self):
        """
        Validate standard logs (stdout and stderr).

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError
