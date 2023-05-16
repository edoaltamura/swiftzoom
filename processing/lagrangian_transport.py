import sys
import copy
import os
import numpy as np
import unyt
from collections import namedtuple
from typing import Union, Optional
from scipy.spatial import distance

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import swiftsimio
import velociraptor

from output_list import OutputList

Snapshot = namedtuple(
    "Snapshot",
    "run_directory, snap_number, redshift, snapshot_handle, halo_catalogue_handle, r500, m500, centre_of_halo, output_list",
)


class LagrangianTransport(object):
    def __init__(
        self,
        run_directory: str,
        reference_redshift: float,
    ):
        self.reference = self.parse_data(run_directory, reference_redshift)

    def parse_data(
        self,
        run_directory: str,
        reference_redshift: float,
        halo_catalogue_index: int = 0,
        mask_radius_r500_comoving: float = 3,
        convert_to_physical: bool = True,
    ) -> Snapshot:

        output_list = OutputList(run_directory)
        path_to_snap, path_to_catalogue = output_list.files_from_redshift(
            reference_redshift
        )
        nearest_redshift, snap_number = output_list.match_redshift(reference_redshift)

        # Get basic data from Velociraptor catalogue
        vr_data = velociraptor.load(path_to_catalogue, disregard_units=True)

        m500 = vr_data.spherical_overdensities.mass_500_rhocrit[
            halo_catalogue_index
        ].to("Msun")
        r500 = vr_data.spherical_overdensities.r_500_rhocrit[halo_catalogue_index].to(
            "Mpc"
        )
        xcminpot = vr_data.positions.xcminpot[halo_catalogue_index].to("Mpc")
        ycminpot = vr_data.positions.ycminpot[halo_catalogue_index].to("Mpc")
        zcminpot = vr_data.positions.zcminpot[halo_catalogue_index].to("Mpc")
        centre_of_halo = unyt.unyt_array([xcminpot, ycminpot, zcminpot])

        m500_comoving = m500 / vr_data.a
        r500_comoving = r500 / vr_data.a
        centre_of_halo_comoving = centre_of_halo / vr_data.a

        # Apply spatial mask to particles. SWIFTsimIO needs comoving coordinates
        # to filter particle coordinates, while VR outputs are in physical units.
        # Convert the region bounds to comoving, but keep the CoP and Rcrit in
        # physical units for later use.
        swift_mask = swiftsimio.mask(path_to_snap)
        mask_radius = mask_radius_r500_comoving * r500_comoving
        region = [
            [
                centre_of_halo_comoving[0] - mask_radius,
                centre_of_halo_comoving[0] + mask_radius,
            ],
            [
                centre_of_halo_comoving[1] - mask_radius,
                centre_of_halo_comoving[1] + mask_radius,
            ],
            [
                centre_of_halo_comoving[2] - mask_radius,
                centre_of_halo_comoving[2] + mask_radius,
            ],
        ]
        swift_mask.constrain_spatial(region)
        sw_data = swiftsimio.load(path_to_snap, mask=swift_mask)

        particle_types = ["gas", "dark_matter", "stars", "black_holes"]

        for particle_type in particle_types:
            if getattr(sw_data.metadata, f"n_{particle_type:s}") > 0:

                # If the mask overlaps with the box boundaries, wrap coordinates.
                getattr(sw_data, particle_type).coordinates = self.wrap_coordinates(
                    getattr(sw_data, particle_type).coordinates,
                    centre_of_halo_comoving,
                    sw_data.metadata.boxsize,
                )

                # Compute radial distances
                getattr(
                    sw_data, particle_type
                ).radial_distances = self.get_radial_distance(
                    getattr(sw_data, particle_type).coordinates, centre_of_halo_comoving
                )

                # Compute coordinates in the CoP frame (CoP at the origin)
                getattr(
                    sw_data, particle_type
                ).centered_coordinates = self.center_coordinates(
                    getattr(sw_data, particle_type).coordinates, centre_of_halo_comoving
                )

                if convert_to_physical:

                    field_names = getattr(
                        getattr(sw_data, particle_type).metadata,
                        f"{particle_type}_properties",
                    ).field_names

                    for field_name in field_names:

                        field = getattr(getattr(sw_data, particle_type), field_name)

                        # Check if the datasets is integer type. If undefined (e.g. GasElementMassFractionsColumns)
                        # then set as integer to avoid comoving conversion.
                        if hasattr(field, "dtype"):
                            is_integer_type = np.issubdtype(field.dtype, np.integer)
                        else:
                            is_integer_type = True

                        if (
                            hasattr(field, "convert_to_physical")
                            and not is_integer_type
                        ):
                            field.convert_to_physical()

        return Snapshot(
            run_directory=run_directory,
            snap_number=snap_number,
            redshift=sw_data.metadata.redshift,
            snapshot_handle=sw_data,
            halo_catalogue_handle=vr_data,
            r500=r500,
            m500=m500,
            centre_of_halo=centre_of_halo,
            output_list=output_list,
        )

    @staticmethod
    def center_coordinates(
        coords: swiftsimio.cosmo_array,
        centre: unyt.unyt_array,
    ) -> swiftsimio.cosmo_array:

        assert coords.units == centre.units

        centered_coordinates = coords - centre

        shifted_coords = swiftsimio.cosmo_array(
            centered_coordinates.value,
            units=centered_coordinates.units,
            dtype=centered_coordinates.dtype,
            name="centered_coordinates",
            cosmo_factor=coords.cosmo_factor,
            comoving=coords.comoving,
        )

        return centered_coordinates

    @staticmethod
    def wrap_coordinates(
        coords: swiftsimio.cosmo_array,
        centre: unyt.unyt_array,
        boxsize: unyt.unyt_array,
    ) -> swiftsimio.cosmo_array:

        assert coords.units == centre.units
        assert coords.units == boxsize.units

        wrapped_coordinates = (
            np.mod(coords - centre + 0.5 * boxsize, boxsize) + centre - 0.5 * boxsize
        )

        wrapped_coords = swiftsimio.cosmo_array(
            wrapped_coordinates.value,
            units=wrapped_coordinates.units,
            dtype=wrapped_coordinates.dtype,
            name="coordinates",
            cosmo_factor=coords.cosmo_factor,
            comoving=coords.comoving,
        )

        return wrapped_coords

    @staticmethod
    def get_radial_distance(
        coords: swiftsimio.cosmo_array, centre: unyt.unyt_array
    ) -> swiftsimio.cosmo_array:

        radial_distances_value = distance.cdist(
            coords, centre.reshape(1, 3), metric="euclidean"
        ).reshape(
            len(coords),
        )

        radial_distances = swiftsimio.cosmo_array(
            radial_distances_value,
            units=coords.units,
            dtype=radial_distances_value.dtype,
            name="radial_distances",
            cosmo_factor=coords.cosmo_factor,
            comoving=coords.comoving,
        )

        return radial_distances

    def jump_snapshot(
        self, snapshot: Union[str, None, Snapshot], n_jump: int = 0, **kwargs
    ) -> Snapshot:

        if snapshot == "reference" or snapshot == None:
            snapshot = self.reference

        next_redshift = snapshot.output_list.output_redshifts[
            snapshot.snap_number + n_jump
        ]

        return self.parse_data(snapshot.run_directory, next_redshift, **kwargs)

    def select_shell(
        self,
        snapshot: Union[str, None, Snapshot],
        radius: unyt.unyt_quantity = 1 * unyt.dimensionless,
        thickness: unyt.unyt_quantity = 0.01 * unyt.dimensionless,
    ):

        if snapshot == "reference" or snapshot == None:
            snapshot = self.reference

        if radius.units == unyt.dimensionless:
            radius *= snapshot.r500
        if thickness.units == unyt.dimensionless:
            thickness *= snapshot.r500

        assert radius.units in [unyt.Mpc, unyt.kpc] and thickness.units in [
            unyt.Mpc,
            unyt.kpc,
        ]

        index_particles_in_shell = np.where(
            (snapshot.snapshot_handle.gas.radial_distances > radius - thickness / 2)
            & (snapshot.snapshot_handle.gas.radial_distances < radius + thickness / 2)
        )[0]
        particle_ids = snapshot.snapshot_handle.gas.particle_ids[
            index_particles_in_shell
        ]

        return particle_ids

    def plot_shell(
        self,
        snapshot: Union[str, None, Snapshot],
        particle_ids,
        draw_sphere: bool = True,
        radius_sphere: unyt.unyt_quantity = 1 * unyt.dimensionless,
    ):

        if snapshot == "reference" or snapshot == None:
            snapshot = self.reference

        particle_mask = np.where(
            np.isin(snapshot.snapshot_handle.gas.particle_ids, particle_ids)
        )[0]
        particle_coordinates = snapshot.snapshot_handle.gas.centered_coordinates[
            particle_mask
        ]
        particle_radial_distances = snapshot.snapshot_handle.gas.radial_distances[
            particle_mask
        ]

        fig = plt.figure(figsize=(7, 10))
        ax = plt.axes(projection="3d")
        fig.set_facecolor("white")

        ax.scatter3D(*particle_coordinates.T, s=5, ec="none", fc="k")

        if draw_sphere:

            # Draw centre of the sphere
            ax.scatter3D(0, 0, 0, s=100, color="r", marker="x")

            if radius_sphere.units == unyt.dimensionless:
                radius_sphere *= snapshot.r500

            u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 100j]
            x = radius_sphere * np.cos(u) * np.sin(v)
            y = radius_sphere * np.sin(u) * np.sin(v)
            z = radius_sphere * np.cos(v)
            ax.plot_wireframe(x, y, z, color="lime", alpha=0.1, zorder=0)

        plt.show()

        fig = plt.figure(figsize=(7, 7))
        ax = fig.subplots()
        plt.hist(particle_radial_distances.value, bins=10, histtype="step")
        plt.show()
