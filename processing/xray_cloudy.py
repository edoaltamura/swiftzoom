import h5py
import os.path
import numpy as np
from numba import jit

class Interpolate:

    cloudy_table_path = '/cosma/home/dp004/dc-alta2/xl-zooms/xray/X_Ray_table.hdf5'
    
    def init(self):
        
        with h5py.File(self.cloudy_table_path, 'r') as cloudy_table:
            self.X_Ray = cloudy_table['0.5-2.0keV']['emissivities'][:]
            self.He_bins = cloudy_table['/Bins/He_bins'][:]
            self.missing_elements = cloudy_table['/Bins/Missing_element'][:]
            self.density_bins = cloudy_table['/Bins/Density_bins/'][:]
            self.temperature_bins = cloudy_table['/Bins/Temperature_bins/'][:]
            self.solar_metallicity = cloudy_table['/Bins/Solar_metallicities/'][:]
            
        self.dn = 0.2
        self.dT = 0.1

@jit(nopython=True)
def find_dx(subdata, bins, idx_0):
    dx_p = np.zeros(len(subdata))
    for i in range(len(subdata)):
        dx_p[i] = np.abs(bins[idx_0[i]] - subdata[i])

    return dx_p


def find_idx(subdata, bins, dbins):
    idx_p = np.zeros((len(subdata), 2))
    for i in range(len(subdata)):
        idx_p[i, :] = np.sort(np.argsort(np.abs(bins - subdata[i]))[:2])

    return idx_p


@jit(nopython=True)
def find_idx_he(subdata, bins):
    num_bins = len(bins)
    idx_p = np.zeros((len(subdata), 2))
    for i in range(len(subdata)):

        # When closest to the highest bin, or above the highest bin, return the one but highest bin,
        # otherwise we will select a second bin which is outside the binrange
        bin_below = min(np.argsort(np.abs(bins[bins <= subdata[i]] - subdata[i]))[0], num_bins - 2)
        idx_p[i, :] = np.array([bin_below, bin_below + 1])

    return idx_p


@jit(nopython=True)
def find_dx_he(subdata, bins, idx_0):
    dx_p = np.zeros(len(subdata))
    for i in range(len(subdata)):
        dx_p[i] = np.abs(subdata[i] - bins[idx_0[i]]) / (bins[idx_0[i] + 1] - bins[idx_0[i]])

    return dx_p


@jit(nopython=True)
def get_table_interp(dn, dT, dx_T, dx_n, idx_T, idx_n, idx_he, dx_he, X_Ray, abundance_to_solar):
    f_n_T_Z = np.zeros(len(idx_n[:, 0]))
    for i in range(len(idx_n[:, 0])):
        # interpolate He
        f_000 = X_Ray[0, idx_he[i, 0], :, idx_T[i, 0], idx_n[i, 0]]
        f_001 = X_Ray[0, idx_he[i, 0], :, idx_T[i, 0], idx_n[i, 1]]
        f_010 = X_Ray[0, idx_he[i, 0], :, idx_T[i, 1], idx_n[i, 0]]
        f_011 = X_Ray[0, idx_he[i, 0], :, idx_T[i, 1], idx_n[i, 1]]

        f_100 = X_Ray[0, idx_he[i, 1], :, idx_T[i, 0], idx_n[i, 0]]
        f_101 = X_Ray[0, idx_he[i, 1], :, idx_T[i, 0], idx_n[i, 1]]
        f_110 = X_Ray[0, idx_he[i, 1], :, idx_T[i, 1], idx_n[i, 0]]
        f_111 = X_Ray[0, idx_he[i, 1], :, idx_T[i, 1], idx_n[i, 1]]

        f_00 = f_000 * (1 - dx_he[i]) + f_100 * dx_he[i]
        f_01 = f_001 * (1 - dx_he[i]) + f_101 * dx_he[i]
        f_10 = f_010 * (1 - dx_he[i]) + f_110 * dx_he[i]
        f_11 = f_011 * (1 - dx_he[i]) + f_111 * dx_he[i]

        # interpolate density
        f_n_T0 = (dn - dx_n[i]) / dn * f_00 + dx_n[i] / dn * f_01
        f_n_T1 = (dn - dx_n[i]) / dn * f_10 + dx_n[i] / dn * f_11

        # interpolate temperature
        f_n_T = (dT - dx_T[i]) / dT * f_n_T0 + dx_T[i] / dT * f_n_T1

        # Apply linear scaling for removed metals
        f_n_T_Z_temp = f_n_T[-1]
        for j in range(len(f_n_T) - 1):
            f_n_T_Z_temp -= (f_n_T[-1] - f_n_T[j]) * abundance_to_solar[i, j]

        f_n_T_Z[i] = f_n_T_Z_temp

    return f_n_T_Z


def interpolate_X_Ray(data_n, data_T, element_mass_fractions, fill_value=-50.):
    
    # Initialise interpolation class
    interp = Interpolate()

    # Initialise the emissivity array which will be returned
    emissivities = np.zeros_like(data_n, dtype=float)

    # Create density mask, round to avoid numerical errors
    density_mask = (
        (data_n >= np.round(interp.density_bins.min(), 1)) & 
        (data_n <= np.round(interp.density_bins.max(), 1))
    )
    # Create temperature mask, round to avoid numerical errors
    temperature_mask = (
        (data_T >= np.round(interp.temperature_bins.min(), 1)) & 
        (data_T <= np.round(interp.temperature_bins.max(), 1))
    )

    # Combine masks
    joint_mask = density_mask & temperature_mask

    # Check if within density and temperature bounds
    density_bounds = np.sum(density_mask) == density_mask.shape[0]
    temperature_bounds = np.sum(temperature_mask) == temperature_mask.shape[0]
    
    if not all(density_bounds & temperature_bounds):
        
        # If no fill_value is set, return an error with some explanation
        if fill_value == None:
            raise ValueError(
                'Temperature or density are outside of the interpolation range and no fill_value is supplied\n' +
                '\tTemperature ranges between log(T) = [5, 9.5]\n' +
                '\t    Density ranges between log(nH) = [-8, 6]\n' +
                'Set the kwarg `fill_value = some value` to set all particles outside of the interpolation range to `some value` ' +
                'or limit your particle data set to be within the interpolation range.'
            )
        
        emissivities[~joint_mask] = fill_value

    mass_fraction = np.zeros((len(data_n[joint_mask]), 9))

    # get individual mass fraction
    mass_fraction[:, 0] = element_mass_fractions.hydrogen[joint_mask]
    mass_fraction[:, 1] = element_mass_fractions.helium[joint_mask]
    mass_fraction[:, 2] = element_mass_fractions.carbon[joint_mask]
    mass_fraction[:, 3] = element_mass_fractions.nitrogen[joint_mask]
    mass_fraction[:, 4] = element_mass_fractions.oxygen[joint_mask]
    mass_fraction[:, 5] = element_mass_fractions.neon[joint_mask]
    mass_fraction[:, 6] = element_mass_fractions.magnesium[joint_mask]
    mass_fraction[:, 7] = element_mass_fractions.silicon[joint_mask]
    mass_fraction[:, 8] = element_mass_fractions.iron[joint_mask]

    # From Cooling tables, integrate into X-Ray tables in a future version
    solar_mass_fractions = np.array([
        7.3738831e-01, 2.4924204e-01, 2.3647151e-03, 6.9290539e-04,
        5.7326527e-03, 1.2564836e-03, 7.0797943e-04, 6.6494779e-04,
        1.2919896e-03
        ])

    # At the moment the tables only go up to solar metallicity for He
    # Enforce this for all metals to have consistency
    max_metallicity = 1
    clip_mass_fractions = solar_mass_fractions * max_metallicity
    # Reset such to mass fractions of Hydrogen = 1
    clip_mass_fractions /= clip_mass_fractions[0]
    mass_fraction = np.clip(mass_fraction, a_min=0, a_max=clip_mass_fractions)

    # Find density offsets
    idx_n = find_idx(data_n[joint_mask], interp.density_bins, interp.dn)
    dx_n = find_dx(data_n[joint_mask], interp.density_bins, idx_n[:, 0].astype(int))

    # Find temperature offsets
    idx_T = find_idx(data_T[joint_mask], interp.temperature_bins, interp.dT)
    dx_T = find_dx(data_T[joint_mask], interp.temperature_bins, idx_T[:, 0].astype(int))

    # Find element offsets
    # mass of ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    element_masses = [1, 4.0026, 12.0107, 14.0067, 15.999, 20.1797, 24.305, 28.0855, 55.845]

    # Calculate the abundance wrt to solar
    abundances = mass_fraction / np.array(element_masses)
    abundance_to_solar = 1 - abundances / 10 ** interp.solar_metallicity

    # Add columns for Calcium and Sulphur and add Iron at the end
    abundance_to_solar = np.c_[
        abundance_to_solar[:, :-1], 
        abundance_to_solar[:, -2], 
        abundance_to_solar[:, -2], 
        abundance_to_solar[:, -1]
    ]

    # Find helium offsets
    idx_he = find_idx_he(np.log10(abundances[:, 1]), interp.He_bins)
    dx_he = find_dx_he(np.log10(abundances[:, 1]), interp.He_bins, idx_he[:, 0].astype(int))

    emissivities[joint_mask] = get_table_interp(
        interp.dn, 
        interp.dT, 
        dx_T, 
        dx_n, 
        idx_T.astype(int), 
        idx_n.astype(int), 
        idx_he.astype(int), 
        dx_he, 
        interp.X_Ray, 
        abundance_to_solar[:, 2:]
    )

    return emissivities


def logsumexp(
        a: np.ndarray,
        axis: int = None,
        b: np.ndarray = None,
        keepdims: bool = False,
        return_sign: bool = False,
        base: float = 10.
) -> float:
    """
    Compute the log of the sum of exponentials of input elements.
    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.

    base : float, optional
        This base is used in the exponentiation and the logarithm.
        $\log_{base} \sum (base)^{array}$

    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        implement subtraction.

    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False (no sign information).

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.

    sgn : ndarray
        If return_sign is True, this will be an array of floating-point
        numbers matching res and +1, 0, or -1 depending on the sign
        of the result. If False, only one result is returned.

    See Also
    --------
    numpy.logaddexp, numpy.logaddexp2

    Notes
    -----
    NumPy has a logaddexp function which is very similar to `logsumexp`, but
    only handles two arguments. `logaddexp.reduce` is similar to this
    function, but may be less stable.
    """
    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -np.inf

    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        if base is None or base == np.e:
            tmp = b * np.exp(a - a_max)
        else:
            tmp = b * base ** (a - a_max)
    else:
        if base is None or base == np.e:
            tmp = np.exp(a - a_max)
        else:
            tmp = base ** (a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero

        log_base_conversion = 1 / np.log(base)
        out = np.log(s) * log_base_conversion

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out
