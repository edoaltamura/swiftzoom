import numpy as np
import unyt
import astropy
from typing import Optional, Union
from swiftsimio import cosmo_array

def astropy_to_unyt(astropy_quantity: astropy.units.quantity.Quantity) -> unyt.unyt_quantity:
    """
    Convert an astropy quantity to a unyt quantity.

    Parameters:
    astropy_quantity (astropy.units.quantity.Quantity): An astropy quantity to be converted to unyt.

    Returns:
    unyt.unyt_quantity: A new unyt quantity with the same value and units as the input astropy quantity.

    Example:
    >>> import astropy.units as u
    >>> import unyt
    >>> q = 10*u.m
    >>> unyt_q = astropy_to_unyt(q)
    >>> print(unyt_q)
    10.0 m
    """    
    value = astropy_quantity.value
    units_str = astropy_quantity.unit.to_string(format='ogip')
    
    return unyt.unyt_quantity(value, units_str)


def histogram_unyt(data: unyt.unyt_array,
                   bins: unyt.unyt_array,
                   weights: Optional[unyt.unyt_array] = None,
                   normalizer: Optional[unyt.unyt_array] = None,
                   replace_zero_nan: Optional[bool] = True) -> unyt.unyt_array:
    """
    Compute a histogram of the input data with bin edges given by bins.
    This is a soft wrapper around numpy.histogram to operate with unyt_array
    objects, and it also provides extra functionalities for weighting the
    dataset w.r.t. a separate quantity provided by the normalizer. It can
    optionally replace zeros in the final histogram with NaNs. Only supports
    1D binning.
    
    TODO: switch from np.histogram to scipy.binned_statistic, which has more function options.

    Parameters
    ----------
    data : unyt.unyt_array
        The array to bin, with the same units as bins. For example, in a radial
        profile, this would accept the radial distance of particles.
    bins : unyt.unyt_array
        The bin edges with size (number_bins + 1), with the same units as data.
        For example, in a radial profile, this would accept the intervals at
        which to bin radial shells.
    weights : unyt.unyt_array
        The weights to apply to the data array. For example, in a radial
        profile, this could be the mass of particles for a mass-weighted
        profile. The histogram returned contains the sum of the weights in
        each bin.
    normalizer : unyt.unyt_array, optional
        An additional dataset to provide extra flexibility for normalization.
        Unlike weights, the normalizer computes the sum of (weights * normalizer)
        in each bin and divides this result by the sum of normalizer in each bin.
        This measures the average normalizer-weighted quantity in each bin.
        Normalizer units cancel out in the division. Default is None.
    replace_zero_nan : bool, optional
        Set to NaN bins with zero counts. Default is True.

    Returns
    -------
    hist : unyt.unyt_array
        The binned histogram, weighted and normalized. Bin edges are not
        returned.

    Raises
    ------
    AssertionError
        If data and weights arrays do not have the same shape, if data and bins
        do not have the same units, or if data and normalizer arrays do not have
        the same shape.

    """
    if weights is None:
        weights = np.ones_like(data.value) * unyt.dimensionless
        
    assert data.shape == weights.shape, \
        f"Data and weights arrays must have the same shape. Detected data {data.shape}, weights {weights.shape}."
    assert data.units == bins.units, \
        f"Data and bins must have the same units. Detected data {data.units}, bins {bins.units}."
        
    if normalizer is not None:
        assert data.shape == normalizer.shape, \
            f"Data and normalizer arrays must have the same shape. Detected data {data.shape}, normalizer {normalizer.shape}."

        hist, bin_edges = np.histogram(data.value, bins=bins.value, weights=weights.value * normalizer.value)
        hist *= weights.units * normalizer.units

        norm, _ = np.histogram(data.value, bins=bins.value, weights=normalizer.value)
        norm *= normalizer.units

        hist /= norm
    else:
        hist, bin_edges = np.histogram(data.value, bins=bins.value, weights=weights.value)
        hist *= weights.units

    if replace_zero_nan:
        hist[hist == 0] = np.nan

    assert hist.units == weights.units

    return hist


def cumsum_unyt(data: unyt.unyt_array, ignore_nan: bool = True) -> unyt.unyt_array:
    """
    Compute the cumulative sum of an input array, while preserving its units.

    Parameters
    ----------
    data : unyt.unyt_array
        Input array with units to be cumulatively summed.
    ignore_nan : bool, optional
        Use the `nancumsum` instead of `cumsum`. This is the same as
        setting to NaN bins with zero counts. Default is True.

    Returns
    -------
    unyt.unyt_array
        A new array containing the cumulative sum of the input array,
        with the same units as the input array.

    Notes
    -----
    This function uses the NumPy `cumsum` function to compute the cumulative
    sum of the input array's values, and then multiplies the resulting array
    by the input array's units to preserve the units in the output.

    Examples
    --------
    >>> import unyt
    >>> import numpy as np
    >>> x = unyt.unyt_array([1, 2, 3], "m")
    >>> cumsum_unyt(x)
    unyt_array([1, 3, 6], 'm')
    """
    cumsum_function = np.nancumsum
    
    if not ignore_nan:
        cumsum_function = np.cumsum
    
    res = cumsum_function(data.value)
    
    return res * data.units


def numpy_to_cosmo_array(input_array: Union[float, np.ndarray, unyt.unyt_quantity, unyt.unyt_array], 
                         template_cosmo_array: cosmo_array) -> cosmo_array:
    """_summary_

    Parameters
    ----------
    input_array (Union[float, np.ndarray, unyt.unyt_quantity, unyt.unyt_array]): The input array or float to convert
    template_cosmo_array (cosmo_array): The template cosmo_array to draw unit and scale-factor information from

    Returns
    -------
    cosmo_array: Final cosmo_array with data from input_array and cosmo information from the template
    """
    
    # Convert to cosmo_array to reconcile with SWIFT input
    return cosmo_array(np.array(input_array),
                       units=template_cosmo_array.units,
                       cosmo_factor=template_cosmo_array.cosmo_factor,
                       comoving=template_cosmo_array.comoving)
    
def cosmo_to_unyt_array(input_array: cosmo_array) -> unyt.unyt_array:
    
    # Convert to cosmo_array to reconcile with SWIFT input
    return unyt.unyt_array(np.array(input_array.value), units=input_array.units)
