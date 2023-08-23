import warnings
from swiftsimio import SWIFTDataset, cosmo_array
import numpy as np
from unyt import mp, unyt_array, Msun
from typing import Optional, Tuple

from .helper_functions import astropy_to_unyt, histogram_unyt, cumsum_unyt

mean_atomic_weight_per_free_electron = 1.14

element_names = {
    'H' : 'hydrogen',
    'He': 'helium',
    'C' : 'carbon',
    'N' : 'nitrogen',
    'O' : 'oxygen',
    'Ne': 'neon',
    'Mg': 'magnesium',
    'Si': 'silicon',
    'Fe': 'iron',
}

atomic_mass = {
    'H' : 1.00794,
    'He': 4.002602,
    'C' : 12.0107,
    'N' : 14.0067,
    'O' : 15.9994,
    'Ne': 20.1797,
    'Mg': 24.3050,
    'Si': 28.0855,
    'S' : 32.065,
    'Ca': 40.078,
    'Fe': 55.845
}

atomic_number = {
    'H' : 1,
    'He': 2,
    'C' : 6,
    'N' : 7,
    'O' : 8,
    'Ne': 10,
    'Mg': 12,
    'Si': 14,
    'S' : 16,
    'Ca': 20,
    'Fe': 26
}

# Log10(abundances)
solar_abundances = {
    'AG82': {
        'H' : 12,
        'He': 10.99,
        'C' : 8.56,
        'N' : 8.05,
        'O' : 8.93,
        'Ne': 8.09,
        'Mg': 7.58,
        'Si': 7.55,
        'S' : 7.21,
        'Ca': 6.36,
        'Fe': 7.67
    }
}


def get_metal_fractions(sw_data: SWIFTDataset, element_symbol: str, normalise_to_hydrogen: bool = True):
    metal_fraction = getattr(sw_data.gas.element_mass_fractions, element_names[element_symbol]).value

    if normalise_to_hydrogen:
        return metal_fraction / sw_data.gas.element_mass_fractions.hydrogen.value

    return metal_fraction


def get_molecular_weights(sw_data: SWIFTDataset) -> Tuple[float]:
    
    if not hasattr(sw_data.gas, 'element_mass_fractions'):
        warnings.warn("Element mass fractions not found. Reverting to mean_atomic_weight_per_free_electron.", RuntimeWarning)
        return 1., 0., mean_atomic_weight_per_free_electron
    
    ne_nH = np.zeros_like(sw_data.gas.element_mass_fractions.hydrogen.value)
    ni_nH = np.zeros_like(sw_data.gas.element_mass_fractions.hydrogen.value)
    mu = np.zeros_like(sw_data.gas.element_mass_fractions.hydrogen.value)

    # Sum element contributions
    for element in ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Fe']:
        relative_metal_fraction = get_metal_fractions(sw_data, element)
        atomic_mass_ratio = atomic_mass['H'] / atomic_mass[element]

        ne_nH += relative_metal_fraction * atomic_mass_ratio * (atomic_number[element] / atomic_number['H'])
        ni_nH += relative_metal_fraction * atomic_mass_ratio
        mu += relative_metal_fraction / (atomic_number[element] + atomic_number['H'])

    # Silicon, Sulphur & Calcium (abundances relative to Si)
    Si_H = get_metal_fractions(sw_data, 'Si')
    ne_nH += Si_H * (atomic_mass['H'] / atomic_mass['Si']) * (atomic_number['Si'] / atomic_number['H'])
    ni_nH += Si_H * (atomic_mass['H'] / atomic_mass['Si'])
    mu += Si_H / (atomic_number['Si'] + atomic_number['H'])

    Ca_Si = 0.0941736
    ne_nH += (Si_H * Ca_Si) * (atomic_mass['H'] / atomic_mass['Ca']) * (atomic_number['Ca'] / atomic_number['H'])
    ni_nH += (Si_H * Ca_Si) * (atomic_mass['H'] / atomic_mass['Ca'])
    mu += (Si_H * Ca_Si) / (atomic_number['Ca'] + atomic_number['H'])

    S_Si = 0.6054160
    ne_nH += (Si_H * S_Si) * (atomic_mass['H'] / atomic_mass['S']) * (atomic_number['S'] / atomic_number['H'])
    ni_nH += (Si_H * S_Si) * (atomic_mass['H'] / atomic_mass['S'])
    mu += (Si_H * S_Si) / (atomic_number['S'] + atomic_number['H'])

    Xe = ne_nH  # = ne / nH
    Xi = ni_nH  # = ni / nH

    return Xe, Xi, mu


def get_electron_number_density(sw_data: SWIFTDataset) -> cosmo_array:
    Xe, Xi, mu = get_molecular_weights(sw_data)
    
    sw_data.gas.densities.convert_to_physical()
    electron_number_density = (Xe / (Xe + Xi)) * sw_data.gas.densities.to('g*cm**-3') / (mu * mp)

    electron_number_density.convert_to_units('cm**-3')
    electron_number_density = cosmo_array(
        electron_number_density.value,
        units=electron_number_density.units,
        cosmo_factor=sw_data.gas.densities.cosmo_factor
    )

    return electron_number_density


def get_electron_weighted_gas_mass(sw_data: SWIFTDataset) -> cosmo_array:
    Xe, Xi, mu = get_molecular_weights(sw_data)
    
    sw_data.gas.masses.convert_to_physical()
    electron_gas_mass = (Xe / (Xe + Xi)) * sw_data.gas.masses.to('Solar_Mass') / mu

    electron_gas_mass.convert_to_units('Solar_Mass')
    electron_gas_mass = cosmo_array(
        electron_gas_mass.value,
        units=electron_gas_mass.units,
        cosmo_factor=sw_data.gas.masses.cosmo_factor
    )

    return electron_gas_mass


def get_electron_number_density_shell_average(
        sw_data: SWIFTDataset,
        bins: unyt_array,
        normalizer: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
) -> unyt_array:
    normalise_flag = normalizer is not None and normalizer.astype(int).all() != 1

    if mask is None:
        mask = ...

    if xlargs.debug:
        print(f"[Electron number fractions] normalise_flag: {normalise_flag}")

    Xe, Xi, mu = get_molecular_weights(sw_data)
    
    sw_data.gas.masses.convert_to_physical()
    mass = sw_data.gas.masses.value.astype(np.float64) * 1e10 * Msun
    
    electron_number = (Xe / (Xe + Xi)) * mass / (mu * mp)
    volume_shell = (4. * np.pi / 3.) * (bins[1:] ** 3 - bins[:-1] ** 3)

    electron_number_weights = histogram_unyt(
        sw_data.gas.radial_distances[mask],
        bins=bins,
        weights=electron_number[mask],
        normalizer=normalizer
    )
    electron_number_density = electron_number_weights / volume_shell
    electron_number_density.convert_to_units('cm**-3')

    return electron_number_density


def get_scaled_abundances(sw_data: SWIFTDataset, abundances_ref: str = 'AG82'):
    # -- Sum element contributions
    # Hydrogen
    H_H = get_metal_fractions(sw_data, 'H')
    atomic_mass_ratio = atomic_mass['H'] / atomic_mass['H']
    element_abundance_ratio = solar_abundances[abundances_ref]['He'] - solar_abundances[abundances_ref]['H']
    H_H = 10.0 ** (np.log10(H_H * atomic_mass_ratio) - element_abundance_ratio)

    # Helium
    He_H = get_metal_fractions(sw_data, 'He')
    atomic_mass_ratio = atomic_mass['H'] / atomic_mass['He']
    element_abundance_ratio = solar_abundances[abundances_ref]['He'] - solar_abundances[abundances_ref]['H']
    He_H = 10.0 ** (np.log10(He_H * atomic_mass_ratio) - element_abundance_ratio)

    # Carbon
    C_H = get_metal_fractions(sw_data, 'C')
    atomic_mass_ratio = atomic_mass['H'] / atomic_mass['C']
    element_abundance_ratio = solar_abundances[abundances_ref]['C'] - solar_abundances[abundances_ref]['H']
    C_H = 10.0 ** (np.log10(C_H * atomic_mass_ratio) - element_abundance_ratio)

    # Nitrogen
    N_H = get_metal_fractions(sw_data, 'N')
    atomic_mass_ratio = atomic_mass['H'] / atomic_mass['N']
    element_abundance_ratio = solar_abundances[abundances_ref]['N'] - solar_abundances[abundances_ref]['H']
    N_H = 10.0 ** (np.log10(N_H * atomic_mass_ratio) - element_abundance_ratio)

    # Oxygen
    O_H = get_metal_fractions(sw_data, 'O')
    atomic_mass_ratio = atomic_mass['H'] / atomic_mass['O']
    element_abundance_ratio = solar_abundances[abundances_ref]['O'] - solar_abundances[abundances_ref]['H']
    O_H = 10.0 ** (np.log10(O_H * atomic_mass_ratio) - element_abundance_ratio)

    # Neon
    Ne_H = get_metal_fractions(sw_data, 'Ne')
    atomic_mass_ratio = atomic_mass['H'] / atomic_mass['Ne']
    element_abundance_ratio = solar_abundances[abundances_ref]['Ne'] - solar_abundances[abundances_ref]['H']
    Ne_H = 10.0 ** (np.log10(Ne_H * atomic_mass_ratio) - element_abundance_ratio)

    # Magnesium
    Mg_H = get_metal_fractions(sw_data, 'Mg')
    atomic_mass_ratio = atomic_mass['H'] / atomic_mass['Mg']
    element_abundance_ratio = solar_abundances[abundances_ref]['Mg'] - solar_abundances[abundances_ref]['H']
    Mg_H = 10.0 ** (np.log10(Mg_H * atomic_mass_ratio) - element_abundance_ratio)

    # Silicon, Sulphur & Calcium
    Si_H = get_metal_fractions(sw_data, 'Si')
    Ca_Si = 0.0941736
    S_Si = 0.6054160
    AG_Si = solar_abundances[abundances_ref]['Si'] - solar_abundances[abundances_ref]['H']
    AG_Ca = solar_abundances[abundances_ref]['Ca'] - solar_abundances[abundances_ref]['H']
    AG_S = solar_abundances[abundances_ref]['S'] - solar_abundances[abundances_ref]['H']
    Ca_H = 10.0 ** (np.log10((Ca_Si * Si_H) * (atomic_mass['H'] / atomic_mass['Ca'])) - AG_Ca)
    S_H = 10.0 ** (np.log10((S_Si * Si_H) * (atomic_mass['H'] / atomic_mass['S'])) - AG_S)
    Si_H = 10.0 ** (np.log10(Si_H * (atomic_mass['H'] / atomic_mass['Si'])) - AG_Si)

    # Iron
    Fe_H = get_metal_fractions(sw_data, 'Fe')
    AG_Fe = solar_abundances[abundances_ref]['Fe'] - solar_abundances[abundances_ref]['H']
    Fe_H = 10.0 ** (np.log10(Fe_H * (atomic_mass['H'] / atomic_mass['Fe'])) - AG_Fe)

    return {
        'H_H' : H_H,
        'He_H': He_H,
        'C_H' : C_H,
        'N_H' : N_H,
        'O_H' : O_H,
        'Ne_H': Ne_H,
        'Mg_H': Mg_H,
        'Ca_H': Ca_H,
        'S_H' : S_H,
        'Si_H': Si_H,
        'Fe_H': Fe_H,
    }
