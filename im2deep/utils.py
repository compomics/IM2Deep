import numpy as np


def im2ccs(reverse_im, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15):
    """
    Convert ion mobility to collisional cross section.

    Parameters
    ----------
    reverse_im
        Reduced ion mobility.
    mz
        Precursor m/z.
    charge
        Precursor charge.
    mass_gas
        Mass of gas, default 28.013
    temp
        Temperature in Celsius, default 31.85
    t_diff
        Factor to convert Celsius to Kelvin, default 273.15

    Notes
    -----
    Adapted from theGreatHerrLebert/ionmob (https://doi.org/10.1093/bioinformatics/btad486)

    """

    SUMMARY_CONSTANT = 18509.8632163405
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return (SUMMARY_CONSTANT * charge) / (np.sqrt(reduced_mass * (temp + t_diff)) * 1 / reverse_im)


def ccs2im(ccs, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15):
    """
    Convert collisional cross section to ion mobility.

    Parameters
    ----------
    ccs
        Collisional cross section.
    mz
        Precursor m/z.
    charge
        Precursor charge.
    mass_gas
        Mass of gas, default 28.013
    temp
        Temperature in Celsius, default 31.85
    t_diff
        Factor to convert Celsius to Kelvin, default 273.15

    Notes
    -----
    Adapted from theGreatHerrLebert/ionmob (https://doi.org/10.1093/bioinformatics/btad486)

    """

    SUMMARY_CONSTANT = 18509.8632163405
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return ((np.sqrt(reduced_mass * (temp + t_diff))) * ccs) / (SUMMARY_CONSTANT * charge)
