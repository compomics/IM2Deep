from pathlib import Path
import numpy as np

MULTI_BACKBONE_PATH = (
    Path(__file__).parent / "models" / "TIMS_multi" / "Transfer_single_backbone.ckpt"
)


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


multi_config = {
    "model_name": "IM2DeepMulti",
    "batch_size": 16,
    "learning_rate": 0.0001,
    "AtomComp_kernel_size": 4,
    "DiatomComp_kernel_size": 2,
    "One_hot_kernel_size": 2,
    "AtomComp_out_channels_start": 256,
    "DiatomComp_out_channels_start": 128,
    "Global_units": 16,
    "OneHot_out_channels": 2,
    "Concat_units": 128,
    "AtomComp_MaxPool_kernel_size": 2,
    "DiatomComp_MaxPool_kernel_size": 2,
    "Mol_MaxPool_kernel_size": 2,
    "OneHot_MaxPool_kernel_size": 10,
    "LRelu_negative_slope": 0.1,
    "LRelu_saturation": 20,
    "L1_alpha": 0.00001,
    "delta": 0,
    "device": 0,
    "add_X_mol": False,
    "init": "normal",
    "backbone_SD_path": MULTI_BACKBONE_PATH,
}
