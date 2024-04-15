import logging

import numpy as np
import pandas as pd
from numpy import ndarray
from psm_utils.peptidoform import Peptidoform

LOGGER = logging.getLogger(__name__)


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


def get_ccs_shift(
    cal_df: pd.DataFrame, reference_dataset: pd.DataFrame, use_charge_state: int = 2
) -> float:
    """
    Calculate CCS shift factor, i.e. a constant offset,
    based on identical precursors as in reference dataset.

    Parameters
    ----------
    cal_df
        PSMs with CCS values.
    reference_dataset
        Reference dataset with CCS values.
    use_charge_state
        Charge state to use for CCS shift calculation, needs to be [2,4], by default 2.
    return_shift_factor
        CCS shift factor.

    """
    LOGGER.debug(f"Using charge state {use_charge_state} for CCS shift calculation.")

    reference_tmp = reference_dataset[reference_dataset["charge"] == use_charge_state]
    df_tmp = cal_df[cal_df["charge"] == use_charge_state]
    both = pd.merge(
        left=reference_tmp,
        right=df_tmp,
        right_on=["sequence", "charge"],
        left_on=["peptidoform", "charge"],
        how="inner",
        suffixes=("_ref", "_data"),
    )
    LOGGER.debug(
        """Calculating CCS shift based on {} overlapping peptide-charge pairs
        between PSMs and reference dataset""".format(
            both.shape[0]
        )
    )

    # How much CCS in calibration data is larger than reference CCS, so predictions
    # need to be increased by this amount
    return 0 if both.empty else np.mean(both["ccs_observed"] - both["CCS"])


def get_ccs_shift_per_charge(cal_df: pd.DataFrame, reference_dataset: pd.DataFrame) -> ndarray:
    """
    Calculate CCS shift factor per charge state,
    i.e. a constant offset based on identical precursors as in reference.

    Parameters
    ----------
    cal_df
        PSMs with CCS values.
    reference_dataset
        Reference dataset with CCS values.

    Returns
    -------
    ndarray
        CCS shift factors per charge state.

    """
    both = pd.merge(
        left=reference_dataset,
        right=cal_df,
        right_on=["sequence", "charge"],
        left_on=["peptidoform", "charge"],
        how="inner",
        suffixes=("_ref", "_data"),
    )
    return both.groupby("charge").apply(lambda x: np.mean(x["ccs_observed"] - x["CCS"])).to_dict()


def calculate_ccs_shift(
    cal_df: pd.DataFrame, reference_dataset: pd.DataFrame, per_charge=True, use_charge_state=None
) -> float:
    """
    Apply CCS shift to CCS values.

    Parameters
    ----------
    cal_df
        PSMs with CCS values.
    reference_dataset
        Reference dataset with CCS values.
    per_charge
        Whether to calculate shift factor per charge state, default True.
    use_charge_state
        Charge state to use for CCS shift calculation, needs to be [2,4], by default None.

    Returns
    -------
    float
        CCS shift factor.

    """
    cal_df = cal_df[cal_df["charge"] < 7]  # predictions do not go higher for IM2Deep

    if not per_charge:
        shift_factor = get_ccs_shift(
            cal_df,
            reference_dataset,
            use_charge_state=use_charge_state,
        )
        LOGGER.debug(f"CCS shift factor: {shift_factor}")
        return shift_factor

    else:
        shift_factor_dict = get_ccs_shift_per_charge(cal_df, reference_dataset)
        LOGGER.debug(f"CCS shift factor dict: {shift_factor_dict}")
        return shift_factor_dict


def linear_calibration(
    preds_df: pd.DataFrame,
    calibration_dataset: pd.DataFrame,
    reference_dataset: pd.DataFrame,
    per_charge: bool = True,
    use_charge_state: bool = None,
) -> pd.DataFrame:
    """
    Calibrate PSM df using linear calibration.

    Parameters
    ----------
    preds_df
        PSMs with CCS values.
    calibration_dataset
        Calibration dataset with CCS values.
    reference_dataset
        Reference dataset with CCS values.
    per_charge
        Whether to calculate shift factor per charge state, default True.
    use_charge_state
        Charge state to use for CCS shift calculation, needs to be [2,4], by default None.

    Returns
    -------
    pd.DataFrame
        PSMs with calibrated CCS values.

    """
    LOGGER.info("Calibrating CCS values using linear calibration...")
    calibration_dataset['sequence'] = calibration_dataset['peptidoform'].apply(lambda x: x.proforma.split("\\")[0])
    calibration_dataset['charge'] = calibration_dataset['peptidoform'].apply(lambda x: x.precursor_charge)
    # reference_dataset['sequence'] = reference_dataset['peptidoform'].apply(lambda x: x.split('/')[0])
    reference_dataset['charge'] = reference_dataset['peptidoform'].apply(lambda x: int(x.split('/')[1]))

    if per_charge:
        LOGGER.info('Getting general shift factor')
        general_shift = calculate_ccs_shift(
            calibration_dataset,
            reference_dataset,
            per_charge=False,
            use_charge_state=use_charge_state,
        )
        LOGGER.info('Getting shift factors per charge state')
        shift_factor_dict = calculate_ccs_shift(
            calibration_dataset, reference_dataset, per_charge=True
        )

        preds_df['shift'] = preds_df['charge'].map(shift_factor_dict).fillna(general_shift)
        preds_df['predicted_ccs'] = preds_df['predicted_ccs'] + preds_df['shift']

    else:
        shift_factor = calculate_ccs_shift(
            calibration_dataset,
            reference_dataset,
            per_charge=False,
            use_charge_state=use_charge_state,
        )
        preds_df['predicted_ccs'] += shift_factor

    LOGGER.info("CCS values calibrated.")
    return preds_df
